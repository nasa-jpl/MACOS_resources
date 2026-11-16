function chans = rigid_body_channels(session, rx_path, opts)
%MACOS.CHANNELS.RIGID_BODY_CHANNELS  Build per-element rigid-body channels.
%   chans = macos.channels.rigid_body_channels(SESSION, RX_PATH)
%   returns a cell array of channel handles (RigidBodyChannel or
%   FocalPlaneChannel), one per (actual optic, DOF) pair.
%
%   "Actual optic" = any Element= kind except Reference / Return.
%   FocalPlane elements get a FocalPlaneChannel; everything else
%   gets the plain RigidBodyChannel.
%
%   Name-value pairs:
%     'elts'        column vector of element ids to restrict the
%                   eligibility set to.  Default: all actual optics.
%     'dofs'        vector of DOF indices (0..5) to emit.  Default:
%                   all 6 in DOF order Rx,Ry,Rz,Tx,Ty,Tz.
%     'fp_mode'     'track' (default) | 'srs' | 'sxp' | 'none' --
%                   forwarded to FocalPlaneChannel.
%     'ep_elt'      EP element id for fp_mode={track,srs,sxp}.
%                   Default -1 = nElt-1 (the standard XP convention).
%     'include_non_optics'  logical -- include Reference / Return
%                           too (default false).
%
%   See also: macos.channels.RigidBodyChannel,
%             macos.channels.FocalPlaneChannel,
%             macos.channels.source_channels.

arguments
    session
    rx_path                  (1,:) char {mustBeNonempty}
    opts.elts                (:,1) double = []
    opts.dofs                (:,1) double = (0:5).'
    opts.fp_mode             (1,:) char {mustBeMember( ...
        opts.fp_mode, {'track','srs','sxp','none'})} = 'track'
    opts.ep_elt              (1,1) double {mustBeInteger} = -1
    opts.include_non_optics  (1,1) logical = false
end

% Discover eligible elements + their kinds from the Rx text.
elt_kinds = parse_rx_actual_optic_elts_(rx_path, ...
    opts.include_non_optics);

target_elts = sort(double(cell2mat(keys(elt_kinds))));
target_elts = require_elts(target_elts(:), opts.elts, 'rigid-body', ...
    @(id) rb_reason(session, id));

if isempty(target_elts)
    chans = {};
    return;
end

chans = cell(0, 1);
for k = 1:numel(target_elts)
    iElt = target_elts(k);
    kind = elt_kinds(int32(iElt));
    for jj = 1:numel(opts.dofs)
        d = double(opts.dofs(jj));
        if strcmp(kind, 'FocalPlane')
            ch = macos.channels.FocalPlaneChannel(session, iElt, d, ...
                'mode', opts.fp_mode, ...
                'ep_elt', opts.ep_elt);
        else
            ch = macos.channels.RigidBodyChannel(session, iElt, d);
        end
        chans{end+1, 1} = ch; %#ok<AGROW>
    end
end
end

function r = rb_reason(session, id)
% Why a requested element is not in the rigid-body eligibility set.
    n = session.num_elt();
    if id < 1 || id ~= round(id) || id > n
        r = sprintf('element id out of range (nElt = %d)', n);
        return;
    end
    t = macos.get_elt_info(id).type;
    if any(strcmp(t, {'Reference','Return'}))
        r = sprintf(['Element= %s is excluded from rigid-body channels ' ...
                     '(pass ''include_non_optics'', true to include it)'], t);
    else
        r = sprintf('Element= %s not in the discovered eligibility set -- report this', t);
    end
end
