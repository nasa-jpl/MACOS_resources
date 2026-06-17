function chans = grouped_rigid_body_channels(session, groups, opts)
%MACOS.CHANNELS.GROUPED_RIGID_BODY_CHANNELS  Build group channels.
%   chans = macos.channels.grouped_rigid_body_channels(SESSION, GROUPS)
%   returns a cell array of GroupedRigidBodyChannel handles, one per
%   (group, DOF) pair.
%
%   GROUPS is a containers.Map (char -> column vector of double):
%     keys()   = group names (char arrays)
%     values() = column vector of member element ids
%
%   Name-value pairs:
%     'ref_elt_by_group'  containers.Map name -> int (optional)
%                         Default: first member.
%     'dofs'              vector of DOF indices 0..5.  Default all 6.
%     'rx_path'           if given, parse the Rx to auto-detect a
%                         FocalPlane-typed element among each group's
%                         members; that element becomes the group's
%                         fp_elt (enables 'auto' fp_mode dispatch).
%     'fp_mode'           propagated to each channel.  Default 'auto'.
%     'ep_elt'            propagated.  Default -1.
%     'fp_elt_by_group'   per-group explicit FP override; supersedes
%                         Rx-based detection.
%     'coords'            'global' (default) | 'local'.
%     'stop_mode'         'obj' (default) | 'elt' | 'none'.
%     'stop_obj_pos'      1x3 default [0 0 0].
%     'stop_elt'          element id when stop_mode='elt'.
%
%   See also: macos.channels.GroupedRigidBodyChannel,
%             macos.channels.parse_rx_groups.

arguments
    session
    groups                    {mustBeA(groups, 'containers.Map')}
    opts.ref_elt_by_group     = []
    opts.dofs                 (:,1) double = (0:5).'
    opts.rx_path              (1,:) char = ''
    opts.fp_mode              (1,:) char {mustBeMember(opts.fp_mode, ...
                                  {'auto','none','sxp','srs'})} = 'auto'
    opts.ep_elt               (1,1) double {mustBeInteger} = -1
    opts.fp_elt_by_group      = []
    opts.coords               (1,:) char {mustBeMember(opts.coords, ...
                                  {'global','local'})} = 'global'
    opts.stop_mode            (1,:) char {mustBeMember(opts.stop_mode, ...
                                  {'obj','elt','none'})} = 'obj'
    opts.stop_obj_pos         (1,3) double = [0 0 0]
    opts.stop_elt             (1,1) double {mustBeInteger} = 0
end

% Discover FP elements from the Rx text, if a path was given.
fp_elts = zeros(0, 1);
if ~isempty(opts.rx_path)
    elt_kinds = parse_rx_actual_optic_elts_(opts.rx_path, false);
    k = keys(elt_kinds);
    for ii = 1:numel(k)
        if strcmp(elt_kinds(k{ii}), 'FocalPlane')
            fp_elts(end+1, 1) = double(k{ii}); %#ok<AGROW>
        end
    end
end

chans = cell(0, 1);
gnames = keys(groups);
for gi = 1:numel(gnames)
    nm = gnames{gi};
    mems = groups(nm);
    mems = mems(:);
    if numel(mems) < 2
        continue;
    end

    ref = 0;
    if isa(opts.ref_elt_by_group, 'containers.Map') ...
            && isKey(opts.ref_elt_by_group, nm)
        ref = double(opts.ref_elt_by_group(nm));
    end
    if ref == 0
        ref = mems(1);
    end

    fp = 0;
    if isa(opts.fp_elt_by_group, 'containers.Map') ...
            && isKey(opts.fp_elt_by_group, nm)
        fp = double(opts.fp_elt_by_group(nm));
    end
    if fp == 0 && ~isempty(fp_elts)
        in_grp = intersect(mems, fp_elts);
        if ~isempty(in_grp)
            fp = double(in_grp(1));
        end
    end

    for jj = 1:numel(opts.dofs)
        d = double(opts.dofs(jj));
        chans{end+1, 1} = macos.channels.GroupedRigidBodyChannel( ...
            session, mems, d, ...
            'group_name', nm, ...
            'ref_elt', ref, ...
            'fp_elt', fp, ...
            'fp_mode', opts.fp_mode, ...
            'ep_elt', opts.ep_elt, ...
            'coords', opts.coords, ...
            'stop_mode', opts.stop_mode, ...
            'stop_obj_pos', opts.stop_obj_pos, ...
            'stop_elt', opts.stop_elt); %#ok<AGROW>
    end
end
end
