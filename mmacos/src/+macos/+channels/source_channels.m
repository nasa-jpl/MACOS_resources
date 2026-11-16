function chans = source_channels(session, opts)
%MACOS.CHANNELS.SOURCE_CHANNELS  Build SourceChannel instances.
%   chans = macos.channels.source_channels(SESSION) returns a cell
%   array of SourceChannel handles, one per DOF (Rx,Ry,Rz,Tx,Ty,Tz)
%   in DOF order.
%
%   Name-value pairs:
%     'dofs'         vector of DOF indices (0..5).  Default all 6.
%     'stop_mode'    'obj' (default) | 'elt' | 'none'.
%     'stop_obj_pos' 1x3 object-space stop coordinates (default [0 0 0],
%                    matches ApStop= 0 0 0 in many Rxes).
%     'stop_elt'     element id when stop_mode='elt'.
%
%   See also: macos.channels.SourceChannel,
%             macos.channels.rigid_body_channels.

arguments
    session
    opts.dofs         (:,1) double = (0:5).'
    opts.stop_mode    (1,:) char {mustBeMember(opts.stop_mode, ...
                          {'obj','elt','none'})} = 'obj'
    opts.stop_obj_pos (1,3) double = [0 0 0]
    opts.stop_elt     (1,1) double {mustBeInteger} = 0
end

chans = cell(numel(opts.dofs), 1);
for k = 1:numel(opts.dofs)
    d = double(opts.dofs(k));
    chans{k} = macos.channels.SourceChannel(session, d, ...
        'stop_mode', opts.stop_mode, ...
        'stop_obj_pos', opts.stop_obj_pos, ...
        'stop_elt', opts.stop_elt);
end
end
