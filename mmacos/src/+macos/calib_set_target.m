function calib_set_target(target, varargin)
%MACOS.CALIB_SET_TARGET  Set the CALIB optimization target.
%   macos.calib_set_target(NAME) selects one of:
%     'WFE'         -- RMS wavefront error
%     'WFE_ZMODE'   -- specific Zernike modes (alias 'ZWF')
%     'BEAM'        -- beam waist / position / divergence
%     'SPOT'        -- RMS spot size
%     'OPL'         -- optical path length
%
%   macos.calib_set_target('WFE_ZMODE', wf_zern_modes) additionally
%   passes the list of Zernike modes (1..45) the optimizer should
%   drive to zero.
%
%   Integer enum is also accepted (1..5 mirroring dopt_mod's
%   *_TARGET constants).
%
%   Example:
%     m.calib_set_target('WFE')
%     m.calib_set_target('WFE_ZMODE', [4 5 11])
TARGETS = struct( ...
    'WFE',        1, ...
    'WFE_ZMODE',  2, ...
    'ZWF',        2, ...
    'BEAM',       3, ...
    'SPOT',       4, ...
    'OPL',        5);
if ischar(target) || isstring(target)
    key = upper(strtrim(char(target)));
    if ~isfield(TARGETS, key)
        error('macos:calib_set_target:badName', ...
              'unknown target name %s; valid: WFE, WFE_ZMODE, ZWF, BEAM, SPOT, OPL', ...
              key);
    end
    t = TARGETS.(key);
else
    t = double(target);
end
if isempty(varargin)
    wf_zern_count = 0;
    wf_zern_arg = double(0);  % dummy 1-elem; Fortran ignores when count=0
else
    wf_zern = double(varargin{1}(:));
    wf_zern_count = numel(wf_zern);
    if wf_zern_count == 0
        wf_zern_arg = double(0);
    else
        wf_zern_arg = wf_zern;
    end
end
mmacos('calib_set_target', t, wf_zern_arg, double(wf_zern_count));
end
