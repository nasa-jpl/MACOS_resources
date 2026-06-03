function calib_set_var_elt(srf, varargin)
%MACOS.CALIB_SET_VAR_ELT  Mark an element as a CALIB variable.
%   macos.calib_set_var_elt(SRF, dof_name1, dof_name2, ...) marks
%   element SRF as a variable with the named DOFs free.  Valid names
%   (case-insensitive): TIP, TILT, CLOCK, DX, DY, PIST, ROC, CONIC.
%
%   macos.calib_set_var_elt(SRF, mask) accepts an 8-int positional
%   mask [TIP TILT CLOCK DX DY PIST ROC CONIC]; nonzero = vary.
%
%   macos.calib_set_var_elt(..., 'ZernModes', [1 4 5 11]) additionally
%   marks Zernike modes 1, 4, 5, 11 as free on this element.
%
%   If SRF was already a variable, this call REPLACES its DOF +
%   Zernike configuration in place (MVAR semantics).
%
%   Examples:
%     macos.calib_set_var_elt(7, 'TIP', 'TILT')
%     macos.calib_set_var_elt(3, 'DX', 'DY', 'PIST', ...
%                             'ZernModes', [4 5 11])
%     macos.calib_set_var_elt(7, [1 1 0 0 0 0 0 0])   % positional mask
%
%   See also: macos.calib, macos.calib_clear_var_elts.
DOF_NAMES = {'TIP', 'TILT', 'CLOCK', 'DX', 'DY', 'PIST', 'ROC', 'CONIC'};

% Pull out optional 'ZernModes' name/value pair if present.
zern_modes = double([]);
keep = true(1, numel(varargin));
i = 1;
while i <= numel(varargin)
    arg = varargin{i};
    if (ischar(arg) || isstring(arg)) && strcmpi(arg, 'ZernModes')
        if i+1 > numel(varargin)
            error('macos:calib_set_var_elt:badPair', ...
                  '''ZernModes'' must be followed by a vector of modes');
        end
        zern_modes = double(varargin{i+1}(:));
        keep([i i+1]) = false;
        i = i + 2;
    else
        i = i + 1;
    end
end
dof_args = varargin(keep);

% Build the 8-int positional mask.
mask = zeros(1, 8);
if numel(dof_args) == 1 && (isnumeric(dof_args{1}) || islogical(dof_args{1}))
    v = double(dof_args{1}(:).');
    if numel(v) ~= 8
        error('macos:calib_set_var_elt:badMask', ...
              'positional dofs must be length 8 (got %d)', numel(v));
    end
    mask = double(v ~= 0);
else
    for k = 1:numel(dof_args)
        nm = upper(strtrim(char(dof_args{k})));
        idx = find(strcmp(nm, DOF_NAMES), 1);
        if isempty(idx)
            error('macos:calib_set_var_elt:badName', ...
                  'unknown DOF name %s; valid: %s', ...
                  nm, strjoin(DOF_NAMES, ', '));
        end
        mask(idx) = 1;
    end
end

% Ensure non-empty array for mex (Fortran shape derived from numel).
if isempty(zern_modes)
    zern_modes_arg = double(0);
else
    zern_modes_arg = double(zern_modes(:));
end
mmacos('calib_set_var_elt', double(srf), mask, zern_modes_arg, ...
       double(numel(zern_modes)));
end
