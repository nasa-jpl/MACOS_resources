function ox = dwdx_trans_per_metre(ox)
%MACOS.DWDX_TRANS_PER_METRE  Put a dw_dx harvest's translation columns per SI metre.
%   OX = macos.dwdx_trans_per_metre(OX) takes a macos.dw_dx / dw_dx_multi
%   result (live, or loaded from a jac .mat) and returns it with every
%   TRANSLATION column (dof_idx 3..5) and the matching dcdx row expressed
%   per SI METRE of translation, the convention the design runners
%   (run_compare, run_simulator, run_met) and jacobian_check work in.
%
%   - ox.trans_output == 'base' (the dw_dx default since 2026-10-06):
%     those columns / rows are divided by ox.cbm (metres per BaseUnit).
%   - ox.trans_output == 'si', or the field ABSENT (every harvest written
%     before 2026-10-06, which was per metre): IDENTITY, bit for bit.
%   Rotation columns (per rad) and the OPD numerator (BaseUnits) are never
%   touched.  On return ox.trans_output = 'si', so a second call is a no-op.
%
%   Fields converted when present: dwdx, dwdxall, per_field_dwdx (cell,
%   any shape), dcdx, dcdx_per_field (cell).  Column selection is
%   ox.dof_idx >= 3 -- the same rule dw_dx scales by.
%
%   See also: macos.dw_dx, macos.dw_dx_multi.

if ~isstruct(ox) || ~isfield(ox, 'trans_output') ...
        || ~strcmp(ox.trans_output, 'base')
    return
end
assert(isfield(ox, 'cbm') && ox.cbm > 0, ...
    'macos:dwdx_trans_per_metre:cbm', ...
    'trans_output=''base'' harvest carries no cbm -- cannot convert');
assert(isfield(ox, 'dof_idx'), 'macos:dwdx_trans_per_metre:dof', ...
    'harvest carries no dof_idx -- cannot locate the translation columns');
t = reshape(ox.dof_idx >= 3, 1, []);
f = 1 / ox.cbm;
for nm = {'dwdx', 'dwdxall'}
    if isfield(ox, nm{1}) && ~isempty(ox.(nm{1}))
        ox.(nm{1})(:, t) = ox.(nm{1})(:, t) * f;
    end
end
if isfield(ox, 'per_field_dwdx')
    for k = 1:numel(ox.per_field_dwdx)
        ox.per_field_dwdx{k}(:, t) = ox.per_field_dwdx{k}(:, t) * f;
    end
end
if isfield(ox, 'dcdx') && ~isempty(ox.dcdx)
    ox.dcdx(t, :) = ox.dcdx(t, :) * f;
end
if isfield(ox, 'dcdx_per_field')
    for k = 1:numel(ox.dcdx_per_field)
        if ~isempty(ox.dcdx_per_field{k})
            ox.dcdx_per_field{k}(t, :) = ox.dcdx_per_field{k}(t, :) * f;
        end
    end
end
ox.trans_output = 'si';
end
