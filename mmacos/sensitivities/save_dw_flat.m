function save_dw_flat(out, matpath, opts)
%SAVE_DW_FLAT  Write a dw_d*_multi result as a FLAT, channel-named .mat.
%   save_dw_flat(OUT, MATPATH) writes the Jacobian and its bookkeeping at
%   the TOP LEVEL of the .mat -- no wrapper struct -- with the channel's
%   OWN name (dwdx / dwdz / dwdsurf / dwdgrid), so a downstream `load`
%   sees e.g.
%
%       load('..._dwdgrid.mat')   % -> dwdgrid, w0_stacked, indxall, ...
%
%   rather than the generic `dwdxall` alias buried in an `og` struct.
%   Contrast run_sensitivities' own <name>_sens.mat, which nests ox/oz/
%   og/os for the pipeline (run_simulator / run_compare) and is left as
%   is.  This saver is what the zoom_5x5 drivers hand the user.
%
%   The channel name is inferred from which Jacobian field OUT carries
%   (dwdxall/dwdzall/dwdsall/dwdgall) unless 'name' overrides it.  Only
%   NON-EMPTY fields are written (no empty ox/og clutter, no absent
%   config table on a single-configuration run).
%
%   OUT       struct from any macos.dw_d*_multi call.
%   MATPATH   output .mat path.
%   OPTIONS
%     'name'        channel variable name ('dwdx'|'dwdz'|'dwdsurf'|
%                   'dwdgrid'); default inferred from OUT's fields.
%     'model_size'  engine model size, recorded for verifyall ([] = skip).
%     'extra'       struct of additional top-level variables to save
%                   (e.g. the dwdgrid influence basis: struct('sgb', sgb)).
%
%   See also: run_sensitivities, macos.dw_dx_multi, save_dw_multi.

arguments
    out (1,1) struct
    matpath (1,1) string
    opts.name (1,:) char = ''
    opts.model_size double = []
    opts.extra struct = struct()
end

% ---- channel name + Jacobian ------------------------------------------
JAC = struct('dwdx','dwdxall', 'dwdz','dwdzall', ...
             'dwdsurf','dwdsall', 'dwdgrid','dwdgall');
name = opts.name;
if isempty(name)
    % infer from the channel-specific field OUT carries (the generic
    % dwdxall alias is set by every supervisor, so check it LAST)
    if     isfield(out,'dwdgall'), name = 'dwdgrid';
    elseif isfield(out,'dwdsall'), name = 'dwdsurf';
    elseif isfield(out,'dwdzall'), name = 'dwdz';
    else,                          name = 'dwdx';
    end
end
assert(isfield(JAC, name), 'save_dw_flat: unknown channel name ''%s''', name);
jfield = JAC.(name);
assert(isfield(out, jfield), ...
    'save_dw_flat: OUT has no %s field for channel %s', jfield, name);

S = struct();
S.(name) = out.(jfield);          % the channel-named Jacobian at top level

% ---- flat bookkeeping: copy through only the fields that exist + are
%      non-empty (request: no empty structs / absent-run clutter) -------
copy = {'w0_stacked','indxall','channel_names','field_table', ...
        'field_names','chfraydir_nom','delta','method','wf_elt', ...
        'config_names','config_table','iElt','kind','dof_idx', ...
        'map_idx','zmodes','sgb'};
for f = copy
    if isfield(out, f{1}) && ~isempty(out.(f{1}))
        S.(f{1}) = out.(f{1});
    end
end
if isfield(out, 'rx_path') && ~isempty(out.rx_path)
    S.rx = out.rx_path;           % 'rx' name matches save_dw_multi/verifyall
end
if isfield(out, 'OPDall') && ~isempty(out.OPDall)
    S.opdall_shape = size(out.OPDall);
end
if ~isempty(opts.model_size)
    S.model_size = opts.model_size;
end
for f = fieldnames(opts.extra).'   % caller extras (e.g. sgb) win
    S.(f{1}) = opts.extra.(f{1});
end

save(matpath, '-struct', 'S', '-v7.3');
fprintf('wrote %s (%s + %d flat fields)\n', matpath, name, numel(fieldnames(S))-1);
end
