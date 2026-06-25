function save_dw_multi(out, model_size, matpath)
%SAVE_DW_MULTI  Write a dw_d*_multi result in the canonical .mat layout.
%   Matches examples/sensitivities/.../run_dwdz.m so the shared verifyall.m
%   lifecycle (m2v / v2m / indxall round-trip) consumes it directly:
%
%       wall = dwdxall * x + w0_stacked
%
%   OUT         struct from any macos.dw_d*_multi call (dwdxall is the
%               canonical alias every supervisor sets).
%   MODEL_SIZE  the macos.Session model size (recorded for verifyall).
%   MATPATH     output .mat path.
dwdxall       = out.dwdxall;        %#ok<NASGU>
w0_stacked    = out.w0_stacked;     %#ok<NASGU>
indxall       = out.indxall;        %#ok<NASGU>
channel_names = out.channel_names;  %#ok<NASGU>
field_table   = out.field_table;    %#ok<NASGU>
field_names   = out.field_names;    %#ok<NASGU>
chfraydir_nom = out.chfraydir_nom;  %#ok<NASGU>
delta         = out.delta;          %#ok<NASGU>
method        = out.method;         %#ok<NASGU>
wf_elt        = out.wf_elt;         %#ok<NASGU>
rx            = out.rx_path;         %#ok<NASGU>
opdall_shape  = size(out.OPDall);   %#ok<NASGU>
save(matpath, 'dwdxall', 'w0_stacked', 'indxall', 'channel_names', ...
    'field_table', 'field_names', 'chfraydir_nom', 'delta', 'method', ...
    'wf_elt', 'rx', 'opdall_shape', 'model_size');
fprintf('wrote %s\n', matpath);
end
