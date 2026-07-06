function apply_ngridpts(session, n, who)
%APPLY_NGRIDPTS  Override the source ray-grid sampling (nGridPts).
%   Shared by the dw_d* sensitivity wrappers ('ngridpts' option).
%   Empty n = keep the .in-file value (no-op).  The engine clamps to
%   [3, model-size limit] and re-runs MODIFY itself, so the override
%   takes effect on the next trace; it persists until the next
%   load_rx.  Warns (does not error) when the request was clamped.
if isempty(n), return; end
session.set_src_sampling(n);
ng = session.get_src_sampling();
if ng ~= n
    warning(['macos:' who ':ngridpts'], ...
        'ngridpts=%d clamped to %d by the engine (3..model-size limit)', ...
        n, ng);
end
fprintf('[setup] ray-grid sampling override: nGridPts = %d\n', ng);
end
