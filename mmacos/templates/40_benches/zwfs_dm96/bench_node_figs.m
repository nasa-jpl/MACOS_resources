function out = bench_node_figs(varargin)
%BENCH_NODE_FIGS  The splitter-angle comparison figures: 7, 22.5 and 30 deg.
%   bench_node_figs() renders bench_bs7.png, bench_bs22.png and bench_bs30.png
%   with dmg_bench_clearance -- each a two-panel figure, the whole train on top
%   and the node (+-320 mm about the splitter) below -- from zwfs_params' bench
%   block as it stands, so the substrates and the beam of the day are in them.
%
%   These three are the deck's bench-layout and splitter-angle figures
%   (crop_panels.py cuts the train and node panels out of them).  They were
%   rendered by hand until 2026-09-18 and the invocation was never committed,
%   so a rebuild could not reproduce them and they silently went stale: the
%   set in the deck predated the substrate decision by two days and showed a
%   bench with no glass on its polarizers, mask or splitter.  Hence this file.
%
%   Options (name/value):
%     'angles'  splitter AOIs to render (default [7 22.5 30]); the file name
%               is bench_bs<AOI>.png with the decimal point dropped, which is
%               the naming crop_panels.py already expects (22.5 -> bs22).
%     'outdir'  where the PNGs land (default: this directory).
%     any other name/value pair is passed through to dmg_bench_clearance, so
%     'D_BS_CMP', 225 renders the compensator move Dave has under decision.
%
%   Returns a struct array with each angle's clearance table, so the figure and
%   the number that justifies it come from ONE call.

here = fileparts(mfilename('fullpath'));
% mmacos_setup does not put the gauge library on the path -- the bench runners
% add it themselves -- so a driver that calls dmg_bench_clearance directly has
% to do the same or it dies at the first render with "Unrecognized function".
addpath(fullfile(here, '..', 'dm_gauge_lib'));
% THE SEAT.  zwfs_params carries bench.MASK_TRIM = 'scan' since 2026-09-17:
% the runner re-finds the focus at run time, because the collimation fix moved
% it and a carried constant seats the mask off focus and then blames the glass.
% A LAYOUT DRAWING cannot run that scan -- the scan is a runner stage -- and the
% builder needs a scalar, so these figures use the solved lens-rig seat that
% tg96_params records (1.231759 mm).  It is the right number for this bench and
% it moves nothing at drawing scale: a millimetre of seat on a two-metre bench
% is invisible in the layout and changes no clearance in the table.  Pass
% 'MASK_TRIM', <mm> to draw a different one.
o = struct('angles', [7 22.5 30], 'outdir', here, 'MASK_TRIM', 1.231759);
pass = {};
for i = 1:2:numel(varargin)
    if isfield(o, varargin{i}), o.(varargin{i}) = varargin{i+1};
    else, pass(end+1:end+2) = varargin(i:i+1); end %#ok<AGROW>
end

out = struct('aoi', {}, 'png', {}, 'worst', {}, 'nbad', {});
for k = 1:numel(o.angles)
    a = o.angles(k);
    png = fullfile(o.outdir, sprintf('bench_bs%d.png', floor(a)));
    fprintf('\n=== node figure at %g deg -> %s\n', a, png);
    r = dmg_bench_clearance('BS_AOI', a, 'MASK_TRIM', o.MASK_TRIM, 'draw', png, pass{:});
    % column 6 of rows is the clearance in mm; NaN = no other beam crosses
    % that part's plane, which is not a clearance of zero
    v = cell2mat(r.rows(:, 6));  v = v(~isnan(v));
    out(end+1) = struct('aoi', a, 'png', png, ...
                        'worst', min(v), 'nbad', sum(v < 0)); %#ok<AGROW>
end

fprintf('\n  AOI    worst clearance    parts in another beam\n');
for k = 1:numel(out)
    fprintf('  %4.1f      %8.1f mm      %d\n', out(k).aoi, out(k).worst, out(k).nbad);
end
fprintf(['\nNow rebuild the deck crops: python3 crop_panels.py  (it cuts the\n' ...
         'train and node panels from these and tiles bench_three_nodes.png)\n']);
end
