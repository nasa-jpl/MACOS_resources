%S3_SEGMENTATION  Stage 3: segment the e2e primary -- PIE and 2-ring HEX.
%
% Stage 3 of the end-to-end worked example -- now a THIN DRIVER over
% the general stage runner design/runners/run_segmentation.m (the
% runners doctrine, Dave 2026-07-19).  Segments M1 of the stage-2
% system (parent = s2_instrument.in, element 1) TWO ways, keeping both
% as artifacts (Dave 2026-07-18):
%
%   pie   1-ring PIE -> 7 segments: a center HEXAGON + 6 wedges whose
%         inner edges abut it along straight chords  ->  e2e_pie.in
%   hex2  2-ring HEX -> 19 hexagonal segments        ->  e2e_hex2.in
%
% Segment geometry SCALES WITH THE APERTURE (SegMirMaker default size
% = Aperture/(2*rings+1)); the explicit knob that scales with it is
% the gap (P.seg.gap_mm; this family is BaseUnits=m, so metres go in).
% Per variant the runner does: bare-splice parity vs the parent,
% engine-truth footprints, physical polygonal apertures (gap rays
% clip -- physically honest), view_std views, and a standalone reload
% verification of the artifact.  The parent's M1 freeform figure rides
% on every segment; the M1 hole carries onto the center segment.
%
% P.seg.variant picks which artifact stages 4-6 consume (one knob;
% both .in files always exist).  Summary -> s3_report.txt.
%
% Run AFTER s2_instrument.m.

addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/runners'));
P = e2e_params();
here  = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
parent = fullfile(here, 's2_instrument.in');
assert(isfile(parent), 's3 needs s2_instrument.in -- run s2_instrument.m first');

gap_m = P.seg.gap_mm * 1e-3;             % BaseUnits=m in this family
vs = P.seg.variants;
log_ = fopen(fullfile(here, 's3_report.txt'), 'w');
say = @(varargin) fprintf(1, varargin{:}) + fprintf(log_, varargin{:});
say('==== e2e stage 3: segmentation of M1 (parent s2_instrument.in) ====\n');
say('parent: D = %g m, BaseUnits=m; gap %g mm; per-variant detail in\n', ...
    P.D_m, P.seg.gap_mm);
say('e2e_<variant>_seg_report.txt (run_segmentation runner reports)\n\n');

S = struct();                            % per-variant results for the .mat
for iv = 1:numel(vs)
    v = char(vs(iv));
    switch v
        case 'pie',  rings = 1; grid = 'Pie';
        case 'hex2', rings = 2; grid = 'Hex';
        otherwise, error('unknown segmentation variant "%s"', v);
    end
    say('---- variant "%s": rings=%d, grid=%s ----\n', v, rings, grid);
    a = run_segmentation(parent, 'rings', rings, 'grid', grid, ...
        'elt', 1, 'gap', gap_m, 'dofs', P.seg.dofs, ...
        'meas_config', P.seg.meas_config, 'grid_npts', P.seg.grid_npts, ...
        'emit_apertures', P.seg.emit_apertures, 'ap_pad', P.seg.ap_pad, ...
        'model_size', P.seg.model_size, ...
        'out_dir', here, 'name', sprintf('e2e_%s', v));
    assert(a.verified, 's3 variant %s failed standalone reload verification', v);
    % stage-prefixed figure names (e2e convention: s<N>_<what>_<variant>)
    movefile(char(a.footprints_png), ...
             fullfile(here, sprintf('s3_footprints_%s.png', v)));
    if strlength(a.views_png) > 0
        movefile(char(a.views_png), ...
                 fullfile(here, sprintf('s3_views_%s.png', v)));
    end
    S.(v) = a;
    say('    -> e2e_%s.in VERIFIED (%d/%d pass, %d clipped)\n\n', v, ...
        a.pass_ap, a.src, a.pass_bare - a.pass_ap);
end

say('==== stage-3 summary ====\n');
say('%-6s %5s %9s %10s %9s %12s\n', 'name','nseg','width_m','pass/src','clipped','rmsWFE d%');
for iv = 1:numel(vs)
    v = char(vs(iv)); q = S.(v);
    say('%-6s %5d %9.4g %6d/%-5d %8d %+11.2f\n', v, q.seg.nseg, q.seg.width, ...
        q.pass_ap, q.src, q.pass_bare - q.pass_ap, ...
        100*(q.rms_ap - q.rms_parent)/q.rms_parent);
end
say('(rmsWFE deltas vs parent = pupil-sampling coverage: the segment tilings\n');
say(' sample the pupil differently from the parent Circular grid)\n');
say('downstream variant (P.seg.variant) = "%s" -> stages 4-6 consume e2e_%s.in\n', ...
    P.seg.variant, P.seg.variant);
fclose(log_);
save(fullfile(here, 's3_segmentation.mat'), 'P', 'S', 'parent');
fprintf('\nStage 3 complete: e2e_pie.in + e2e_hex2.in + s3_report.txt + figures.\n');
fprintf('Next: s4_jacobians.m (dwdx / dwdz / dwdgrid on e2e_%s.in).\n', P.seg.variant);
