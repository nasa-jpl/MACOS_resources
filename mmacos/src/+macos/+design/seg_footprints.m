function labels = seg_footprints(seg, w0, opts)
%SEG_FOOTPRINTS  Per-segment TRUE ray footprints from the trace.
%   LABELS = macos.design.seg_footprints(SEG, W0) measures which OPD
%   pixels each segment of SEG (= macos.design.segment_rx output)
%   actually owns, by ENGINE TRUTH rather than geometry: poke one
%   segment at a time (local +z translation), re-trace, and mark the
%   OPD pixels that move.  OPD is piston-removed, so the poked
%   segment's own pixels sit at a distinct level; pixels are split on
%   deviation from the median.  This is the measurement the e5_pie /
%   e2e-s3 footprint figures are built on -- shared product code, not
%   per-example script (Dave: general-purpose runners).
%
%   Requires the engine loaded with SEG.in and W0 = macos.opd() at the
%   nominal state.  Restores the nominal Rx (load_rx(seg.in)) before
%   returning.  LABELS is N x N (OPD grid), 0 = no segment.
%
%   Options:
%     'poke'    local +z translation magnitude, parent BASE UNITS
%               (default 1e-6: right for mm-unit parents; metre-unit
%               parents want ~1e-7 -- any value that dominates the
%               nominal WFE without clipping works)
%     'thresh'  deviation split as a fraction of the max deviation
%               (default 0.25)
%
%   See also: macos.design.seg_boundary, macos.design.seg_footprint_view.
arguments
    seg (1,1) struct
    w0  (:,:) double
    opts.poke   (1,1) double = 1e-6
    opts.thresh (1,1) double = 0.25
    opts.mask   = []   % valid-ray mask of w0; default = the engine's (opd_mask) for the trace that made w0
end
N = size(w0, 1);
% the pupil from the engine's valid-ray mask, not w0 ~= 0: under the
% chief-ray reference (the default since 2026-10-10) a valid ray at the
% chief's path reads exactly 0.  The caller has just traced w0.
if isempty(opts.mask), opts.mask = macos.opd_mask(); end
if ~isequal(size(opts.mask), size(w0)), opts.mask = (w0 ~= 0); end
labels = zeros(N);
for s = 1:seg.nseg
    macos.load_rx(seg.in);
    macos.perturb(seg.seg_elts(s), 'translation', [0;0;opts.poke], ...
                  'frame', 'local');
    macos.modify(); macos.trace();
    d = macos.opd() - w0;
    ok = isfinite(d) & opts.mask;
    dev = abs(d - median(d(ok)));
    labels(ok & dev > opts.thresh*max(dev(ok))) = s;
end
macos.load_rx(seg.in);                            % restore nominal
end
