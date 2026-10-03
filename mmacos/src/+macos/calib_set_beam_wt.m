function calib_set_beam_wt(wt, centroid)
%MACOS.CALIB_SET_BEAM_WT  Weight of the CALIB beam rows, and the centroid option.
%   macos.calib_set_beam_wt(WT) weights every beam row (calib_set_beam)
%   against the WFE / SPOT rows: the row's sigma is divided by sqrt(WT), so
%   WT = 1e6 makes a 1 um position error count as a 1 mm one.  Mind the
%   units: WFE rows are in the target's units, SPOT rows and beam rows in
%   base units.
%   macos.calib_set_beam_wt(WT, true) makes the beam position the CENTROID
%   of the rays that pass the train, not the chief ray (OptBeamCentroid= Y).
%   Mirrors the Rx keywords OptBeamWt= / OptBeamCentroid=.
if nargin < 2, centroid = false; end
w = double(wt);
if w <= 0
    error('macos:calib_set_beam_wt:bad', 'wt must be > 0 (got %g)', w);
end
mmacos('calib_set_beam_wt', w, double(logical(centroid)));
end
