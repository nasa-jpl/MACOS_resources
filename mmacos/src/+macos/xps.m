function xps(iElt)
%MACOS.XPS  eXit-Pupil Surface -- per-ray exit-pupil crossing cloud.
%   macos.xps(IELT) runs the engine XPS command (FEX generalized to the
%   full ray grid): it crosses each ray's nominal exit ray with its
%   field-differential partner, leaving the per-ray exit-pupil crossing
%   point in the ray buffer.  Read the cloud IMMEDIATELY afterwards with
%   macos.get_ray_info(N) -- the .pos field is the exit-pupil SURFACE,
%   the chief ray (index 1) being the vertex (= FEX).  Pure geometry,
%   no OPD; aperture obscuration does not matter (use .ok_trace).
%
%   IELT is the exit-pupil return surface (a Return/Reference element,
%   typically nElt-1).  A stop must be set (the prescription's ApStop
%   usually suffices).  XPS leaves the source grid in the differential-
%   field state, so do a macos.trace / macos.modify before resuming
%   normal use.
%
%   See also: macos.get_ray_info, macos.fex, macos.sxp.
    arguments
        iElt (1,1) double {mustBeInteger, mustBePositive}
    end
    mmacos('xps_cmd', iElt);
end
