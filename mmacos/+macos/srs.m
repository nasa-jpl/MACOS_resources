function srs(iSlv1, iSlv2, opts)
%MACOS.SRS  Slave an element to another via the SRS (Set Reference Surface) cmd.
%   macos.srs(IS1, IS2) moves IS1 onto IS2 in the sense of macos's
%   interactive SRS command -- IS1's pose (vpt / psi / rpt) is
%   recomputed from the chief-ray geometry as it crosses IS2's plane.
%
%   macos.srs(IS1, IS2, 'link', true) creates a permanent linked-element
%   relationship so subsequent perturbations of IS2 drag IS1 along.
%   Default false (one-shot slave).
%
%   Used by FocalPlaneChannel(mode='srs') to slave the EP element to
%   the moved FP so the OPD reference surface follows the chief ray.
%
%   See also: macos.sxp, macos.stop.
arguments
    iSlv1     (1,1) double {mustBeInteger, mustBePositive}
    iSlv2     (1,1) double {mustBeInteger, mustBePositive}
    opts.link (1,1) logical = false
end
mmacos('srs_run', iSlv1, iSlv2, opts.link);
end
