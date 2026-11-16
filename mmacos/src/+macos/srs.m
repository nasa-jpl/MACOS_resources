function srs(iSlv1, iSlv2, opts)
%MACOS.SRS  Slave an element to another via the SRS (Set Reference Surface) cmd.
%   macos.srs(IS1, IS2) moves IS1 onto IS2 in the sense of macos's
%   interactive SRS command -- IS1's pose (vpt / psi / rpt) is
%   recomputed from the chief-ray geometry as it crosses IS2's plane.
%
%   macos.srs(IS1, IS2, 'link', false) computes and prints the new pose but
%   does not apply it: 'link' is SRS's answer to "Accept the new element?",
%   default true (the CLI's [YES], pymacos srs(link=True)).  It is not a
%   persistent link: SRS re-poses IS1 once; call it again after IS2 moves.
%   (The default was false; since the engine honours a NO (1.2 (gg),
%   2026-10-19) that made macos.srs(IS1, IS2) a no-op -- veneer-default
%   audit, 2026-10-23.)
%
%   Used by FocalPlaneChannel(mode='srs') to slave the EP element to
%   the moved FP so the OPD reference surface follows the chief ray.
%
%   See also: macos.sxp, macos.stop.
arguments
    iSlv1     (1,1) double {mustBeInteger, mustBePositive}
    iSlv2     (1,1) double {mustBeInteger, mustBePositive}
    opts.link (1,1) logical = true
end
mmacos('srs_run', iSlv1, iSlv2, opts.link);
end
