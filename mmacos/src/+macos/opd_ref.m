function out = opd_ref(mode)
%MACOS.OPD_REF  Choose (or read) the OPD map's reference.
%   macos.opd_ref('chief')  reference every ray to the CHIEF ray's own OPL.
%   macos.opd_ref('mean')   reference to the whole-aperture mean OPL.
%   r = macos.opd_ref()     read the current setting ('chief' | 'mean').
%
%   The engine (tracesub.F, SUBROUTINE OPD) fills OPDMat one of two ways:
%
%     'chief'  OPD(ray) = CumRayL(ray) - CumRayL(chief).  A fixed per-trace
%              scalar reference; rays do not couple.  Used only while the
%              chief ray survives to the evaluation element (LRayOK(1));
%              the engine falls back to 'mean' for that trace if it does
%              not, silently.
%     'mean'   OPD(ray) = CumRayL(ray) - CumRayL(chief) - mean over EVERY
%              valid ray.  The engine default.
%
%   WHY YOU MAY WANT 'chief' ON A SEGMENTED PUPIL.  The 'mean' reference is
%   one global scalar shared by all segments, so perturbing ONE segment
%   moves it by (N_seg/N_total) x (that segment's mean response) and the
%   shift is then subtracted from every ray.  Unperturbed segments report a
%   spurious uniform piston; the perturbed one is biased by the same
%   constant.  Measured on e5hex1 (7 hex segments, Tz = 1e-8 m on one
%   segment, OPD at the exit pupil): unperturbed segments piston by 16.7%
%   of the peak response under 'mean' and by exactly 0 under 'chief'.
%
%   SCOPE.  This is session state, and LOADING A PRESCRIPTION RESETS IT to
%   'mean' -- call it AFTER load_rx, not before.  The Rx keyword
%   `UseChfRay4OPD= Y` is the per-prescription equivalent.  Changing the
%   setting dirties the cached trace, so the next macos.opd() re-traces.
%
%   Note that 'chief' and 'mean' maps differ by a CONSTANT (that trace's
%   mean OPD), so RMS WFE, P-V and every mean-removed statistic are
%   unchanged; what changes is absolute piston -- which is exactly what a
%   per-segment sensitivity column is made of.
%
%   See also: macos.opd, macos.trace, mmacos/doc/opd_conventions.md.
arguments
    mode (1,:) char {mustBeMember(mode, {'chief','mean',''})} = ''
end
if isempty(mode)
    if logical(mmacos('opd_ref_get')), out = 'chief'; else, out = 'mean'; end
    return
end
mmacos('opd_ref_set', double(strcmp(mode, 'chief')));
end
