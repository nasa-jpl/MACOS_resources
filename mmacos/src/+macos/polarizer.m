function out = polarizer(srf, opts)
%MACOS.POLARIZER  Set or query an ideal linear polarizer element.
%   macos.polarizer(SRF, 'axis', A) sets the TRANSMISSION axis of the
%   TrPolarizer element at SRF. A is a 3-vector in GLOBAL coordinates; it
%   need not be unit length, and it need not lie in the element's surface.
%   The engine projects it into each ray's transverse plane and normalizes
%   (see PolElt in elemsub.F).
%
%   S = macos.polarizer(SRF) QUERIES and returns a struct:
%     .axis      [3x1] the stored axis (unit length as stored)
%     .elt_type  the element's EltID (15 = TrPolarizer)
%
%   The element is ideal: transmission 1 along the axis, 0 across it, no
%   Fresnel loss at the faces, no substrate. The REJECTED port is not
%   produced -- a polarizing beamsplitter needs two traces, or (better at
%   45 degrees) a coated Reflector, where the s/p basis is physically
%   meaningful and already gated by tJonesPupil.
%
%   Requires macos.polarization('on'); with polarization off the element is
%   a plain geometric surface and the axis is inert. Setting the axis
%   invalidates the cached trace (MODIFY), so re-trace.
%
%   NORMAL INCIDENCE is the gated regime. Off normal, projecting the
%   declared PASS axis is not the same as projecting a declared BLOCK axis
%   and transmitting the complement -- the two differ at O(sin^2 AOI)
%   because projection does not preserve orthogonality. This function
%   declares the PASS axis. See REVIEW_POL_ELEMENTS_2026-07-27.md; the
%   choice is reported, not settled.
%
%   Example -- crossed polarizers:
%     macos.polarization('on', 'ex', 1, 'ey', 0);
%     macos.polarizer(2, 'axis', [1 0 0]);
%     macos.polarizer(4, 'axis', [0 1 0]);
%     macos.trace(4);  f = macos.ray_field(4);   % ~0
%
%   See also: macos.waveplate, macos.elt_jones, macos.polarization.
arguments
    srf       (1,1) double {mustBeInteger, mustBePositive}
    opts.axis double = []
end

if isempty(opts.axis)
    [axis_, ~, eltType] = mmacos('polelt_get', srf);
    out.axis     = axis_(:);
    out.elt_type = round(eltType);
    return
end

a = opts.axis(:).';
if numel(a) ~= 3
    error('macos:polarizer:badAxis', '''axis'' must be a 3-vector.');
end
if ~(norm(a) > 0)
    error('macos:polarizer:zeroAxis', '''axis'' must be non-zero.');
end
% retardance is meaningless for a polarizer; the engine ignores it
mmacos('polelt_set', srf, a, 0);
end
