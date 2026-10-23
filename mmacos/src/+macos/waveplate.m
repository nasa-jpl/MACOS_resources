function out = waveplate(srf, opts)
%MACOS.WAVEPLATE  Set or query a linear retarder (waveplate) element.
%   macos.waveplate(SRF, 'axis', A, 'retardance', R) configures the
%   WavePlate element at SRF. A is the FAST axis as a 3-vector in GLOBAL
%   coordinates (projected into each ray's transverse plane by the engine).
%   R is the retardance in WAVES at the CURRENT wavelength: 0.25 for a
%   quarter-wave plate, 0.5 for a half-wave plate.
%
%   S = macos.waveplate(SRF) QUERIES and returns a struct:
%     .axis        [3x1] the stored fast axis
%     .retardance  scalar, waves at the current wavelength
%     .elt_type    the element's EltID (18 = WavePlate)
%
%   CHROMATIC BY CONSTRUCTION. The engine stores the PHYSICAL retardance
%   (n_slow-n_fast)*d = R*lambda, so a plate set to 0.25 waves at 1 um is
%   0.125 waves at 2 um -- a fixed piece of glass, not a fixed phase. This
%   is the same treatment macos.coating's layer stack gets. A query after a
%   wavelength change therefore returns a DIFFERENT R than was set; that is
%   the physics, not a round-trip failure.
%
%   SIGN CONVENTION (derived from the engine, not chosen). MACOS propagates
%   a field as exp(-i*2*pi*L*N/lambda), i.e. exp(+i*omega*t) time
%   dependence, so the slow axis accumulates the more negative phase and the
%   element Jones in its (fast, slow) eigenbasis is diag(1, exp(-i*delta))
%   with delta = 2*pi*R.  "Fast axis leads", as pinned in the conventions
%   table in macos_f90/CLAUDE.md.
%
%   The element is a thin, non-ray-splitting idealization: no o/e walk-off,
%   no Fresnel loss at the faces, no substrate thickness. R is also
%   independent of incidence angle, where a real crystal plate's retardance
%   is not -- the field-of-view effect that drives compound and Pancharatnam
%   designs; bounding that needs a birefringent-plate model with o/e indices
%   and thickness. It is also the primitive for bounding stress
%   birefringence in a transmissive optic.
%
%   The declared axis IS the material (crystal fast) axis, so off normal
%   incidence it is the vector the engine projects into the ray's
%   transverse plane -- the settled material-axis rule, which for a
%   waveplate is simply the declared axis. (For macos.polarizer the
%   material axis is the absorbing direction instead, so the two elements
%   project different vectors from the same keyword; see that function.)
%
%   Requires macos.polarization('on'). Setting invalidates the cached trace.
%
%   Example -- linear to circular:
%     macos.polarization('on', 'ex', 1, 'ey', 0);
%     macos.waveplate(3, 'axis', [1 1 0], 'retardance', 0.25);
%     macos.trace(3);  f = macos.ray_field(3);
%
%   See also: macos.polarizer, macos.elt_jones, macos.polarization.
arguments
    srf             (1,1) double {mustBeInteger, mustBePositive}
    opts.axis       double = []
    opts.retardance double = []
end

if isempty(opts.axis) && isempty(opts.retardance)
    [axis_, ret, eltType] = mmacos('polelt_get', srf);
    out.axis       = axis_(:);
    out.retardance = ret;
    out.elt_type   = round(eltType);
    return
end

% partial updates read the current state first, so setting one field does
% not silently zero the other
cur = [];
if isempty(opts.axis) || isempty(opts.retardance)
    [a0, r0, ~] = mmacos('polelt_get', srf);
    cur = struct('axis', a0(:).', 'retardance', r0);
end

if isempty(opts.axis), a = cur.axis; else, a = opts.axis(:).'; end
if isempty(opts.retardance), r = cur.retardance; else, r = opts.retardance; end

if numel(a) ~= 3
    error('macos:waveplate:badAxis', '''axis'' must be a 3-vector.');
end
if ~(norm(a) > 0)
    error('macos:waveplate:zeroAxis', '''axis'' must be non-zero.');
end
if ~isscalar(r)
    error('macos:waveplate:badRet', '''retardance'' must be a scalar.');
end

mmacos('polelt_set', srf, a, r);
end
