function cf = complex_field(srf, opts)
%MACOS.COMPLEX_FIELD  WFElt at element SRF (N x N complex double).
%   cf = macos.complex_field(SRF) propagates to SRF and returns the
%   complex wavefront on the diffraction grid.  |cf|^2 matches
%   macos.intensity(SRF) to numerical precision.
%
%   cf = macos.complex_field(SRF, 'plane', K) returns a single Cartesian
%   FIELD COMPONENT.  In vector-diffraction mode the three wavefront
%   storage planes are repurposed as Ex/Ey/Ez of one wavefront, so
%   K = 1, 2, 3 selects Ex, Ey, Ez.  This is the only way to see how the
%   three components contribute to a propagated intensity:
%   macos.intensity sums them, and the default K = 0 returns the
%   element's own wavefront (identical to omitting the option).
%
%   Requesting K = 1..3 with vector diffraction OFF is an ERROR, not a
%   silent fallback: in scalar mode plane K is an unrelated wavefront,
%   not a field component, and returning it would invite exactly the
%   misreading the option exists to prevent.  Turn it on with
%   macos.polarization('on') + macos.vector_diffraction(true).
%
%   The three planes sum in intensity, not in amplitude:
%       Ix = abs(macos.complex_field(s, 'plane', 1)).^2;   % etc.
%       Ix + Iy + Iz  ==  macos.intensity(s)
%   which is the identity the round-trip test pins.
%
%   Name-value pairs:
%     'reset_trace' true (default) — see macos.intensity note.
%     'plane'       0 (default) = the element's own wavefront;
%                   1|2|3 = Ex|Ey|Ez (vector diffraction only).
%
%   See also: macos.intensity, macos.vector_diffraction, macos.ray_field.
arguments
    srf              (1,1) double {mustBeInteger, mustBePositive}
    opts.reset_trace (1,1) logical = true
    opts.plane       (1,1) double {mustBeInteger, mustBeInRange(opts.plane, 0, 3)} = 0
end
cf = mmacos('complex_field', srf, double(opts.reset_trace), opts.plane);
end
