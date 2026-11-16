function vector_diffraction(on)
%MACOS.VECTOR_DIFFRACTION  Toggle vector (3-component) diffraction.
%   macos.vector_diffraction(true)  -> VECTOR: propagate Ex/Ey/Ez as
%       three independent fields.  Since PLAN_POLARIZATION Phase 3a
%       Tranche 1 this covers the WHOLE chain -- every near-field,
%       plane-to-plane, spherical, Fresnel and DFT leg, plus FFObscure
%       and the ray-side aperture masking -- not just the far-field FFT
%       leg.  Intensity/complex-field readouts sum the three components.
%   macos.vector_diffraction(false) -> SCALAR: single-field diffraction.
%
%   VECTOR requires polarization to be ON already (macos.polarization('on'))
%   and a model with mWF>=3; otherwise this errors (unlike the CLI, which
%   silently reverts to scalar).
%
%   CONSTRAINT: vector mode repurposes the model's mWF=3 wavefront planes
%   as Ex/Ey/Ez of ONE wavefront, so only a single wavefront can be in
%   flight -- do not combine it with multi-WF / COMPOSE work.
%
%   Chains with COATED or reflective surfaces BETWEEN physical propagation
%   legs still need Tranche 2 (per-ray running Jones); Tranche 1 is exact
%   when the elements between legs are non-polarizing (Obscuring /
%   Reference / FocalPlane) -- the coronagraph pupil->FPM->Lyot->focal case.
%
%   See also: macos.polarization.
arguments
    on (1,1) logical
end
mmacos('vecdif_set', double(on));
end
