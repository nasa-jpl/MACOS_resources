function vector_diffraction(on)
%MACOS.VECTOR_DIFFRACTION  Toggle vector (3-component) diffraction.
%   macos.vector_diffraction(true)  -> VECTOR: propagate Ex/Ey/Ez as
%       three independent fields (far-field FFT leg only; see the engine
%       polarization notes -- near-field/DFT legs remain scalar).
%   macos.vector_diffraction(false) -> SCALAR: single-field diffraction.
%
%   VECTOR requires polarization to be ON already (macos.polarization('on'))
%   and a model with mWF>=3; otherwise this errors (unlike the CLI, which
%   silently reverts to scalar).
%
%   See also: macos.polarization.
arguments
    on (1,1) logical
end
mmacos('vecdif_set', double(on));
end
