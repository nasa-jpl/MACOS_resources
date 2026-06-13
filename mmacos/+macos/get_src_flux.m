function flux = get_src_flux()
%MACOS.GET_SRC_FLUX  Source flux (propagated intensity scales with it).
%   Implementation: api src_flux is intent(inout) -- caller provides a
%   placeholder value, gets the engine's value back in getter mode
%   (setter=0).
flux = mmacos('src_flux', 0.0, 0);
end
