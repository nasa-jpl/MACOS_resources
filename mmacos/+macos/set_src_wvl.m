function set_src_wvl(wvl)
%MACOS.SET_SRC_WVL  Set source wavelength (in the Rx's WaveUnits).
arguments
    wvl (1,1) double {mustBePositive}
end
mmacos('src_wvl', wvl, 1);
end
