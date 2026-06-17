function wvl = get_src_wvl()
%MACOS.GET_SRC_WVL  Source wavelength (WaveUnits).
%   Returns the currently-configured source wavelength in the
%   prescription's WaveUnits (typically µm).  Multiply by macos.cbm()
%   * wave-unit-ratio for SI if needed.
%
%   Implementation: api src_wvl is intent(inout) — caller provides a
%   placeholder value, gets the engine's value back in getter mode
%   (setter=0).
wvl = mmacos('src_wvl', 0.0, 0);
end
