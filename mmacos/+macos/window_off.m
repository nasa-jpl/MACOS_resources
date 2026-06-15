function window_off()
%MACOS.WINDOW_OFF  Turn the WINDOW pixel-location option back off.
%   macos.window_off() cancels macos.window(...) so PIX / COMPOSE go back
%   to re-centring each source's image on its own grid (the default).
%
%   See also: macos.window, macos.compose.
mmacos('window_off');
end
