function varargout = ffcut(on)
%MACOS.FFCUT  The far-field evanescent cut (engine 2026-10-03, dyson5 addendum 15).
%   macos.ffcut(true) makes every far-field leg zero the output pixels with
%   x^2 + y^2 > dz^2 -- |sin theta| > 1, spatial frequencies above 1/lambda,
%   which carry no propagating energy but which a wide output window (a
%   pupil sampled finer than lambda/2) otherwise hands to an energy-fraction
%   metric.  macos.ffcut(false) restores the default (off).
%   [on, npix] = macos.ffcut() returns the state and how many pixels the
%   last far-field leg zeroed.  Session state (not reset by a load);
%   setting it dirties the cached propagation.
if nargin == 0
    [on, npix] = mmacos('ffcut_get');
    varargout = {logical(on), double(npix)};
else
    mmacos('ffcut_set', double(logical(on)));
end
end
