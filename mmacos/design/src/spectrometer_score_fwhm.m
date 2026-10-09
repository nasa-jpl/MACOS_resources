function w = spectrometer_score_fwhm(dv, slit_px, a_px)
%SPECTROMETER_SCORE_FWHM  FWHM (px) of rect(slit_px) (x) LSF(rays) (x) rect(1 px) (x) Airy-LSF(a_px).
%   Shared by spectrometer_score (engine rays) and spectrometer_score_chain
%   (chain rays).  dv: ray offsets from the centroid in pixels; slit_px: slit
%   image width (0 = none, the cross-track case); a_px: lambda F in pixels
%   (the Airy line-spread function = the 2-D Airy integrated along the other
%   axis; incoherent approximation, legitimate while 2.44 lambda F < pixel).
    h = 0.01;  g = -8:h:8;
    lsf = hist_(dv, g);
    if slit_px > 0, lsf = conv_(lsf, rect_(g, slit_px)); end
    lsf = conv_(lsf, rect_(g, 1));
    lsf = conv_(lsf, airy_lsf_(g, a_px));
    lsf = lsf/max(lsf);
    i = find(lsf >= 0.5);
    if isempty(i), w = NaN; return; end
    i1 = i(1);  i2 = i(end);
    w = interp_(g, lsf, i2, i2+1) - interp_(g, lsf, i1-1, i1);
end

function x = interp_(g, f, ia, ib)
    if ia < 1 || ib > numel(g), x = g(max(1, min(ia, numel(g)))); return; end
    x = g(ia) + (0.5 - f(ia))*(g(ib)-g(ia))/(f(ib)-f(ia));
end

function y = hist_(d, g)
    h = g(2)-g(1);  y = zeros(size(g));
    k = round((d - g(1))/h) + 1;  k = k(k >= 1 & k <= numel(g));
    for q = k(:)', y(q) = y(q) + 1; end
    y = y/sum(y);
end

function y = rect_(g, w)
    y = double(abs(g) <= w/2);  y = y/sum(y);
end

function y = airy_lsf_(g, a)
    if a <= 0, y = double(g == 0); return; end
    [X, Y] = meshgrid(g, -4*a:(g(2)-g(1)):4*a);
    r = sqrt(X.^2 + Y.^2);  z = pi*r/a;  z(z == 0) = 1e-12;
    psf = (2*besselj(1, z)./z).^2;
    y = sum(psf, 1);  y = y/sum(y);
end

function y = conv_(a, b)
    n = numel(a);  y = conv(a, b, 'full');
    k0 = floor(numel(b)/2);  y = y(k0 + (1:n));
end
