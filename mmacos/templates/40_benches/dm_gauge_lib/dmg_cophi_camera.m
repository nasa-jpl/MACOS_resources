function [Fn, scale] = dmg_cophi_camera(Fr, cam, seed)
%DMG_COPHI_CAMERA  Put heterodyne frames onto a camera: well, shot and read noise.
%   Fn = dmg_cophi_camera(Fr, cam) scales the frames so the brightest pixel's
%   MEAN over the beat sits at cam.fill * cam.well electrons (the exposure a
%   heterodyne reading can afford: the fringe swings about that mean by the
%   contrast, so fill <= 1/(1+g) keeps the peak under the well), then adds
%   Poisson shot noise and Gaussian read noise cam.read e- rms per pixel per
%   frame.  Dave's rule (BRIEF_cophi_campaign sec. 7 item 3): read noise is in
%   EVERY camera comparison; the well sets the photons per frame.
%     cam.well   full well, e-            (survey: 60e3 Feng's baseline; 200e3 Photonfocus)
%     cam.read   read noise, e- rms/frame (survey)
%     cam.fill   fraction of the well at the frame mean (default 0.5)
%   scale returns the e-/intensity-unit factor applied, so photons per frame
%   are known for the pm-per-photon rows.  seed (optional) sets rng.
%   Pure MATLAB; gate tCophi (1/sqrt(N) scaling, the shot-noise phase formula).
if nargin >= 3 && ~isempty(seed), rng(seed); end
fill = 0.5;  if isfield(cam, 'fill') && ~isempty(cam.fill), fill = cam.fill; end
m = mean(Fr, 3);
scale = fill * cam.well / max(m(:));
Fe = Fr * scale;
Fn = poissrnd_(Fe) + cam.read * randn(size(Fe));
end

function x = poissrnd_(lam)
% Poisson draws without the Statistics toolbox: Gaussian above 50 e-
% (relative error < 1e-3 at that mean), exact small-count sampling below.
x = lam + sqrt(max(lam, 0)) .* randn(size(lam));
small = lam < 50;
if any(small(:))
    L = lam(small); n = numel(L); k = zeros(n,1); p = exp(-L); s = p; u = rand(n,1);
    while any(u > s)
        more = u > s;  k(more) = k(more) + 1;
        p(more) = p(more) .* L(more) ./ k(more);  s(more) = s(more) + p(more);
    end
    x(small) = k;
end
end
