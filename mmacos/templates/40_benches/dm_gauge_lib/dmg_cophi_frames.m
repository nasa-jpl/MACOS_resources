function [Fr, alpha, info] = dmg_cophi_frames(Et, Elo, opt)
%DMG_COPHI_FRAMES  Heterodyne (COPHI) camera frames from two complex fields.
%   [Fr, alpha] = dmg_cophi_frames(Et, Elo, opt) forms the N integrating-bucket
%   frames a camera records while the LO arm's phase advances through one beat:
%
%       I(t) = |Et + Elo exp(i a(t))|^2 = I0 (1 + g cos(phi - a(t)))
%       phi  = arg(Et conj(Elo)),  g = 2|Et||Elo| / (|Et|^2 + |Elo|^2)
%       I_k  = (1/D) int_{a_k}^{a_k+D} I da
%            = I0 (1 + g sinc(D/2) cos(phi - a_k - D/2)),   D = 2 pi / N
%
%   i.e. the four-step interferometer's reading with FREQUENCY-SET steps (the
%   AOM pair's beat, Feng Zhao's COPHI), each frame integrating a quarter
%   cycle: the fringe contrast is reduced by sinc(pi/4) = 0.9003 for N = 4
%   (the "integrating bucket" penalty), nothing else changes.  The returned
%   `alpha` are the BUCKET CENTRES a_k + D/2 -- the phases a reconstruction
%   must assume (dmg_cophi_recon).  Et and Elo are complex fields on the same
%   detector grid (ny x nx, or ny x nx x 3 for the three Cartesian components
%   of a vector-mode trace: the intensities add over components).
%
%   opt (all optional):
%     .N        buckets per beat (default 4)
%     .jitter   rms TRIGGER jitter per frame, radians of beat phase (default 0):
%               a_k -> a_k + d_k, d_k ~ N(0, jitter)
%     .walk     DIFFERENTIAL fiber-phase walk within the scan, radians per
%               frame rms (default 0): phi -> phi + w_k, w a random walk
%               (w_1 = 0); this is the intra-scan drift the four-step has no
%               cure for
%     .drift    deterministic phase ramp per frame, radians (default 0)
%     .seed     rng seed for jitter/walk (default: leave rng alone)
%     .nsub     sub-steps per bucket for the integration (default 32; the
%               closed form is used when jitter = walk = drift = 0)
%   info returns the per-frame applied jitter (.dk), walk (.wk) and the
%   contrast factor sinc(D/2) (.cbucket).
%
%   The LO's WAVEFRONT (tilt / defocus per cycle) is bench-side: it is in Elo
%   as the engine traces the LO arm; this routine is the detector physics
%   only.  Pure MATLAB; gate tCophi.  (COPHI campaign, 2026-10-11.)
if nargin < 3, opt = struct(); end
N      = getf(opt, 'N', 4);
jit    = getf(opt, 'jitter', 0);
walk   = getf(opt, 'walk', 0);
drift  = getf(opt, 'drift', 0);
nsub   = getf(opt, 'nsub', 32);
if isfield(opt, 'seed') && ~isempty(opt.seed), rng(opt.seed); end
D   = 2*pi/N;
a0  = (0:N-1)*D;                        % bucket starts
dk  = jit  * randn(1, N);               % trigger jitter per frame
wk  = cumsum([0, walk*randn(1, N-1)]) + drift*(0:N-1);   % fiber walk + ramp
alpha = a0 + D/2;                       % the nominal bucket centres
% the two arms' intensities and the interference term, summed over components
I0  = sum(abs(Et).^2 + abs(Elo).^2, 3);
X   = sum(Et .* conj(Elo), 3);          % |Et||Elo| exp(i phi)
Fr  = zeros([size(I0), N]);
if jit == 0 && walk == 0 && drift == 0
    cb = sinc_(D/2);
    for k = 1:N
        Fr(:,:,k) = I0 + 2*cb*real(X .* exp(-1i*alpha(k)));
    end
else
    cb = sinc_(D/2);
    s  = (0.5:nsub-0.5)/nsub;           % mid-point rule over the bucket
    for k = 1:N
        acc = zeros(size(I0));
        for j = 1:nsub
            a = a0(k) + dk(k) + s(j)*D;
            acc = acc + 2*real(X .* exp(1i*wk(k)) .* exp(-1i*a));
        end
        Fr(:,:,k) = I0 + acc/nsub;
    end
end
info = struct('dk', dk, 'wk', wk, 'cbucket', cb, 'D', D);
end

function y = sinc_(x)
if x == 0, y = 1; else, y = sin(x)/x; end
end

function v = getf(s, f, d)
if isfield(s, f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end
