function [Dc, info] = iodc2010_capacity(L, p, opts)
%IODC2010_CAPACITY  The largest entrance pupil a green lens passes ON AXIS at its EFL.
%   [Dc, info] = iodc2010_capacity(L, p) returns the on-axis pupil capacity
%   (mm): the largest paraxial entrance-pupil diameter D at which the lens, as
%   it stands (its radius, signs, spacings and stop place), passes the metric's
%   on-axis test -- no ray lost, RMS <= p.rms_limit -- found by a log-bisection
%   of D between opts.D_lo and opts.D_hi to opts.steps halvings of the log
%   interval (9 steps over 1..150 mm resolve D to ~0.3 mm), the stop
%   semi-diameter re-sized for each probe and the clear apertures open.  The
%   capacity is what the on-axis construction stage (iodc2010_relax) must
%   beat and what the 2026-10-31 record quotes for every seed family (README
%   "The on-axis construction").
%
%   info: .rms (the RMS at Dc, waves), .reason (what fails just above Dc:
%   'rms' | 'vignetting' | 'passes at D_hi'), .n_probes.
arguments
    L struct
    p struct
    opts.D_lo (1,1) double = 1
    opts.D_hi (1,1) double = 150
    opts.steps (1,1) double = 9
    opts.nGridpts (1,1) double = 33
    opts.deck (1,:) char = ''
end
pq = p;  pq.nGridpts = opts.nGridpts;  pq.size_apertures = false;
if isempty(opts.deck), opts.deck = fullfile(p.outdir, [L.name '_capacity.in']); end
lo = opts.D_lo;  hi = opts.D_hi;  info.rms = NaN;  info.reason = '';  info.n_probes = 0;
F = probe_(L, lo, pq, opts.deck);  info.n_probes = 1;
if ~F.pass, Dc = 0;  info.rms = F.rms_waves;  info.reason = F.reason;  return; end
info.rms = F.rms_waves;
F = probe_(L, hi, pq, opts.deck);  info.n_probes = 2;
if F.pass, Dc = hi;  info.rms = F.rms_waves;  info.reason = sprintf('passes at %g', hi);  return; end
info.reason = F.reason;
for it = 1:opts.steps
    mid = sqrt(lo * hi);
    F = probe_(L, mid, pq, opts.deck);  info.n_probes = info.n_probes + 1;
    if F.pass, lo = mid;  info.rms = F.rms_waves; else, hi = mid;  info.reason = F.reason; end
end
Dc = lo;
end

function F = probe_(L, D, pq, deck)
P0 = iodc2010_paraxial(L, pq.lambda_nm);
L.stop(2) = L.stop(2) * D / P0.D;  L.semidia = inf(1, L.N);
P = iodc2010_paraxial(L, pq.lambda_nm);
E = iodc2010_emit(L, P, pq, deck);
macos.load_rx(E.path);
F = iodc2010_eval(E, P, pq, 0);
macos.modify();
end
