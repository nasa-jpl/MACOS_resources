"""V5: the vector pair solved for AMPLITUDE and phase per pixel (plan 11.2) + the amplitude-dip gate.
Usage: python3 patch_amp.py <dir holding dm_gauge_lib/ and zwfs_dm96/>   (apply AFTER patch_analyzer.py)"""
import sys, re
root = sys.argv[1]
def patch(path, pairs):
    s = open(path).read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:70], s.count(old))
        s = s.replace(old, new)
    open(path, 'w').write(s)
    print('patched', path)

g = root + '/dm_gauge_lib/dmg_zwfs_gauge.m'
patch(g, [
("""function h = measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm, ana)
[Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana);
h = reconV_(Ip, Im, C);
end
""",
"""function h = measV_(M, iTO, iMASK, iDET, V, Vm, N_WF, C, leak, arm, ana)
[Ip, Im] = frameV_(M, iTO, iMASK, iDET, V, Vm, N_WF, leak, arm, ana);
if isfield(C, 'V_CLEAR') && C.V_CLEAR                      % V5: the state's clear frame -> the amplitude
    macos.set_elt_grid(iTO, macos.get_elt_grid_spacing(iTO), M);
    I0 = abs(macos.complex_field(iDET)).^2;
    h = reconV_(Ip, Im, C, I0);
else
    h = reconV_(Ip, Im, C);
end
end
"""),
("""ana = struct('mode', 'none', 'lA', 0, 'cA', 0, 'lB', 0, 'cB', 0);
""",
"""ana = struct('mode', 'none', 'lA', 0, 'cA', 0, 'lB', 0, 'cB', 0);
% V5 (2026-09-14, plan 11.2): the COMPLEX AMPLITUDE from the vector
% sensor.  The pair alone cannot: the two images are two circles in the
% complex plane (|E + r+|^2 = I+/|kappa+|^2, |E + r-|^2 = I-/|kappa-|^2,
% r+- = sqrt(eta) c+- b+- / kappa+-) whose two intersections are mirror
% images across the line of centers (Re E = |b| for the pi/2 dimple): the
% pair measures A sin(phi) and |A cos(phi) - b|, so amplitude and phase
% are ambiguous where A cos(phi) crosses b (100 nm pokes do), and the
% amplitude is a square-root observable near it (ZW.solveVA keeps the
% intersection for the record of that).  With the STATE'S CLEAR FRAME
% (the unmasked intensity, a third exposure) the amplitude is measured
% directly and the pair gives the phase exactly: V_CLEAR true makes the
% V reading take that frame per state (measV_: I0 = |E_state|^2 into the
% solver's I0).  Default false: the phase-only reading of record (the
% flat's amplitude).
vclear = false;  if isfield(opt, 'V_CLEAR'), vclear = logical(opt.V_CLEAR); end
"""),
("""           'EbP0', Ebf, 'EbM0', Ebf);                                   % per-channel flat reference waves
""",
"""           'EbP0', Ebf, 'EbM0', Ebf, ...                                % per-channel flat reference waves
           'V_CLEAR', vclear);                                          % V5: the clear frame per state (complex amplitude)
"""),
("""ZW.reconV   = @(Ip, Im, varargin) reconV_(Ip, Im, C, varargin{:});   % (Ip, Im, I0, b0, niter)
""",
"""ZW.reconV   = @(Ip, Im, varargin) reconV_(Ip, Im, C, varargin{:});   % (Ip, Im, I0, b0, niter)
ZW.solveVA  = @(Ip, Im, varargin) solveVA_(Ip, Im, C, varargin{:});  % V5: the pair-only complex solve, [phi, info] with info.A, info.E (Ip, Im, Eprior, niter) -- ambiguous at gauge-level phases, kept for the record
ZW.frameV_sur = @(E) deal(abs(E + cc*bsur(E)).^2, abs(E + ccm*bsur(E)).^2);   % the pair from a detector-plane field (ideal channels; == the engine's chained frames, gate G8)
"""),
("""function [h, info] = reconV_(Ip, Im, C, varargin)
[phi, info] = solveV_(Ip, Im, C, varargin{:});
h = C.S_CONV*phi*C.LAM/(4*pi);
end
""",
"""function [h, info] = reconV_(Ip, Im, C, varargin)
[phi, info] = solveV_(Ip, Im, C, varargin{:});
h = C.S_CONV*phi*C.LAM/(4*pi);
end

function [phi, info] = solveVA_(Ip, Im, C, Eprior, niter)
%SOLVEVA_  Per-pixel exact solve of the COMPLEX field from the image pair.
%   Same per-channel model as solveV_ (kappa+-, eta, c+-, b+-, qL, qR):
%       I+ = |kappa+|^2 |E + r+|^2,  r+ = sqrt(eta) c+ b+ / kappa+
%       I- = |kappa-|^2 |E + r-|^2,  r- = sqrt(eta) c- b- / kappa-
%   Two circles in the complex plane, centers -r+-, radii sqrt(I+-)/|kappa+-|;
%   E is at their intersection (two points, mirror images across the line
%   of centers; the one nearest the prior is taken; when noise leaves the
%   circles apart, the nearest points' midpoint).  Amplitude AND phase
%   per pixel; the reference waves b+- iterated from the solved field.
%   Eprior: the complex field the root choice starts from (default the
%   flat's, C.E0).  info.A = |E| (the amplitude map), info.E, info.phi
%   unwrapped against the flat, info.dphi, info.sep (rms of the circles'
%   gap, |d - R1 - R2| where they do not meet, 0 for an exact model).
if nargin < 4 || isempty(Eprior), Eprior = C.E0; end
if nargin < 5 || isempty(niter), niter = C.NITER; end
N = C.N_WF;  m = C.msk;
wrap = @(p) atan2(sin(p), cos(p));
th0 = angle(C.E0);
if isempty(C.qL), qL = 1; else, qL = C.qL; end
if isempty(C.qR), qR = 1; else, qR = C.qR; end
bP = C.EbP0;  bM = C.EbM0;
kP = C.kapP;  kM = C.kapM;  se = sqrt(C.eta);
if isscalar(kP), kP = kP*ones(N); end;  if isscalar(kM), kM = kM*ones(N); end
E = Eprior;  phi = zeros(N);
info = struct('dphi', zeros(1, niter+1), 'sep', zeros(1, niter+1));
for it = 0:niter
    c1 = -se*C.cc*bP./kP;   c2 = -se*C.ccm*bM./kM;          % the circles' centers
    R1 = sqrt(max(Ip, 0))./abs(kP);  R2 = sqrt(max(Im, 0))./abs(kM);
    dv = c2 - c1;  d = max(abs(dv), realmin);  u = dv./d;
    a = (R1.^2 - R2.^2 + d.^2) ./ (2*d);
    h2 = R1.^2 - a.^2;  gap = zeros(N);  gap(h2 < 0) = abs(d(h2 < 0) - R1(h2 < 0) - R2(h2 < 0));
    h = sqrt(max(h2, 0));
    p = c1 + a.*u;
    z1 = p + 1i*h.*u;  z2 = p - 1i*h.*u;
    pick = abs(z1 - E) <= abs(z2 - E);
    Enew = z2;  Enew(pick) = z1(pick);  Enew(~m) = 0;
    ph = wrap(angle(Enew) - th0);  ph(~m) = 0;
    info.dphi(it+1) = sqrt(mean((ph(m) - phi(m)).^2));  info.sep(it+1) = sqrt(mean(gap(m).^2));
    phi = ph;  E = Enew;
    if it < niter
        bP = C.bsur(qL .* E);  bM = C.bsur(qR .* E);       % the reference waves from the SOLVED field
    end
end
info.A = abs(E);  info.E = E;  info.phi = phi;  info.b = bP;  info.bM = bM;
end
"""),
])

p = root + '/zwfs_dm96/zwfs_params.m'
s = open(p).read()
m = re.search(r"^(P\.mask\.v_qwp_az\s*=.*\n)", s, re.M); assert m
s = s.replace(m.group(1), m.group(1) + """P.mask.v_clear = false;      % V5 (plan 11.2): the V reading takes the state's CLEAR frame too (3 frames) and reads amplitude and phase; false = the phase-only pair of record
P.mask.v_dip = [];           % V5 gate G9: pupil amplitude dips (fractions, e.g. [0.05 0.20]) the vector pair must read through; [] = skip
""", 1)
open(p, 'w').write(s); print('patched', p)

r = root + '/zwfs_dm96/zwfs_run.m'
patch(r, [
("""if isstruct(P.mask.v_analyzer), g.V_ANALYZER = P.mask.v_analyzer; end   % V4 (resolved from 'engine' in the bench stage)
""",
"""if isstruct(P.mask.v_analyzer), g.V_ANALYZER = P.mask.v_analyzer; end   % V4 (resolved from 'engine' in the bench stage)
g.V_CLEAR = P.mask.v_clear;                                              % V5
"""),
("""    priced = (ZW.leak.eta < 1 && strcmp(P.mask.v_cal, 'ideal')) || ...
             (~strcmp(ZW.arm.mode, 'none') && ~strcmp(P.mask.v_cal, 'map')) || ...
             ~strcmp(ZW.ana.mode, 'none');
""",
"""    priced = (ZW.leak.eta < 1 && strcmp(P.mask.v_cal, 'ideal')) || ...
             (~strcmp(ZW.arm.mode, 'none') && ~strcmp(P.mask.v_cal, 'map')) || ...
             ~strcmp(ZW.ana.mode, 'none');
    % ---- G9 (V5, plan 11.2): the pair read through a pupil AMPLITUDE dip.
    % The dip multiplies the unmasked poke field at the pupil image (a
    % Gaussian well of depth d over a quarter of the pupil, off center); the
    % frames are the surrogate's (== the engine's chained frames, G8).  The
    % phase-only solve with the FLAT's amplitude misreads by the dip; with
    % the STATE's clear frame (I0) it must not; the pair-only complex solve
    % (solveVA_) is printed for the record: ambiguous where A cos(phi)
    % crosses the reference wave, which 100 nm pokes do.
    if ~isempty(P.mask.v_dip)
        [ii, jj] = ndgrid(1:N_WF, 1:N_WF);  rp = sqrt(nnz(msk)/pi);
        cy = ZW.ctr(1) + 0.45*rp;  cx = ZW.ctr(2) + 0.30*rp;  sg = 0.25*rp;
        well = exp(-((ii-cy).^2 + (jj-cx).^2)/(2*sg^2));
        A0m = abs(ZW.E0);
        for d = P.mask.v_dip(:)'
            Adip = 1 - d*well;
            Ed = Adip .* Et;                                      % the G4 poke field with the dip
            [Ipd, Imd] = ZW.frameV_sur(Ed);
            hF = ZW.reconV(Ipd, Imd);                             % the flat's amplitude
            hC = ZW.reconV(Ipd, Imd, abs(Ed).^2);                 % the state's clear frame
            [phA, ia] = ZW.solveVA(Ipd, Imd, [], 0);  hA = P.mask.S_CONV*phA*P.LAM/(4*pi);   % the pair alone, one pass
            eF = sqrt(mean((pm_(hF) - pm_(h_t)).^2))*1e9;  eC = sqrt(mean((pm_(hC) - pm_(h_t)).^2))*1e9;
            eA = sqrt(mean((pm_(hA) - pm_(h_t)).^2))*1e9;  eAmp = sqrt(mean((ia.A(msk)./A0m(msk) - Adip(msk)).^2));
            ok = eC < 1e-3*gV.rmsfig && eF > 10*eC;
            dmg_say(rep, 'G9 vector pair through a %.0f%% pupil amplitude dip (Gaussian, sigma 0.25 of the pupil radius, off center): the flat''s amplitude %.2f pm; the state''s clear frame %.3f pm (gate < 0.1%% of the figure; non-vacuity: the flat''s > 10x); the pair alone %.0f pm, its amplitude %.1e rms off (ambiguous: A cos phi crosses b at these pokes)   -> %s\\n', ...
                100*d, eF, eC, eA, eAmp, ifelse_(ok, 'PASS', 'FAIL'));
            assert(ok, 'G9 FAIL at a %.0f%% dip', 100*d);
        end
    end
"""),
])
