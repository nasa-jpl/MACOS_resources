function S = pbs_macneille(opts)
%MACOS.DESIGN.PBS_MACNEILLE  MacNeille polarizing-beamsplitter cube coating.
%
%   S = MACOS.DESIGN.PBS_MACNEILLE(...) returns the layer stack and the
%   substrate index of a MacNeille-type polarizing beamsplitter: a
%   quarter-wave H/L stack cemented between two glass prisms, designed so
%   that the internal angle at every H/L interface is BREWSTER'S ANGLE.
%   The p component then sees zero reflection at every interface and is
%   transmitted essentially completely, while the s component sees a full
%   quarter-wave high reflector.
%
%   SOURCE OF THE DESIGN.  The condition is MacNeille's (US Patent
%   2,403,731, 1946), in the form given by Macleod, "Thin-Film Optical
%   Filters", chapter on polarizers:
%
%       Brewster at the H/L interface :  tan(theta_H) = nL/nH
%       Snell from the prism          :  n_g*sin(45) = nH*sin(theta_H)
%       =>  n_g*sin(aoi) = nH*nL / sqrt(nH^2 + nL^2)                   (*)
%
%   (*) is what fixes the PRISM GLASS: given a coating pair, only one
%   substrate index makes the cube work.  The default H/L pair is the
%   classic visible MacNeille pair, zinc sulphide and cryolite, at the
%   indices Macleod tabulates for the visible (nH = 2.35, nL = 1.35);
%   those give n_g = 1.6555, i.e. a dense flint (~SF2), which is exactly
%   why real MacNeille cubes are made of dense flint and not of BK7.
%
%   Each layer is a QUARTER WAVE AT ANGLE -- n_j*d_j*cos(theta_j) = lam/4 --
%   so the physical thickness carries the internal angle, not just the index.
%
%   BREWSTER AT THE H/L INTERFACES IS NOT ENOUGH, and this is the design
%   lesson the model surfaces.  (*) equalizes the tilted p ADMITTANCES,
%   eta_p = n/cos(theta): with the default pair both come to 2.7101, so for
%   the p component the whole stack is one HOMOGENEOUS slab.  That kills
%   every internal p reflection -- but the slab still has two boundaries
%   with the PRISM (eta_p = 2.3414), and those are not Brewster.  What the
%   slab does there depends only on its TOTAL p phase thickness:
%
%     'design','qw'         H(LH)^N, 2N+1 quarter waves -- an ODD number, so
%                           the p slab is a quarter-wave layer and reflects
%                           p MAXIMALLY.  Measured: R_p = 2.11e-2 at N = 5.
%                           A plausible-looking stack that satisfies the
%                           textbook condition and is still a poor polarizer.
%     'design','symmetric'  (1/2 H  L  1/2 H)^N -- the standard symmetric
%                           period (Macleod ch. 6).  2N quarter waves, an
%                           EVEN number, so the p slab is a half-wave
%                           ABSENTEE and R_p is ZERO IDENTICALLY.  DEFAULT.
%
%   Symmetry matters for a second, independent reason: r of a stack depends
%   on which side you approach from unless the stack is symmetric, and this
%   cube is used from BOTH sides (the test arm transmits then reflects, the
%   reference arm reflects then transmits).  A symmetric stack makes the two
%   arms interchangeable by construction rather than by accident, so the
%   interferogram carries no coating-induced differential piston.
%
%   OPTIONS (all defaults are the design above)
%     'nH'      2.35     high-index layer (ZnS)
%     'nL'      1.35     low-index layer (cryolite, Na3AlF6)
%     'kH','kL' 0        extinction (both are transparent in the visible)
%     'nperiod' 4        N above; either design has 2N+1 layers.  Capped at
%                        4 by the engine's mCoat = 10 -- see the assertion.
%     'design'  'symmetric'  see above ('qw' is the counter-example)
%     'lambda'  6.328e-4 design wavelength, BaseUnits (HeNe in mm)
%     'aoi'     45       angle of incidence IN THE PRISM GLASS, degrees
%     'n_glass' NaN      NaN = solve (*) for the MacNeille index.  Give a
%                        number to DETUNE the cube deliberately (a real
%                        catalogue glass instead of the exact design index):
%                        Brewster is then violated, r_p is no longer zero,
%                        and the PBS starts to rotate the arm states.  That
%                        is the v2 tolerance knob.
%
%   RETURNS struct S
%     .n_glass      prism index actually used
%     .n_glass_mn   the MacNeille index from (*) (equal to .n_glass unless
%                   the cube was detuned)
%     .layers       nlayer-by-3 [n  k  thk_waves], OUTERMOST FIRST, in the
%                   Rx "Coating=" form: thk_waves is the OPTICAL thickness
%                   n*d/lambda, which is what the parser scales by
%                   Wavelen/IndRef at load.
%     .thk          nlayer-by-1 PHYSICAL thickness (BaseUnits) -- what
%                   macos.coating / coat_set take.
%     .theta        nlayer-by-1 internal angle per layer (deg)
%     .theta_H .theta_L .brewster_resid   design diagnostics; the residual
%                   is theta_H + theta_L - 90 (deg), zero on the MacNeille
%                   solution and the size of the detune otherwise.
%     .rt           macos.design.thinfilm_rt at the design point (glass in,
%                   glass out) -- the textbook R/T the engine must reproduce.
%
%   See also: macos.design.thinfilm_rt, macos.design.twyman_green.

arguments
    opts.nH      (1,1) double {mustBePositive} = 2.35
    opts.nL      (1,1) double {mustBePositive} = 1.35
    opts.kH      (1,1) double {mustBeNonnegative} = 0
    opts.kL      (1,1) double {mustBeNonnegative} = 0
    opts.nperiod (1,1) double {mustBeInteger, mustBePositive} = 4
    opts.design  (1,:) char {mustBeMember(opts.design,{'symmetric','qw'})} = 'symmetric'
    opts.lambda  (1,1) double {mustBePositive} = 6.328e-4
    opts.aoi     (1,1) double {mustBePositive} = 45
    opts.n_glass (1,1) double = NaN
end
assert(opts.nH > opts.nL, 'pbs_macneille: nH must exceed nL.');
nlayer = 2*opts.nperiod + 1;
%  mCoat = 10 in elt_mod.F is a HARD engine ceiling on Model-A layers, and
%  the Rx "Coating=" parser does NOT bound-check against it -- it writes
%  IndRefArr(k,iElt)/EltCoatThk(k,iElt) for k = 1..EltCoat with no guard,
%  so an 11-layer stack loads WITHOUT COMPLAINT and corrupts the heap
%  (coat_get then fails, which is the only visible symptom).  Caught
%  building this cube; written up in BRIEF_tg_ifo_v2.md as a separate
%  engine finding.  N = 4 (9 layers) is the largest symmetric design that
%  fits, and at 2380:1 extinction it is if anything MORE representative of
%  a real MacNeille cube than the 9-period ideal would be.
assert(nlayer <= 10, ...
    ['pbs_macneille: %d layers exceeds the engine ceiling mCoat = 10 ' ...
     '(elt_mod.F).  The Rx parser does not check this -- it would load ' ...
     'silently and corrupt memory.  Use nperiod <= 4.'], nlayer);

% ---- (*) the MacNeille substrate index --------------------------------
n_mn = opts.nH*opts.nL / (sind(opts.aoi)*hypot(opts.nH, opts.nL));
n_g  = opts.n_glass;  if isnan(n_g), n_g = n_mn; end

% ---- per-layer internal angle and quarter-wave-AT-ANGLE thickness -----
%  Snell invariant n*sin(theta) is conserved across the whole stack, so the
%  internal angle follows from the PRISM angle, not from the previous layer.
inv_s = n_g*sind(opts.aoi);
nseq  = repmat([opts.nH; opts.nL], opts.nperiod+1, 1);  nseq = nseq(1:nlayer);
kseq  = repmat([opts.kH; opts.kL], opts.nperiod+1, 1);  kseq = kseq(1:nlayer);
assert(all(inv_s < nseq), ...
    ['pbs_macneille: the Snell invariant %.4f exceeds a layer index -- the ' ...
     'ray is beyond the critical angle inside the stack.'], inv_s);
% quarter waves per layer: 'qw' = 1 everywhere; 'symmetric' halves the two
% OUTER H layers, which is what takes the p slab from odd to even quarter
% waves (absentee) and keeps the stack symmetric.
qw = ones(nlayer,1);
if strcmp(opts.design,'symmetric'), qw([1 end]) = 0.5; end
cth   = sqrt(1 - (inv_s./nseq).^2);
thk_w = qw./(4*cth);                       % n*d/lambda for a QW at angle
thk   = opts.lambda*thk_w./nseq;           % physical thickness

th_H = asind(inv_s/opts.nH);
th_L = asind(inv_s/opts.nL);

S = struct();
S.n_glass    = n_g;
S.n_glass_mn = n_mn;
S.layers     = [nseq, kseq, thk_w];
S.thk        = thk;
S.theta      = acosd(cth);
S.theta_H    = th_H;
S.theta_L    = th_L;
S.brewster_resid = th_H + th_L - 90;
S.design     = opts.design;
S.nperiod    = opts.nperiod;
S.qw_total   = sum(qw);              % p slab thickness in quarter waves:
                                     %  EVEN = absentee for p (R_p == 0)
S.lambda     = opts.lambda;
S.aoi        = opts.aoi;
S.rt = macos.design.thinfilm_rt([nseq - 1i*kseq, thk], n_g, n_g, opts.aoi, opts.lambda);
end
