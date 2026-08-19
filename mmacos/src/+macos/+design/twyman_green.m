function G = twyman_green(opts)
%MACOS.DESIGN.TWYMAN_GREEN  Build a compensated Twyman-Green IFO rig.
%   G = macos.design.twyman_green(...) builds BOTH arms of the generic
%   Twyman-Green interferometer (the templates/40_benches/bench_ifo layout)
%   with the Bench add-optic utilities and returns them ready to emit:
%
%     TEST ARM  source -> baffle -> L1 -> BS reflect (45 deg,
%               front-coated plate) -> compensator (double-passed) ->
%               TEST OPTIC (retro) -> BS transmit -> Recomb -> L2 ->
%               focal mask -> detector at the test-optic pupil image.
%     REF ARM   same front end -> BS transmit -> PZT (retro) -> BS
%               internal-reflect return -> same Recomb plane -> same
%               output train to the same detector plane.
%
%   Glass paths balance exactly (identical plate + compensator, real
%   internal V on the reference return); the flat-test-optic null is
%   ~2.5e-9 rad.  See example_bench_ifo.m for the fully-annotated
%   walk-through and the PSI processing that runs on this rig.
%
%   The TEST OPTIC can carry the "unknown":
%     'to_Kr'         weak-sphere figure (Surface=Conic vertex radius)
%     'to_grid_file'/'to_grid_n'/'to_grid_dx'
%                     a GridData figure map (e.g. a DM surface built
%                     with PROPER) in the optic's own local frame.
%
%   All lengths mm.  Returns struct G:
%     .bt .br     test / reference Bench objects (call .emit yourself)
%     .T  .R      element-index structs (.iTO/.iRC/.iMASK/.iDET; .iPZT)
%     .bs         the shared BS plate token
%     .det_leg    detector leg length (shared plane)
%     .P          the resolved parameter struct
%
%   See also: macos.design.Bench, templates/40_benches/bench_ifo.

arguments
    opts.F1 (1,1) double = 500
    opts.F2 (1,1) double = 250
    % ---- beam-splitter angle of incidence (slice-2 AOI/clearance trade) --
    % BS_AOI is the chief AOI on the beam-splitter, degrees.  The fold turn
    % angle is 180 - 2*BS_AOI, so BS_AOI=45 is the canonical right-angle
    % fold.  Every downstream face (compensator, BS substrate transits, the
    % internal-reflect return) tracks the BS normal automatically through
    % the shared bs token and the recomputed chief -- only the reflect
    % turn direction is set here.  DEFAULT 45 emits BIT-IDENTICALLY to the
    % pre-slice-2 rig (the exact [0;-1;0] literal, no cosd(90) round-off).
    opts.BS_AOI (1,1) double {mustBePositive} = 45
    opts.D_LENS (1,1) double = 60
    opts.N_GLASS (1,1) double = 1.5
    opts.R_BAFFLE (1,1) double = 12.5
    opts.D_SB (1,1) double = 250
    opts.FILL (1,1) double = 0.95
    opts.BS_T (1,1) double = 1.5
    opts.D_L1_BS (1,1) double = 150
    opts.D_BS_TO (1,1) double = 250
    opts.D_BS_CMP (1,1) double = 100
    opts.D_RECOMB (1,1) double = 5
    opts.D_RC_L2 (1,1) double = 200
    opts.R_TO_AP (1,1) double = 30
    opts.L1_Kr (1,1) double = 236.866
    opts.L1_Kc (1,1) double = -0.5829
    opts.L2_Kr (1,1) double = -124.076
    opts.L2_Kc (1,1) double = -0.5826
    opts.ngridpts (1,1) double = 63
    opts.to_Kr (1,1) double = 0
    opts.to_grid_file (1,:) char = ''
    opts.to_grid_n (1,1) double = 0
    opts.to_grid_dx (1,1) double = 0
    % ---- detector-leg (tail) architecture (l2_trade work; default is the
    % original singlet, bit-identical when these are omitted) -------------
    opts.tail_arch (1,:) char {mustBeMember(opts.tail_arch, ...
        {'singlet','fieldlens','doublet'})} = 'singlet'
    opts.DET_TRIM (1,1) double = 0       % additive trim on det_leg (all arches)
    opts.MASK_TRIM (1,1) double = 0      % additive trim on the FocalMask
                                         %  position (thin-lens seed -> true
                                         %  focus; det_leg re-chains from the
                                         %  moved mask, pupil conjugate kept)
    % 'fieldlens': pupil-relay field lens just behind the FocalMask
    opts.FL_F (1,1) double = 150         % field-lens focal length
    opts.FL_Kc (1,1) double = NaN        % powered-face conic (NaN = seed -n^2)
    opts.FL_D (1,1) double = 12          % field-lens diameter (beam at the
                                         %  focus is sub-mm; a small FL keeps
                                         %  the sag/thickness physical at
                                         %  short focal lengths)
    opts.D_MASK_FL (1,1) double = 5      % mask -> field lens distance
    % 'doublet': L2 split into two air-spaced plano singlets
    opts.L2A_F (1,1) double = 500        % front element focal length
    opts.L2B_F (1,1) double = 500        % back element focal length
    opts.L2_SEP (1,1) double = 25        % powered-surface separation
    opts.L2A_Kc (1,1) double = NaN       % conics (NaN = add_lens seed -n^2)
    opts.L2B_Kc (1,1) double = NaN
    % ---- polarizing phase-shifting variant (slice 3) --------------------
    % When 'polarizing' is true the builder inserts real TrPolarizer /
    % WavePlate elements to make a ROTATING-ANALYZER polarization PSI:
    %   input polarizer (both arms) -> [BS] -> a double-passed QWP in each
    %   arm (net half-wave, rotating that arm's linear state) -> [recomb] ->
    %   output QWP -> rotating analyzer -> detector.
    % The two arms leave orthogonally polarized; the output QWP maps them to
    % orthogonal circular, so an analyzer at angle t imposes a fringe phase
    % 2t -- stepping t = 0/45/90/135 deg is a four-step PSI with NO moving
    % PZT.  Every pol element sits in a COLLIMATED, NORMAL-INCIDENCE leg
    % (psi = chief), where the material-axis convention is identically
    % absent.  Axes are given as ANGLES (deg) in each leg's LOCAL transverse
    % plane (perp(dir), cross(dir,perp)) so "45 deg" is fold-correct.  The
    % analyzer axis is a default; the harness steps it at runtime.  DEFAULT
    % false emits BIT-IDENTICALLY to the non-polarizing rig (all insertions
    % gated, and each steals its standoff from the following leg so the BS,
    % test-optic, PZT and pupil conjugates are unmoved).
    opts.polarizing (1,1) logical = false
    opts.pol_in_deg   (1,1) double = 45     % input polarizer, both arms
    opts.qwp_test_deg (1,1) double = 0      % test-arm QWP fast axis
    opts.qwp_ref_deg  (1,1) double = 45     % ref-arm QWP fast axis
    opts.out_qwp_deg  (1,1) double = 0      % output QWP fast axis (shared leg)
    opts.analyzer_deg (1,1) double = 0      % analyzer default (stepped at run)
    opts.qwp_ret      (1,1) double = 0.25   % nominal QWP retardance (waves)
    opts.D_QWP        (1,1) double = 25     % arm-QWP standoff from the retro
    opts.D_POL        (1,1) double = 10     % input-polarizer / output-leg standoff
end
P = opts;

% ---- BS fold direction from the AOI ---------------------------------
% turn = 180 - 2*AOI about +z, applied to the +x chief toward -y.  Pin the
% 45-deg case to the exact literal so the default rig stays bit-identical
% (cosd(90) is 6.1e-17, which would perturb every emitted coordinate).
if abs(P.BS_AOI - 45) < 1e-12
    bs_out = [0; -1; 0];
else
    turn = 180 - 2*P.BS_AOI;
    bs_out = [cosd(turn); -sind(turn); 0];
end

% ---- test arm -------------------------------------------------------
bt = front_end(P, 'ifo_test');
% input polarizer in the collimated pre-BS leg (slice-3 variant); it steals
% its standoff from the L1->BS leg so the BS stays put (bit-identical off)
if P.polarizing
    bt.add_polarizer(P.D_POL, ax_local(bt.dir, P.pol_in_deg), 'name','PolIn');
    d_l1_bs = P.D_L1_BS - P.D_POL;
else
    d_l1_bs = P.D_L1_BS;
end
[~, bs] = bt.add_bs_reflect(d_l1_bs, bs_out, 'thickness',P.BS_T, 'n',P.N_GLASS);
cmp = bt.plate(P.D_BS_CMP, bs.psi, 'thickness',P.BS_T, 'n',P.N_GLASS, 'name','Comp');
bt.add_bs_transmit(cmp, 'tag','d');
leg_to = P.D_BS_TO - P.D_BS_CMP - P.BS_T;
if P.polarizing
    % double-passed QWP: SAME global fast axis both passes -> net half-wave,
    % rotating this arm's linear state.  The forward pass steals D_QWP from
    % the retro leg; the return pass rides the geometry-absolute comp transit.
    qa_t = ax_local(bt.dir, P.qwp_test_deg);
    bt.add_waveplate(P.D_QWP, qa_t, P.qwp_ret, 'name','QWPtestIn');
    leg_to = leg_to - P.D_QWP;
end
T.iTO = bt.add_mirror(leg_to, 'name','TestOptic', ...
    'aprad',P.R_TO_AP, 'Kr',P.to_Kr, 'grid_file',P.to_grid_file, ...
    'grid_n',P.to_grid_n, 'grid_dx',P.to_grid_dx);
if P.polarizing
    bt.add_waveplate(P.D_QWP, qa_t, P.qwp_ret, 'name','QWPtestOut');
end
bt.add_bs_transmit(cmp, 'tag','u');
bt.add_bs_transmit(bs, 'tag','o');
T.iRC = bt.add_reference(P.D_RECOMB, 'Recomb');
[T, det_leg] = tail(bt, P, T, T.iTO, []);

% ---- reference arm --------------------------------------------------
br = front_end(P, 'ifo_ref');
if P.polarizing
    br.add_polarizer(P.D_POL, ax_local(br.dir, P.pol_in_deg), 'name','PolIn');
end
br.add_bs_transmit(bs, 'tag','f');
leg_pzt = P.D_BS_TO;
if P.polarizing
    qa_r = ax_local(br.dir, P.qwp_ref_deg);
    br.add_waveplate(P.D_QWP, qa_r, P.qwp_ret, 'name','QWPrefIn');
    leg_pzt = leg_pzt - P.D_QWP;
end
R.iPZT = br.add_mirror(leg_pzt, 'name','PZT');
if P.polarizing
    br.add_waveplate(P.D_QWP, qa_r, P.qwp_ret, 'name','QWPrefOut');
end
br.add_bs_reflect_return(bs);
d_rc = dot(bt.E(T.iRC).vpt - br.pos, br.dir);
assert(d_rc > 0, 'twyman_green: recomb plane behind the reference return');
R.iRC = br.add_reference(d_rc, 'Recomb');
[R, ~] = tail(br, P, R, [], det_leg);

G = struct('bt',bt, 'br',br, 'T',T, 'R',R, 'bs',bs, 'det_leg',det_leg, 'P',P);
end

% ---------------------------------------------------------------------
function b = front_end(P, name)
    AP = 2*atan(P.R_BAFFLE/P.D_SB)*P.FILL;
    b = macos.design.Bench(name, 'aperture', AP, 'ngridpts', P.ngridpts);
    b.add_baffle(P.D_SB, P.R_BAFFLE);
    L1 = b.add_lens(P.F1 - P.D_SB, P.F1, P.D_LENS, 'mode','collimate', ...
                    'n',P.N_GLASS, 'name','L1');
    b.E(L1.i_pow).Kr = P.L1_Kr;  b.E(L1.i_pow).Kc = P.L1_Kc;
end

function [ix, det_leg] = tail(b, P, ix, conj_elt, det_leg)
%TAIL  Detector leg: Recomb -> (L2 architecture) -> FocalMask -> Detector at
%   the TEST-OPTIC pupil conjugate.  Shared by both arms (same det_leg), so
%   any architecture stays common-path.  The invariant every arch keeps:
%   the detector sits at the thin-lens pupil image of CONJ_ELT (plus
%   P.DET_TRIM, a knob for nulling the DM-tilt lever the thin-lens seed
%   leaves -- ~4.6 mm on the baseline singlet).
% Slice-3 output leg (both arms, before L2, collimated): output QWP then
% rotating analyzer.  They steal 2*D_POL from the Recomb->L2 leg so L2 and
% the pupil conjugate stay put.  ix.iOutQWP / ix.iAnalyzer are exposed.
d_rc_l2 = P.D_RC_L2;
if P.polarizing
    ix.iOutQWP   = b.add_waveplate(P.D_POL, ax_local(b.dir, P.out_qwp_deg), ...
                                   P.qwp_ret, 'name','OutQWP');
    ix.iAnalyzer = b.add_polarizer(P.D_POL, ax_local(b.dir, P.analyzer_deg), ...
                                   'name','Analyzer');
    d_rc_l2 = P.D_RC_L2 - 2*P.D_POL;
end
switch P.tail_arch
case 'singlet'                     % original architecture (default)
    L2 = b.add_lens(d_rc_l2, P.F2, P.D_LENS, 'mode','focus', ...
                    'n',P.N_GLASS, 'name','L2');
    b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
    ix.iMASK = b.add_reference(P.F2 - L2.thickness + P.MASK_TRIM, 'FocalMask');
    if ~isempty(conj_elt)
        s_o = b.E(L2.i_pow).s - b.E(conj_elt).s;
        s_i = 1/(1/P.F2 - 1/s_o);
        det_leg = s_i - (b.E(ix.iMASK).s - b.E(L2.i_pow).s) + P.DET_TRIM;
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');

case 'fieldlens'                   % C1: field lens just behind the mask
    L2 = b.add_lens(d_rc_l2, P.F2, P.D_LENS, 'mode','focus', ...
                    'n',P.N_GLASS, 'name','L2');
    b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
    ix.iMASK = b.add_reference(P.F2 - L2.thickness + P.MASK_TRIM, 'FocalMask');
    fl_args = {'mode','focus', 'n',P.N_GLASS, 'name','FL'};
    if ~isnan(P.FL_Kc), fl_args = [fl_args {'Kc', P.FL_Kc}]; end
    FL = b.add_lens(P.D_MASK_FL, P.FL_F, P.FL_D, fl_args{:});
    if ~isempty(conj_elt)
        s_o  = b.E(L2.i_pow).s - b.E(conj_elt).s;
        s_i1 = 1/(1/P.F2 - 1/s_o);                 % DM image via L2
        d12  = b.E(FL.i_pow).s - b.E(L2.i_pow).s;
        s_o2 = d12 - s_i1;                         % <0 = virtual object
        s_i2 = 1/(1/P.FL_F - 1/s_o2);
        det_leg = s_i2 - FL.thickness + P.DET_TRIM;
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');

case 'doublet'                     % C2: L2 as two air-spaced singlets
    aA = {'mode','focus', 'n',P.N_GLASS, 'name','L2A'};
    if ~isnan(P.L2A_Kc), aA = [aA {'Kc', P.L2A_Kc}]; end
    A = b.add_lens(d_rc_l2, P.L2A_F, P.D_LENS, aA{:});
    aB = {'mode','focus', 'n',P.N_GLASS, 'name','L2B'};
    if ~isnan(P.L2B_Kc), aB = [aB {'Kc', P.L2B_Kc}]; end
    gap = P.L2_SEP - A.thickness;
    assert(gap > 0, 'twyman_green: L2_SEP %.3g <= L2A thickness %.3g', ...
        P.L2_SEP, A.thickness);
    B = b.add_lens(gap, P.L2B_F, P.D_LENS, aB{:});
    % focus of the pair for collimated input (thin-lens seed; the runner
    % trims conics against the mask spot)
    s_iB = 1/(1/P.L2B_F - 1/(P.L2_SEP - P.L2A_F));
    assert(s_iB > B.thickness, 'twyman_green: doublet focus inside L2B');
    ix.iMASK = b.add_reference(s_iB - B.thickness + P.MASK_TRIM, 'FocalMask');
    if ~isempty(conj_elt)
        s_o  = b.E(A.i_pow).s - b.E(conj_elt).s;
        s_i1 = 1/(1/P.L2A_F - 1/s_o);
        d12  = b.E(B.i_pow).s - b.E(A.i_pow).s;
        s_o2 = d12 - s_i1;
        s_i2 = 1/(1/P.L2B_F - 1/s_o2);
        det_leg = (b.E(B.i_pow).s + s_i2) - b.E(ix.iMASK).s + P.DET_TRIM;
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');
end
end

% ---------------------------------------------------------------------
function a = ax_local(dir, deg)
%AX_LOCAL  A polarization axis at DEG (from the local x) in the transverse
%   plane of a beam travelling along DIR.  The local basis is
%   (u1, u2) = (perp(dir), cross(dir, perp(dir))), the SAME right-handed
%   transverse frame the Bench emitter uses for xObs, so "45 deg" means the
%   same physical direction in every folded leg.  Returned as a global
%   3-vector already in the transverse plane (exactly normal-incidence).
    u1 = macos.design.Bench.perp(dir(:));
    u2 = cross(dir(:), u1);
    a  = cosd(deg)*u1 + sind(deg)*u2;
end
