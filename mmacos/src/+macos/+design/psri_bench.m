function G = psri_bench(opts)
%MACOS.DESIGN.PSRI_BENCH  A buildable phase-shifting self-referenced
%   interferometer (P/SRI, Dube et al. 2024) on the DM-gauge bench: the
%   TG96 front end (source, collimator, 7 deg splitter, the 96 mm DM on
%   its leg, back through the splitter) followed by a Mach-Zehnder whose
%   REFERENCE arm carries the point-diffraction pinhole (the waveguide
%   seat) and the phase shifter, and whose TEST arm carries the beam
%   unfiltered; the two arms are recombined and sent down the same
%   pupil-imaging tail to one camera.  Two Bench decks (test, reference),
%   as the Twyman-Green is built, each traced on its own.
%
%     common   source -> baffle -> L1 -> BS1 reflect -> Comp -> DM (retro)
%              -> Comp -> BS1 transmit -> Recomb plane
%     TEST     -> BS2 transmit (top leg, +x) -> [lens-glass compensator]
%              -> M1 fold (-y, right leg) -> BS3 REFLECT (front face) -> tail
%     REF      -> BS2 REFLECT (front face, -y, left leg) -> Lr1 (focus)
%              -> entrance sphere (NF1) -> PINHOLE seat (NF2) -> exit
%              sphere -> Lr2 (recollimate) -> M3 fold (+x, bottom leg)
%              -> BS3 transmit -> tail
%     tail     L2 (focus) -> seat (empty) -> field lens -> camera at the
%              DM image (test-arm conjugate; the reference deck copies the
%              same distances)
%
%   Balance: each arm transmits ONE plate (BS2 or BS3) and reflects off
%   the other's front face, so the plate glass is equal; the reference
%   arm's two lenses are balanced by a normal-incidence plate in the test
%   arm whose thickness is solved so the two chief-ray optical paths from
%   the source to the camera are EQUAL (G.balance); the reference arm's
%   bottom fold is solved so the two exit chiefs COINCIDE after BS3
%   (G.balance.exit_offset_mm).  The pinhole sits at the reference lens's
%   true focus when REF_TRIM is set (psri_layout_fig scans it, as the
%   ZWFS's MASK_TRIM was found).
%
%   Returns G: .bt .br (Bench objects), .T .R (indices: iTO, iRC, iBS2,
%   iM1, iBS3, iL2, iSEAT, iFL, iDET; R also iLR1, iPIN, iLR2, iCHK, iM3),
%   .bs2 .bs3 (plate tokens), .det_leg, .balance, .P.  Emit with
%   G.bt.emit(file) / G.br.emit(file).  See also twyman_green.
arguments
    % ---- front end (twyman_green's, the ZWFS record values) ----------
    opts.F1 (1,1) double = 857.1428571428571
    opts.F2 (1,1) double = 428.5714285714286
    opts.BS_AOI (1,1) double {mustBePositive} = 7
    opts.D_LENS (1,1) double = 102.8571428571429
    opts.N_GLASS (1,1) double = 1.5
    opts.R_BAFFLE (1,1) double = 21.42857142857143
    opts.D_SB (1,1) double = 428.5714285714286
    opts.FILL (1,1) double = 0.95
    opts.BS_T (1,1) double = 2.571428571428571
    opts.D_L1_BS (1,1) double = 257.1428571428571
    opts.D_BS_TO (1,1) double = 700
    opts.D_BS_CMP (1,1) double = 171.4285714285714
    opts.D_RECOMB (1,1) double = 5
    opts.R_TO_AP (1,1) double = 51.42857142857143
    opts.L1_Kr (1,1) double = 406.0560000000000
    opts.L1_Kc (1,1) double = -0.5829
    opts.L2_Kr (1,1) double = -212.7017142857143
    opts.L2_Kc (1,1) double = -0.5826
    opts.ngridpts (1,1) double = 63
    opts.to_Kr (1,1) double = 0
    opts.to_grid_file (1,:) char = ''
    opts.to_grid_n (1,1) double = 0
    opts.to_grid_dx (1,1) double = 0
    % ---- the Mach-Zehnder -------------------------------------------
    opts.D_RC_BS2 (1,1) double = 60      % Recomb plane -> BS2 (the split)
    opts.MZ_T (1,1) double = 2.571428571428571   % BS2 / BS3 plate thickness
    opts.MZ_C (1,1) double = 320         % top leg: BS2 -> M1 (test arm)
    opts.MZ_A (1,1) double = 120         % left leg: BS2 -> Lr1 powered face (ref arm)
    opts.F_REF (1,1) double = 300        % reference lenses' focal length (F/2.9 on the 103 mm beam)
    opts.D_REF_OUT (1,1) double = 25     % exit sphere's station behind the pinhole
    opts.MZ_B (1,1) double = 120         % Lr2 powered face -> M3 (ref arm)
    opts.REF_TRIM (1,1) double = 0       % pinhole seat trim to the true focus (scan it)
    opts.LR1_Kc (1,1) double = NaN       % Lr1 conic (NaN = the add_lens seed); psri_layout_fig solves it
    opts.LR2_Kc (1,1) double = NaN       % Lr2 conic (NaN = seed); solved on the reference wavefront
    opts.M_APRAD (1,1) double = 75       % fold-mirror half-aperture (sketch / bodies)
    opts.D_COMP (1,1) double = 60        % compensator station on the test arm's right leg
    % ---- the tail (twyman_green 'fieldlens' arch, the tuned values) --
    opts.D_BS3_L2 (1,1) double = 200     % BS3 -> L2 powered face
    opts.MASK_TRIM (1,1) double = -5.582
    opts.FL_F (1,1) double = 42.5325
    opts.FL_Kc (1,1) double = -2.58764
    opts.FL_D (1,1) double = 20.57142857142857
    opts.D_MASK_FL (1,1) double = 39.7694
    opts.DET_TRIM (1,1) double = -1.2473
end
P = opts;
turn = 180 - 2*P.BS_AOI;  bs1_out = [cosd(turn); -sind(turn); 0];
% The Mach-Zehnder is laid out in the frame of the chief AFTER the DM
% return (the 7 deg splitter folds it by 166 deg, so it is not +x): 'across'
% = that chief, 'down' = its in-plane normal on the -y side.
[bt, T, bs2, bs3, det_leg, fr] = build_test(P, bs1_out, 0, NaN);
% ===== the reference arm: solve the bottom fold so the exit chiefs coincide
c_exit = dot(bt.E(T.iBS3).vpt, fr.down);            % the test arm's exit chief, 'down' coordinate
[br, R] = build_ref(P, bs1_out, fr, bs2, bs3, bt, T, det_leg, c_exit);
dc = dot(br.E(R.iBS3(2)).vpt, fr.down) - c_exit;    % exit-chief offset after the transmit (walk-off)
[br, R] = build_ref(P, bs1_out, fr, bs2, bs3, bt, T, det_leg, c_exit - dc);
% ===== the compensator: null the chief optical path difference ==========
oplA = chief_opl(bt, T.iDET);  oplB = chief_opl(br, R.iDET);
t_comp = (oplB - oplA) / (P.N_GLASS - 1);          % normal-incidence plate in the TEST arm
assert(t_comp > 0, 'psri_bench: the reference arm is the shorter one (dOPL %.3f mm); shorten MZ_B or lengthen MZ_A', oplB - oplA);
[bt, T, bs2, bs3, det_leg, fr] = build_test(P, bs1_out, t_comp, NaN);
[br, R] = build_ref(P, bs1_out, fr, bs2, bs3, bt, T, det_leg, c_exit - dc);
oplA = chief_opl(bt, T.iDET);  oplB = chief_opl(br, R.iDET);
exit_off = br.E(R.iBS3(2)).vpt - bt.E(T.iBS3).vpt;
det_off = br.E(R.iDET).vpt - bt.E(T.iDET).vpt;
bal = struct('opl_test_mm', oplA, 'opl_ref_mm', oplB, 'dopl_mm', oplB - oplA, ...
             't_comp_mm', t_comp, 'exit_offset_mm', norm(exit_off - dot(exit_off, fr.across)*fr.across), ...
             'det_offset_mm', norm(det_off), 'ref_dir_at_exit', br.E(R.iBS3(2)).psi.', ...
             'test_dir_at_exit', bt.E(T.iL2).psi.');
G = struct('bt', bt, 'br', br, 'T', T, 'R', R, 'bs2', bs2, 'bs3', bs3, ...
           'det_leg', det_leg, 'balance', bal, 'frame', fr, 'P', P);
end

% =====================================================================
function b = front_end(P, name)
AP = 2*atan(P.R_BAFFLE/P.D_SB)*P.FILL;
b = macos.design.Bench(name, 'aperture', AP, 'ngridpts', P.ngridpts);
b.add_baffle(P.D_SB, P.R_BAFFLE);
L1 = b.add_lens(P.F1 - P.D_SB, P.F1, P.D_LENS, 'mode','collimate', 'n',P.N_GLASS, 'name','L1');
b.E(L1.i_pow).Kr = P.L1_Kr;  b.E(L1.i_pow).Kc = P.L1_Kc;
end

function [ix, iRC] = common_arm(b, P, bs1_out, ix)
% BS1 reflect -> compensator -> DM (retro) -> compensator -> BS1 transmit -> Recomb
[~, bs1] = b.add_bs_reflect(P.D_L1_BS, bs1_out, 'thickness',P.BS_T, 'n',P.N_GLASS);
cmp = b.plate(P.D_BS_CMP, bs1.psi, 'thickness',P.BS_T, 'n',P.N_GLASS, 'name','Comp');
b.add_bs_transmit(cmp, 'tag','d');
leg_to = P.D_BS_TO - P.D_BS_CMP - P.BS_T;
ix.iTO = b.add_mirror(leg_to, 'name','TestOptic', 'aprad',P.R_TO_AP, 'Kr',P.to_Kr, ...
    'grid_file',P.to_grid_file, 'grid_n',P.to_grid_n, 'grid_dx',P.to_grid_dx);
b.add_bs_transmit(cmp, 'tag','u');
b.add_bs_transmit(bs1, 'tag','o');
iRC = b.add_reference(P.D_RECOMB, 'Recomb');  ix.iRC = iRC;
end

function [bt, T, bs2, bs3, det_leg, fr] = build_test(P, bs1_out, t_comp, ~)
bt = front_end(P, 'psri_test');  T = struct();
[T, ~] = common_arm(bt, P, bs1_out, T);
d = bt.dir;  out_px = d;  out_dn = [d(2); -d(1); 0];   % the MZ frame: across = the chief, down = its -y-side normal
fr = struct('across', out_px, 'down', out_dn);
% BS2: the plate token (its reflect lives in the reference deck; here a transmit)
psi2 = macos.design.Bench.unit(out_dn - bt.dir);
bs2 = bt.plate(P.D_RC_BS2, psi2, 'thickness',P.MZ_T, 'n',P.N_GLASS, 'name','BS2');
T.iBS2 = bt.add_bs_transmit(bs2, 'tag','t');
% top leg to M1, fold down the right leg
T.iM1 = bt.add_fold(P.MZ_C, out_dn, 'name','M1');
bt.E(T.iM1).aptype = 'Circular';  bt.E(T.iM1).aprad = P.M_APRAD;
leg = P.MZ_A + 2*P.F_REF + P.MZ_B;                 % nominal right-leg length (matched to the left leg)
d_bs3 = leg;
if t_comp > 0
    % the lens-glass compensator: a normal-incidence plate on the right leg
    cp = bt.plate(P.D_COMP, -bt.dir, 'thickness',t_comp, 'n',P.N_GLASS, 'name','CompLens');
    T.iCOMP = bt.add_bs_transmit(cp, 'tag','');
    d_bs3 = leg - P.D_COMP - t_comp;
end
% BS3: the test arm reflects off its front face to +x
[T.iBS3, bs3] = bt.add_bs_reflect(d_bs3, out_px, 'thickness',P.MZ_T, 'n',P.N_GLASS, 'name','BS3');
[T, det_leg] = tail(bt, P, T, T.iTO, []);
end

function [br, R] = build_ref(P, bs1_out, fr, bs2, bs3, bt, T, det_leg, c_m3)
br = front_end(P, 'psri_ref');  R = struct();
[R, ~] = common_arm(br, P, bs1_out, R);
out_dn = fr.down;  out_px = fr.across;
% BS2 reflect (front face) down the left leg
[R.iBS2, tok2] = br.add_bs_reflect(P.D_RC_BS2, out_dn, 'thickness',P.MZ_T, 'n',P.N_GLASS, 'name','BS2');
assert(norm(tok2.vpt - bs2.vpt) < 1e-9 && norm(tok2.psi - bs2.psi) < 1e-9, 'psri_bench: BS2 tokens differ between the arms');
% Lr1 focuses onto the pinhole seat inside the sphere bracket
a1 = {};  if ~isnan(P.LR1_Kc), a1 = {'Kc', P.LR1_Kc}; end
Lr1 = br.add_lens(P.MZ_A, P.F_REF, P.D_LENS, 'mode','focus', 'n',P.N_GLASS, 'name','Lr1', a1{:});
R.iLR1 = Lr1.i_pow;
dpin = P.F_REF - Lr1.thickness + P.REF_TRIM;       % thin-lens focus from the flat back, trimmed to the true one
d_in = 0.85*dpin;
R.iSPHIN = br.add_reference(dpin - d_in, 'RefSphereIn', 'surface','Conic', 'kr',-d_in, 'proptype','NF1', 'zelt',d_in);
R.iPIN = br.add_reference(d_in, 'Pinhole', 'proptype','NF2', 'zelt',1e22);
R.iSPHOUT = br.add_reference(P.D_REF_OUT, 'RefSphereOut', 'surface','Conic', 'kr',-d_in, 'zelt',d_in);
% Lr2 recollimates, placed MIRROR-SYMMETRIC to Lr1 about the pinhole (its flat
% face as far behind the focus as Lr1's flat face is before it, trim
% included): the reverse of Lr1's path, so Lr1's conic recollimates exactly
a2 = {};  if ~isnan(P.LR2_Kc), a2 = {'Kc', P.LR2_Kc}; end
Lr2 = br.add_lens(dpin - P.D_REF_OUT + Lr1.thickness, P.F_REF, P.D_LENS, 'mode','collimate', 'n',P.N_GLASS, 'name','Lr2', a2{:});
R.iLR2 = Lr2.i_pow;
% a passive plane normal to the recollimated reference (where its wavefront
% is measured -- the fold's tilted surface would read geometry, not OPD)
R.iCHK = br.add_reference(10, 'RefCollimated');
% M3 at the solved 'down' coordinate, fold to 'across' along the bottom leg
d_m3 = c_m3 - dot(br.pos, out_dn);
assert(d_m3 > P.MZ_B/4, 'psri_bench: M3 would sit inside Lr2 (d %.1f mm); lengthen MZ_C or shorten the reference leg', d_m3);
R.iM3 = br.add_fold(d_m3, out_px, 'name','M3');
br.E(R.iM3).aptype = 'Circular';  br.E(R.iM3).aprad = P.M_APRAD;
% BS3 transmit (the token the test arm reflected off)
R.iBS3 = br.add_bs_transmit(bs3, 'tag','r');
% the tail at the test arm's stations
[R, ~] = tail(br, P, R, [], det_leg, bt, T);
end

function [ix, det_leg] = tail(b, P, ix, conj_elt, det_leg, bt, T)
% L2 -> empty seat -> field lens -> camera at the DM image (twyman_green's
% 'fieldlens' arch, plain seat); the reference deck places L2 at the test
% deck's L2 point and copies its distances
if nargin >= 7
    d_l2 = dot(bt.E(T.iL2).vpt - b.pos, b.dir);
    assert(d_l2 > 0, 'psri_bench: L2 plane behind the reference exit');
else
    d_l2 = P.D_BS3_L2;
end
L2 = b.add_lens(d_l2, P.F2, P.D_LENS, 'mode','focus', 'n',P.N_GLASS, 'name','L2');
b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
ix.iL2 = L2.i_pow;
dmask = P.F2 - L2.thickness + P.MASK_TRIM;
ix.iSEAT = b.add_reference(dmask, 'Seat');
FL = b.add_lens(P.D_MASK_FL, P.FL_F, P.FL_D, 'mode','focus', 'n',P.N_GLASS, 'name','FL', 'Kc',P.FL_Kc);
ix.iFL = FL.i_pow;
if ~isempty(conj_elt)
    s_o  = b.E(L2.i_pow).s - b.E(conj_elt).s;
    s_i1 = 1/(1/P.F2 - 1/s_o);
    d12  = b.E(FL.i_pow).s - b.E(L2.i_pow).s;
    s_o2 = d12 - s_i1;
    s_i2 = 1/(1/P.FL_F - 1/s_o2);
    det_leg = s_i2 - FL.thickness + P.DET_TRIM;
end
ix.iDET = b.add_detector(det_leg, 'Detector');
end

function opl = chief_opl(b, iend)
% optical path of the chief from the source to element iend: segment
% lengths between consecutive vertices times the index of the medium the
% segment runs in (E(k).indref = the index after element k)
opl = 0;  p = b.src_pos;  n = 1;
for k = 1:iend
    q = b.E(k).vpt;  opl = opl + n*norm(q - p);  p = q;  n = b.E(k).indref;
end
end
