function G = twyman_green(opts)
%MACOS.DESIGN.TWYMAN_GREEN  Build a compensated Twyman-Green IFO rig.
%   G = macos.design.twyman_green(...) builds BOTH arms of the generic
%   Twyman-Green interferometer (the examples/design/bench_ifo layout)
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
%   See also: macos.design.Bench, examples/design/bench_ifo.

arguments
    opts.F1 (1,1) double = 500
    opts.F2 (1,1) double = 250
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
end
P = opts;

% ---- test arm -------------------------------------------------------
bt = front_end(P, 'ifo_test');
[~, bs] = bt.add_bs_reflect(P.D_L1_BS, [0;-1;0], 'thickness',P.BS_T, 'n',P.N_GLASS);
cmp = bt.plate(P.D_BS_CMP, bs.psi, 'thickness',P.BS_T, 'n',P.N_GLASS, 'name','Comp');
bt.add_bs_transmit(cmp, 'tag','d');
T.iTO = bt.add_mirror(P.D_BS_TO - P.D_BS_CMP - P.BS_T, 'name','TestOptic', ...
    'aprad',P.R_TO_AP, 'Kr',P.to_Kr, 'grid_file',P.to_grid_file, ...
    'grid_n',P.to_grid_n, 'grid_dx',P.to_grid_dx);
bt.add_bs_transmit(cmp, 'tag','u');
bt.add_bs_transmit(bs, 'tag','o');
T.iRC = bt.add_reference(P.D_RECOMB, 'Recomb');
[T, det_leg] = tail(bt, P, T, T.iTO, []);

% ---- reference arm --------------------------------------------------
br = front_end(P, 'ifo_ref');
br.add_bs_transmit(bs, 'tag','f');
R.iPZT = br.add_mirror(P.D_BS_TO, 'name','PZT');
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
switch P.tail_arch
case 'singlet'                     % original architecture (default)
    L2 = b.add_lens(P.D_RC_L2, P.F2, P.D_LENS, 'mode','focus', ...
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
    L2 = b.add_lens(P.D_RC_L2, P.F2, P.D_LENS, 'mode','focus', ...
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
    A = b.add_lens(P.D_RC_L2, P.L2A_F, P.D_LENS, aA{:});
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
