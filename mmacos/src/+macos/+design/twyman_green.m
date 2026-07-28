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
    L2 = b.add_lens(P.D_RC_L2, P.F2, P.D_LENS, 'mode','focus', ...
                    'n',P.N_GLASS, 'name','L2');
    b.E(L2.i_pow).Kr = P.L2_Kr;  b.E(L2.i_pow).Kc = P.L2_Kc;
    ix.iMASK = b.add_reference(P.F2 - L2.thickness, 'FocalMask');
    if ~isempty(conj_elt)
        s_o = b.E(L2.i_pow).s - b.E(conj_elt).s;
        s_i = 1/(1/P.F2 - 1/s_o);
        det_leg = s_i - (b.E(ix.iMASK).s - b.E(L2.i_pow).s);
    end
    ix.iDET = b.add_detector(det_leg, 'Detector');
end
