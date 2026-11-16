function R = ctb_fold_build(tel_in, stem, P, K)
%CTB_FOLD_BUILD  The CTB behind a telescope, with the knobs that fold it into the shroud.
%   R = CTB_FOLD_BUILD(TEL_IN, STEM, P, K) builds the CTB chain of
%   templates/30_instruments/bench_ctb (example_ctb's PARAMETER BLOCK P,
%   lengths in mm: 8 OAPs at P.AOI, two flat DMs, the mask markers) behind
%   the telescope deck TEL_IN, seeded on its exit chief ray 'back' metres
%   before its focus, with OAP1's focal length matched to the telescope
%   (f/# x the DM pupil diameter) and DM1 at the telescope stop's image --
%   the mmacos manual's chapter-2 "case C" -- and then FOLDED by the knobs
%   in K (every one of them off by default, which reproduces case C as the
%   chapter drew it, unfolded in the CTB's own plane):
%
%     K.fold0_m     a flat fold on the CONVERGING feed, this many metres
%                   before the telescope's focus (0 = none).  It turns the
%                   chain back toward the launch axis: the focus and the
%                   whole accordion then lie on the fold's far side.
%     K.fold0_tilt_deg  the fold's deviation from exact retro-reflection,
%                   taken OUT of the telescope's bench plane (+ = toward
%                   +z, aft of the primary); its AOI is half of this.
%     K.plane_deg   the accordion's plane: the fold axis of every CTB
%                   turn is  cos(plane) * z_hat + sin(plane) * h_hat,
%                   h_hat = unit(z_hat x d_feed).  0 = the bench plane
%                   (the template's zigzag); 90 = the vertical plane
%                   through the feed, so the accordion's drift is in z.
%     K.sign0       +1 / -1: the sense of the first CTB turn (OAP1); the
%                   turns alternate after it, the ping-pong that keeps the
%                   accordion in a leg-sized pocket (e2e6m_r2's lesson:
%                   a non-alternating pattern fans the chain).
%     K.flip_at     0, or the CTB turn (1 = OAP1, 2 = DM1, ...) AFTER which
%                   the alternation is inverted: the next turn repeats the
%                   sense of the one before it, which mirrors the ping-pong
%                   from there on and REVERSES the accordion's drift (the
%                   legs still alternate between the same two directions'
%                   mirror pair; nothing fans).  A fold-down after a
%                   fold-up halves the chain's height.
%     K.relay_scale scale on the relay focal lengths F_OAP(2:8) AND on the
%                   collimated legs after DM2 (L_O2, L_APOD, L_O4, L_LYOT,
%                   L_O6, L_BACK, L_O8); 1 = the template's relay.  The
%                   pupil relays' magnifications are ratios, unchanged;
%                   the physical size of a lambda/D at every focus scales
%                   with it (reported: R.lamD_fpm_um).  OAP1 -> DM1 (the
%                   stop image) and DM1 -> DM2 (the two-DM spacing, the
%                   amplitude lever) are NOT scaled.
%     K.back_m      the bench's seed before the focus (0.30)
%     K.fno         the telescope's marginal-ray f/# (measured when NaN)
%     K.dm1_m       OAP1 -> DM1 in metres (NaN = the template's L_DM1;
%                   case C passes the measured stop image)
%     K.model       engine model size for the measurements (256)
%
%   Products: <stem>_ctb.in (the bench alone, seeded at the telescope's
%   focus) and <stem>_full.in (spliced onto the telescope with
%   macos.design.append_rx, the telescope's terminal quartet dropped).
%   R: .bench_in .full_in .ix (element indices in the FULL deck: OAP1 DM1
%   DM2 OAP2..OAP8 FPM Lyot FPA, and F0 when present) .names (every CTB
%   element, in order) .n_base .pF .dF .fno .fOAP1 .Dpup .lamD_fpm_um
%   .K (the knobs as used) .P.
%
%   The chain is example_ctb's build (its rot_axis helper reproduced here,
%   the template file itself untouched); the AOI at every OAP and DM is
%   P.AOI by construction, and the added fold's AOI is fold0_tilt_deg/2.
%
%   See also CTB_FOLD_MEASURE, DECK_CLEARANCE, SHROUD_DECK, APPEND_RX.

    arguments
        tel_in (1,:) char
        stem   (1,:) char
        P      struct
        K      struct = struct()
    end
    D = struct('fold0_m', 0, 'fold0_tilt_deg', 10, 'plane_deg', 0, 'sign0', +1, 'flip_at', 0, ...
               'relay_scale', 1, 'back_m', 0.30, 'fno', NaN, 'dm1_m', NaN, 'model', 256);
    fn = fieldnames(D);
    for k = 1:numel(fn), if ~isfield(K, fn{k}), K.(fn{k}) = D.(fn{k}); end, end
    mm = 1e-3;

    % ---- the telescope's exit state ------------------------------------
    macos.init(K.model);
    nT = macos.load_rx(tel_in);
    r1 = macos.get_ray_info(macos.trace(1).nRays);
    sT = macos.trace(nT);  rT = macos.get_ray_info(sT.nRays);
    ok = r1.ok_pass & r1.ok_trace & rT.ok_pass & rT.ok_trace;
    pF = rT.pos(:,1);  dF = rT.dir(:,1)/norm(rT.dir(:,1));
    if isnan(K.fno)
        th = acos(min(1, dF.' * (rT.dir(:,ok)./vecnorm(rT.dir(:,ok)))));
        K.fno = 1/(2*sin(max(th)));
    end
    Dpup  = P.R_DM*P.FILL*mm;                         % example_ctb: R_DM*FILL lands as the DIAMETER
    fOAP1 = K.fno*Dpup;
    LDM1  = K.dm1_m;  if isnan(LDM1), LDM1 = P.L_DM1*mm; end

    % ---- the relay, scaled ----------------------------------------------
    s = K.relay_scale;
    Fo = P.F_OAP*mm;  Fo(1) = fOAP1;  Fo(2:8) = Fo(2:8)*s;
    r  = Fo / cosd(P.AOI)^2;
    Lg = struct('O2', P.L_O2, 'APOD', P.L_APOD, 'O4', P.L_O4, 'LYOT', P.L_LYOT, ...
                'O6', P.L_O6, 'BACK', P.L_BACK, 'O8', P.L_O8);
    g = fieldnames(Lg);
    for k = 1:numel(g), Lg.(g{k}) = Lg.(g{k})*mm*s; end
    LDM2 = P.L_DM2*mm;
    Ra = P.R_OAP*mm;  Rd = P.R_DM*mm;

    % ---- the fold axis of the accordion and the added fold --------------
    zh = [0;0;1];  hh = cross(zh, dF);
    if norm(hh) < 1e-9, hh = cross([1;0;0], dF); end
    hh = hh/norm(hh);
    nf = cosd(K.plane_deg)*zh + sind(K.plane_deg)*hh;  nf = nf/norm(nf);
    turn = @(d, sg) rot_axis_(d, nf, sg*(180 - 2*P.AOI));

    back = K.back_m;
    a0 = K.fold0_m;
    if a0 > 0 && a0 >= back - 0.02, back = a0 + 0.05; end   % the seed must precede the fold
    b = macos.design.Bench([stem '_ctb'], 'baseunits', 'm', 'pos', pF - back*dF, 'dir', dF, ...
            'wavelen', P.WAVLEN*mm, 'aperture', 2*atan(1/(2*K.fno)), 'ngridpts', 63, 'zsource', back);
    names = {};  ix = struct();
    dist_to_focus = back;                              % along the chief from the bench's current pos
    if a0 > 0
        % a flat on the converging feed: retro-reflection deviated by the
        % tilt, out of the bench plane (about h_hat, toward +z for + tilt)
        da = rot_axis_(-dF, hh, K.fold0_tilt_deg);  db = rot_axis_(-dF, hh, -K.fold0_tilt_deg);
        if K.fold0_tilt_deg >= 0, if da(3) >= db(3), d1 = da; else, d1 = db; end
        else,                     if da(3) <= db(3), d1 = da; else, d1 = db; end, end
        ix.F0 = b.add_fold(back - a0, d1, 'name', 'F0');  names{end+1} = 'F0';
        dist_to_focus = a0;
    end
    % the turn senses: alternating from sign0; after turn flip_at the next
    % turn repeats its predecessor's sense, then alternation resumes
    SG = zeros(1, 11);  SG(1) = K.sign0;
    for k = 2:11
        if K.flip_at > 0 && k == K.flip_at + 1, SG(k) = SG(k-1); else, SG(k) = -SG(k-1); end
    end
    kt = 0;
    d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    O1 = b.add_oap(dist_to_focus + r(1), turn(d0, sg), 'mode','collimate','f',Fo(1),'name','OAP1','aprad',Ra);
    ix.OAP1 = O1.i;  names{end+1} = 'OAP1';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.DM1 = b.add_mirror(LDM1, 'out', turn(d0, sg), 'name','DM1','aprad',Rd);  names{end+1} = 'DM1';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.DM2 = b.add_mirror(LDM2, 'out', turn(d0, sg), 'name','DM2','aprad',Rd);  names{end+1} = 'DM2';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    O = b.add_oap(Lg.O2, turn(d0, sg), 'mode','focus','f',Fo(2),'name','OAP2','aprad',Ra);  ix.OAP2 = O.i;  names{end+1} = 'OAP2';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    b.add_reference(r(2), 'Focus23');  names{end+1} = 'Focus23';
    O = b.add_oap(r(3), turn(d0, sg), 'mode','collimate','f',Fo(3),'name','OAP3','aprad',Ra);  ix.OAP3 = O.i;  names{end+1} = 'OAP3';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.Apod = b.add_reference(Lg.APOD, 'Apodizer');  names{end+1} = 'Apodizer';
    O = b.add_oap(Lg.O4, turn(d0, sg), 'mode','focus','f',Fo(4),'name','OAP4','aprad',Ra);  ix.OAP4 = O.i;  names{end+1} = 'OAP4';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.FPM = b.add_reference(r(4), 'FPM');  names{end+1} = 'FPM';
    O = b.add_oap(r(5), turn(d0, sg), 'mode','collimate','f',Fo(5),'name','OAP5','aprad',Ra);  ix.OAP5 = O.i;  names{end+1} = 'OAP5';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.Lyot = b.add_reference(Lg.LYOT, 'Lyot');  names{end+1} = 'Lyot';
    O = b.add_oap(Lg.O6, turn(d0, sg), 'mode','focus','f',Fo(6),'name','OAP6','aprad',Ra);  ix.OAP6 = O.i;  names{end+1} = 'OAP6';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.FieldStop = b.add_reference(r(6), 'FieldStop');  names{end+1} = 'FieldStop';
    O = b.add_oap(r(7), turn(d0, sg), 'mode','collimate','f',Fo(7),'name','OAP7','aprad',Ra);  ix.OAP7 = O.i;  names{end+1} = 'OAP7';  d0 = b.dir;  kt = kt + 1;  sg = SG(kt);
    ix.Backend = b.add_reference(Lg.BACK, 'Backend');  names{end+1} = 'Backend';
    O = b.add_oap(Lg.O8, turn(d0, sg), 'mode','focus','f',Fo(8),'name','OAP8','aprad',Ra);  ix.OAP8 = O.i;  names{end+1} = 'OAP8';
    ix.FPA = b.add_detector(r(8), 'FPA');  names{end+1} = 'FPA';

    bench_in = [stem '_ctb.in'];  b.emit(bench_in);
    full_in  = [stem '_full.in'];
    info = macos.design.append_rx(tel_in, bench_in, full_in, 'drop_base_tail', 3);
    f = fieldnames(ix);
    for k = 1:numel(f), ix.(f{k}) = info.n_base + ix.(f{k}); end
    R = struct('bench_in', bench_in, 'full_in', full_in, 'ix', ix, 'names', {names}, ...
               'n_base', info.n_base, 'n_out', info.n_out, 'pF', pF, 'dF', dF, 'fno', K.fno, ...
               'fOAP1', fOAP1, 'Dpup', Dpup, 'lamD_fpm_um', Fo(4)*P.WAVLEN*mm/Dpup*1e6, ...
               'K', K, 'P', P, 'bench', b, 'nf', nf, 'hh', hh);
end

function d = rot_axis_(d0, ax, ang_deg)
%ROT_AXIS_  Rodrigues rotation of d0 about unit axis AX by ANG_DEG (example_ctb's helper).
    d0 = d0(:);  ax = ax(:)/norm(ax(:));  a = deg2rad(ang_deg);
    d = d0*cos(a) + cross(ax, d0)*sin(a) + ax*(ax.'*d0)*(1 - cos(a));
    d = d/norm(d);
end
