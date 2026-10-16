function ana = dmg_analyzer_maps(P, varargin)
%DMG_ANALYZER_MAPS  The vector Zernike sensor's analyzer leak, from the engine.
%   ana = dmg_analyzer_maps(P, 'qwp_err', e, 'qwp_az', a, ...) builds the
%   record test arm (zwfs_params P + macos.design.twyman_green) three times
%   -- the record deck, and the two channel decks of zwfs_vlayout (quarter-
%   wave plate, cemented MacNeille cube, camera A on the transmitted port,
%   camera B on the reflected port) -- traces each in polarization mode
%   (macos.jones_pupil, double-pole basis with the fold plane's normal as
%   the s reference), and forms the ANALYZER Jones per ray
%       J_an = J_camera * inv(J_record)
%   so the arm's own polarization (V3, dmg_arm_maps) is divided out.  The
%   two circular states at the record detector (the metasurface's two
%   output states, at the pupil image) are pushed through J_an: for each
%   camera the MAIN state's power P_main, the OTHER state's power P_other,
%   and the coherent inner product coh = u_other' * u_main, which is what
%   the camera intensity mixes the two masked images with:
%       I = P_main*( |a_main|^2 + l*|a_other|^2 + 2 Re(a_main conj(a_other) c) )
%   with l = P_other/P_main (the INCOHERENT leak: the cube's finite
%   extinction, orthogonal polarization at the camera) and c = coh/P_main
%   (the COHERENT leak: a plate error -- retardance delta gives |c| = delta/2,
%   azimuth theta gives |c| = theta -- and the plate axis projected onto
%   the cone's rays).  Camera A carries the +phi image (Ip), camera B the
%   -phi image (Im).  The maps are pupil means (scalars) for the gauge --
%   camera A's leak varies 3% across the pupil, camera B's incoherent term
%   varies with the cone's angle of incidence at the 1e-3 level (stated in
%   ana.stats) -- and the per-ray maps are kept in ana.maps.
%
%   Options: 'qwp_err' (retardance error, WAVES; default 0), 'qwp_az'
%   (fast-axis azimuth error, degrees; default 0), 'NGRID' (65), 'MODEL'
%   (512), 'cube_side' (12.7), 'qwp_gap' (2), 'cube_gap' (2), 'deck_dir'
%   (where the three decks are written; default tempdir).
%
%   Returns ana with fields lA, cA, lB, cB (the gauge's V_ANALYZER
%   struct: pass ana itself), stats.A / stats.B (main, Pmain, leak
%   mean/rms/max, coh mean/rms, phase of c), split (A/B main power),
%   maps (per-ray Pmain/Pother/coh per camera on the ray grid + mask),
%   qwp_err, qwp_az, decks, cone_deg.
%
%   See also dmg_arm_maps, zwfs_vlayout, macos.jones_pupil, macos.design.pbs_macneille.
o = struct('qwp_err', 0, 'qwp_az', 0, 'NGRID', 65, 'MODEL', 512, ...
           'cube_side', 12.7, 'qwp_gap', 2, 'cube_gap', 2, 'deck_dir', tempdir);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
gridn = 256;  griddx = P.grid.DX_G * P.grid.N_G / gridn;
if ~isfile(P.grid.flat_file), macos.write_grid_file(P.grid.flat_file, zeros(P.grid.N_G)); end
bn = fieldnames(P.bench);  bp = rmfield(P.bench, bn(strncmp(bn, 'coat_', 5)));
bf = fieldnames(bp);  bargs = cell(1, 2*numel(bf));
for i = 1:numel(bf), bargs{2*i-1} = bf{i};  bargs{2*i} = bp.(bf{i}); end
names = {'record', 'A', 'B'};  modes = {'', 'transmit', 'reflect'};
decks = cell(1, 3);  idet = zeros(1, 3);
for c = 1:3
    G = macos.design.twyman_green(bargs{:}, 'ngridpts', o.NGRID, 'to_grid_file', P.grid.flat_file, ...
        'to_grid_n', gridn, 'to_grid_dx', griddx);
    bt = G.bt;  bt.wavelen = P.LAM;  iDET = G.T.iDET;  iFL = iDET - 1;
    if c == 1
        decks{c} = fullfile(o.deck_dir, 'dmg_analyzer_record.in');  bt.emit(decks{c});  idet(c) = iDET;
        continue
    end
    det_leg = G.det_leg;
    bt.E(iDET) = [];
    bt.pos = bt.E(iFL).vpt;  bt.dir = bt.E(iFL).psi;  bt.path_len = bt.E(iFL).s;
    zhat = [0; 0; 1];  phat = cross(bt.dir, zhat);  phat = phat/norm(phat);   % the cube's s and p axes
    az = deg2rad(45 + o.qwp_az);
    bt.add_waveplate(o.qwp_gap, cos(az)*zhat + sin(az)*phat, 0.25 + o.qwp_err, 'name', 'QWP');
    PBS = macos.design.pbs_macneille();
    tok = bt.pbs_cube(o.cube_gap + o.cube_side/2, phat, 'side', o.cube_side, 'n', PBS.n_glass, ...
        'coat', PBS.layers, 'name', 'PBS');
    bt.add_pbs_pass(tok, 'mode', modes{c}, 'tag', names{c});
    d_rest = det_leg - (o.qwp_gap + o.cube_gap + o.cube_side) + o.cube_side*(1 - 1/PBS.n_glass);
    idet(c) = bt.add_detector(d_rest, sprintf('Camera%s', names{c}));
    decks{c} = fullfile(o.deck_dir, sprintf('dmg_analyzer_cam%s.in', names{c}));  bt.emit(decks{c});
end
J = cell(1, 3);  msk = [];  ax = zeros(3, 3);  cone_deg = 0;
for c = 1:3
    macos.init(o.MODEL);  macos.load_rx(decks{c});
    jp = macos.jones_pupil(idet(c), 'basis', 'double-pole', 'xref', [0; 0; 1]);
    J{c} = jp.J;  ax(:, c) = jp.axis;
    if isempty(msk), msk = jp.mask; else, msk = msk & jp.mask; end
    if c == 1
        ca = jp.kx*jp.axis(1) + jp.ky*jp.axis(2) + jp.kz*jp.axis(3);
        cone_deg = rad2deg(acos(min(1, min(ca(jp.mask)))));
    end
end
eR = [1; -1i]/sqrt(2);  eL = [1; 1i]/sqrt(2);               % the two circular states in the record (s, p) pair
ana = struct('lA', 0, 'cA', 0, 'lB', 0, 'cB', 0, 'qwp_err', o.qwp_err, 'qwp_az', o.qwp_az, ...
             'decks', {decks}, 'cone_deg', cone_deg, 'axis', ax, 'maps', struct(), 'stats', struct());
N = size(J{1}, 1);
for c = 2:3
    PR = nan(N);  PL = nan(N);  CO = nan(N);
    for i = 1:N
        for k = 1:N
            if ~msk(i, k), continue; end
            Ja = squeeze(J{c}(i, k, :, :)) / squeeze(J{1}(i, k, :, :));   % analyzer Jones: record (s,p) in, camera out
            uR = Ja*eR;  uL = Ja*eL;
            PR(i, k) = real(uR'*uR);  PL(i, k) = real(uL'*uL);  CO(i, k) = uL'*uR;   % coh = u_L' u_R
        end
    end
    if mean(PR(msk)) >= mean(PL(msk)), main = 'R';  Pm = PR;  Po = PL;  cc = CO;
    else,                               main = 'L';  Pm = PL;  Po = PR;  cc = conj(CO);   % coh = u_other' u_main
    end
    l = Po(msk)./Pm(msk);  cmap = cc./Pm;  cv = cmap(msk);
    st = struct('main', main, 'Pmain', mean(Pm(msk)), 'Pother', mean(Po(msk)), ...
        'leak_mean', mean(l), 'leak_rms', std(l), 'leak_max', max(l), ...
        'coh_mean', abs(mean(cv)), 'coh_abs_mean', mean(abs(cv)), 'coh_rms', std(abs(cv)), 'coh_phase', angle(mean(cv)));
    ana.stats.(names{c}) = st;
    ana.maps.(names{c}) = struct('Pmain', Pm, 'Pother', Po, 'c', cmap, 'mask', msk);
    ana.(['l' names{c}]) = mean(l);  ana.(['c' names{c}]) = mean(cv);
end
ana.split = ana.stats.A.Pmain / ana.stats.B.Pmain;
end
