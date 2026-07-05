% tma_centered_fold_search.m  (mmacos/design/examples/tma_centered/)
% =====================================================================
%  CONSTRAINT FINDER: extract the centered TMA's focal plane from the
%  beam -- field bias + M3 pushback + a flat fold behind the primary.
% =====================================================================
%  THE PROBLEM (see tma_centered.m [3]): the coaxial Korsch's focal
%  plane lands ON AXIS in the middle of everything -- at j18's own
%  geometry the FP sits at z=-4.07 m, inside the incoming beam, inside
%  the M1->M2 cone, and dead on the M2->M3 science beam (which it
%  swallows whole).  Unbuildable as derived.
%
%  THREE KNOBS (Dave, 2026-07-05):
%   1. FIELD BIAS -- use the field a little off-axis: the image walks
%      EFL*theta off the axis, so the FP body can move out of the
%      science beam ("would a slight shift off-axis move it out of the
%      beam?" -- stage [2] answers with the minimum bias).
%   2. A FLAT FOLD (add_fold) in the M2->M3 FEED at a station BEHIND
%      the primary, turning the beam 90 deg into +x with its normal in
%      the X-Z plane: M3, the image, and the FP all land on a FLAT X-Y
%      BENCH behind the PM (Dave's packaging).  The catch: the M3 ->
%      image RETURN re-crosses the fold station and only separates from
%      the feed (by the bias) away from the mirror --
%      fold_station_report finds where a fold of a given mount margin
%      fits.
%   3. M3 FURTHER BEHIND M1 -- j18's own spacings put the exit pupil
%      ~1 m IN FRONT of the primary, where no fold can live.  Pushing
%      M3 back drags the exit pupil behind the PM and opens the gap.
%
%  The finder scans (M3 pushback) x (field bias), verifies each folded
%  candidate with the FULL 3-D clearance judge (check_clipping, with
%  set_hole declaring the perforated primary and an honestly-sized FP
%  body), and keeps the first (most compact) geometry that is clear.
%  Saved to tma_centered_fold_geometry.mat for tma_centered_foldfp.m.
%
%  Run:  >> run('.../tma_centered/tma_centered_fold_search.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
D          = 6.605;                          % aperture (m) -- j18mono
R          = [15.879722 1.778913 3.016227];  % |radii| M1/M2(convex)/M3
TBET       = [7.169041556 7.965313479];      % spacings (m)
LAM        = 1.0e-6;
FIELD_RAD  = 2.5;               % science-field RADIUS about the biased
                                % center (arcmin; = 5' diameter ring)
BIAS_STEPS = [5 10 15 20];      % field-bias ladder (arcmin)
DM3_STEPS  = [0 1 2 3 4];       % M3 pushback ladder (m, added to t2)
FM_MARGIN  = 0.10;              % fold mount margin beyond its bundle (m)
FP_HOUSING = 0.15;              % detector housing beyond the image (m)
ARM_MIN    = 0.40;              % min fold->image arm (m): a fold right AT
                                % the image (where the return bundle
                                % pinches and the gap peaks) leaves no room
                                % for the detector housing -- body-body
                                % interference check_clipping (bodies vs
                                % BEAMS) cannot see
M1_KEEPOUT = 0.60;              % no fold body within this of the M1 vertex
                                % plane: the primary's support structure
                                % (backplane/whiffle trees) owns that
                                % region (Dave 2026-07-05)
% =====================================================================

fprintf('====================================================================\n');
fprintf(' Centered-TMA FP extraction finder | D=%.2f m | %g''-dia science field\n', ...
        D, 2*FIELD_RAD);
fprintf('====================================================================\n');

%% -- [1] the baseline verdict (why this finder exists) ----------------
t0 = build_j18(D, R, TBET, LAM, []);
t0.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1],'max_iters',120);
K0   = [t0.spec.elt(1).Kc t0.spec.elt(2).Kc t0.spec.elt(3).Kc];
zFP  = t0.spec.elt(end).Vpt(3);
EFL  = t0.spec.derived.EFL;
fpr  = EFL*deg2rad(FIELD_RAD/60) + FP_HOUSING;   % honest FP body radius
fprintf(['\n[1] baseline: FP on axis at z=%.2f m -- inside the incoming beam,\n' ...
         '    the M1->M2 cone, and ON the M2->M3 science beam (it swallows\n' ...
         '    it whole).  EFL=%.1f m; honest FP body radius %.2f m.\n'], zFP, EFL, fpr);

%% -- [2] "would a slight shift off-axis move it out of the beam?" -----
% Bias the field, put the FP body at the biased image, and ask whether
% the SCIENCE beam (M2->M3) now clears it.  (The FP still shadows the
% incoming and M1->M2 cones -- an extra off-center obscuration bite the
% fold in [3] removes entirely.)
fprintf('\n[2] bias-only (no fold): min bias that lifts the FP off the science beam\n');
fprintf('    %6s | %9s %9s | %s\n','bias''','FP ctr y','sci edge','verdict');
bias_clear = NaN;
for b = BIAS_STEPS
    t = build_j18(D, R, TBET, LAM, K0, fpr);
    t.set_field_bias(b);
    t.build();
    t.center_focal_plane();                      % FP body -> biased image
    yFP = t.spec.elt(end).Vpt(2);
    % feed-beam interval just above the FP plane (the M3->FP leg ends AT
    % the plane, so probe a hair upstream for both-leg crossings)
    rep = fold_station_report(t,'mirror','M3','z',zFP+0.01,'quiet',true,'noload',true);
    if isempty(rep), continue; end
    % the M3->FP leg IS the science beam converging onto the FP; the FP
    % body must clear the M2->M3 FEED that also crosses this plane
    edge = rep.c_in + sign(yFP - rep.c_in)*rep.hw_in;
    gap  = abs(yFP - edge) - fpr;
    ok   = gap > 0;
    fprintf('    %6.1f | %9.3f %9.3f | %s (gap %+.2f m)\n', b, yFP, edge, ...
            ternary(ok,'CLEARS','blocked'), gap);
    if ok && isnan(bias_clear), bias_clear = b; end
end
if isnan(bias_clear)
    fprintf('    -> no ladder bias clears the feed; the fold is the real fix.\n');
else
    fprintf(['    -> %g'' bias lifts the FP body off the science beam (the\n' ...
             '       incoming/M1->M2 shadow bites remain -- see [3]).\n'], bias_clear);
end

%% -- [3] the fold finder: (M3 pushback) x (bias) ----------------------
% Fold the M2->M3 FEED leg at a station behind the PM, turning it 90 deg
% into +x (Dave 2026-07-05: the better packaging).  The fold normal then
% lies in the X-Z plane, so y maps to y and EVERYTHING downstream -- M3,
% the image, the FP, any instrument -- lands on a FLAT X-Y bench at the
% fold's z-height behind the primary (instead of M3 on a 2.8 m axial
% stalk with only the focus arm folded out).  The clearance physics is
% unchanged by the isometry: the fold body must cover the FEED bundle
% (+ mount margin) while the RETURN (M3 -> image) clears it -- the same
% fold_station_report gap, read on the UNFOLDED biased design.  Verify
% every candidate with the full 3-D judge.  M2's own central obscuration
% is the accepted price of the centered family, excluded from the verdict.
fprintf('\n[3] fold finder: fold the M2->M3 feed into the X-Y bench behind the PM\n');
fprintf('    %5s %6s | %8s %8s %8s | %6s %9s | %s\n', ...
        'dM3','bias''','z*','gap','fm ap_r','clear','shroud/D','verdict');
% BIAS is the outer loop: bias costs image quality across the science
% ring (field curvature/astig variation grows with bias radius) while
% M3 pushback costs only train length -- so take the SMALLEST bias that
% works, then the most compact dM3.  The per-dM3 conic solve is
% bias-independent and cached.
chosen = [];
Kcache = containers.Map('KeyType','double','ValueType','any');
for b = BIAS_STEPS
    for dm3 = DM3_STEPS
        tb = [TBET(1), TBET(2)+dm3];
        if Kcache.isKey(dm3)
            c3 = Kcache(dm3);  K = c3.K;  zM3 = c3.zM3;
        else
            tK = build_j18(D, R, tb, LAM, []);
            tK.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1], ...
                        'max_iters',120);
            K   = [tK.spec.elt(1).Kc tK.spec.elt(2).Kc tK.spec.elt(3).Kc];
            zM3 = tK.spec.elt(3).Vpt(3);
            Kcache(dm3) = struct('K',K,'zM3',zM3);
        end
        t = build_j18(D, R, tb, LAM, K);
        t.set_field_bias(b);
        t.build();
        rep = fold_station_report(t,'mirror','M3', ...
                'z',linspace(0.05, zM3-0.05, 24),'quiet',true,'noload',true);
        if isempty(rep), continue; end
        % station selection (Dave's placement rules, 2026-07-05): among
        % stations with (a) daylight > the fold mount margin, (b) outside
        % the M1 support-structure keep-out, and (c) a real arm between
        % the fold and the image (detector housing -- body-body clearance
        % check_clipping's body-vs-BEAM test cannot see), take the one
        % NEAREST the keep-out: the shortest backbone, the whole bench as
        % close behind the primary as its support structure allows.
        z_img = t.spec.elt(end).Vpt(3);          % unfolded image station
        zz    = [rep.z];  gaps = [rep.gap];
        qual  = (gaps > FM_MARGIN) & (zz >= M1_KEEPOUT) ...
                & (abs(zz - z_img) >= ARM_MIN);
        if any(qual)
            i = find(qual, 1, 'first');          % stations ascend z: first =
        else                                     % shortest backbone
            legal = (zz >= M1_KEEPOUT) & (abs(zz - z_img) >= ARM_MIN);
            gl = gaps;  gl(~legal) = -inf;
            [~, i] = max(gl);                    % best legal gap, for the row
        end
        g = rep(i).gap;  zstar = rep(i).z;
        if ~any(qual)
            fprintf('    %5.1f %6.1f | %8.3f %8.3f %8s | %6s %9s | over\n', ...
                    dm3, b, zstar, g, '--', '--', '--');
            continue;
        end
        % full folded candidate: the fold sits IN THE FEED (after M2, at
        % t1 + z* along the beam) and covers the feed bundle; psi lands in
        % the X-Z plane automatically (feed +z, 'to' +x), so the bench is
        % flat in X-Y; the perforated primary passes the feed; FP sized
        % to the field.
        fdist = tb(1) + zstar;                   % M2 -> station, along beam
        fmr   = rep(i).hw_in + FM_MARGIN;        % body covers the FEED
        tf = build_j18(D, R, tb, LAM, K, fpr);
        tf.set_field_bias(b);
        tf.add_fold('FM','after','M2','dist_m',fdist,'to',[1 0 0],'ap_r',fmr);
        tf.set_hole('M1', abs(rep(1).c_in) + rep(1).hw_in + 0.05*D);
        tf.build();
        tf.center_focal_plane();
        cc = tf.check_clipping('noload',true,'quiet',true);
        iM2  = find(strcmp({cc.name},'M2'),1);
        ok   = all([cc([1:iM2-1, iM2+1:end]).obstructs] == 0) ...
               && all([cc.margin] >= 0);
        pk = packaging_report(tf,'quiet',true);
        fprintf('    %5.1f %6.1f | %8.3f %8.3f %8.3f | %6s %9.2f | %s\n', ...
                dm3, b, zstar, g, fmr, ...
                ternary(ok,'yes','NO'), pk.shroud_over_D, ...
                ternary(ok,'MEETS','over'));
        if ok && isempty(chosen)
            chosen = struct('dm3',dm3, 'bias',b, 'zfold',zstar, ...
                'fold_dist',fdist, 'fold_ap_r',fmr, ...
                'hole_r',abs(rep(1).c_in) + rep(1).hw_in + 0.05*D, ...
                'fp_ap_r',fpr, 'K',K, 'R',R, 'TBET',tb, 'D',D, ...
                'lambda',LAM, 'field_rad',FIELD_RAD, ...
                'shroud_over_D',pk.shroud_over_D, 'bias_only_clear',bias_clear);
            break;                       % lowest bias, then most compact dM3
        end
    end
    if ~isempty(chosen), break; end
end

if isempty(chosen)
    fprintf('\nNO (dM3, bias) step met the clearance -- extend the ladders.\n');
else
    gfile = fullfile(exdir,'tma_centered_fold_geometry.mat');
    save(gfile,'chosen');
    fprintf(['\nCHOSEN: M3 back %.1f m, bias %.1f'' -- feed folded into +x at\n' ...
             '  z=%.2f m behind the PM (body r=%.2f m): M3, image, and FP on\n' ...
             '  the flat X-Y bench behind the primary.\n' ...
             '  Geometry saved: %s\n' ...
             '  Next: run tma_centered_foldfp.m\n'], ...
            chosen.dm3, chosen.bias, chosen.zfold, chosen.fold_ap_r, gfile);
end

% ---------------------------------------------------------------------
function t = build_j18(D, R, tb, lam, K, fp_ap_r)
% j18-family centered TMA; K = [] -> Seidel/derive, else explicit seeds.
% fp_ap_r (optional) sizes the FP BODY for the clearance judge.
    if nargin < 6, fp_ap_r = NaN; end
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
            'model_size',256,'wavelength_m',lam,'grid_npts',41);
    args = {{}, {'convex',true}, {}};
    for k = 1:3
        a = args{k};
        if ~isempty(K), a = [a, {'conic', K(k)}]; end %#ok<AGROW>
        if k < 3
            t.add_mirror(sprintf('M%d',k),'radius_m',R(k), ...
                         'spacing_after_m',tb(k), a{:});
        else
            t.add_mirror('M3','radius_m',R(3),'spacing_after','derive', a{:});
        end
    end
    t.add_focal_plane('FP','ap_r',fp_ap_r);
end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
