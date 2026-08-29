% demo_tg_psi_v2.m
% =========================================================================
%  LIVE DEMO -- polarization PSI Twyman-Green with a REAL polarizing cube
% =========================================================================
%  The v2 cut of ../tg_psi_dm/demo_tg_psi.m.  Same seven beats, same
%  seconds-fast pacing, one component swapped: the perfect-conductor plate
%  and the ideal per-arm polarizer are replaced by a CEMENTED MacNeille
%  CUBE -- one coated interface at 45 degrees between two prisms of the same
%  glass, modelled as ordinary coated engine surfaces.
%
%    1  BUILD      one call, and the coating stack it installs
%    2  LAYOUT     look at it -- and where the light actually goes
%    3  COATING    the engine's 45-deg stack vs the textbook, and the ONE
%                  design detail that decides whether it polarizes at all
%    4  NULL       flat DM: the arms leave orthogonal with NOTHING to align
%    5  POKE       drive ONE actuator; the fringes bend over it
%    6  SWEEP      rotate the analyzer -- the fringes walk, with NO re-trace
%    7  RECOVER    four-step PSI, the DM beside the truth, v1 beside v2
%
%  v1's beat 3 was "the beamsplitter has rotated an arm -- find it, fix it".
%  v2 has nothing to find, so its beat 3 is the design story instead: why
%  the arm rotation is structurally absent, and what would put it back.
%
%  Every beat prints its numbers and writes demo2_beat<N>_*.png here, so a
%  live hang costs ten seconds -- show the PNG and move on.
%
%  The gated version is example_tg_psi_dm_v2.m; the gates are
%  mmacos/tests/tTgPol2.m.
%
%  Run:  cd <this dir>;  demo_tg_psi_v2         % interactive, pauses per beat
%        matlab -batch "run('demo_tg_psi_v2.m'); exit(0)"   % regenerate PNGs
% =========================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(fullfile(exdir, '..', 'tg_psi_dm'));   % dm_influence_map (v1, read-only)
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

LAM   = 6.328e-4;                  % mm (HeNe)
MODEL = 256;   NGRID = 63;
N_G   = 256;   DX_G  = 0.35;       % DM grid: 89 mm span
NACT  = 16;    PITCH = 3.5;        % 16x16 actuators at 3.5 mm
QWP   = 0.25;
THETAS = [0 45 90 135];
TAIL = {'tail_arch','fieldlens', 'FL_F',25.02100857, 'FL_Kc',-2.11278288, ...
        'D_MASK_FL',6.277463741, 'DET_TRIM',1.085330067};
%  Pause between beats only when someone is there to press Enter.  The
%  desktop test is the normal one; TG_DEMO_PAUSE forces it on so the demo can
%  be driven a beat at a time from OUTSIDE MATLAB -- a persistent session fed
%  one newline per beat, which is how the dialog cut runs (each "run it" from
%  the floor is one Enter).  Unset, this is exactly the previous behaviour.
INTERACTIVE = usejava('desktop') || ~isempty(getenv('TG_DEMO_PAUSE'));
%  The live animation needs figures, not a human, so it is gated separately --
%  set TG_DEMO_ANIMATE=1 to exercise that path in batch (it is the one part of
%  the demo a headless run would otherwise never execute, and the demo is the
%  thing that has to work in front of a room).
%  The in-MATLAB animation needs a MATLAB window.  Under an external driver
%  (TG_DEMO_MARKER) there isn't one -- the GIF above is what gets shown -- so
%  skip it rather than burn 4 s animating into nothing.
ANIMATE = (INTERACTIVE || ~isempty(getenv('TG_DEMO_ANIMATE'))) ...
          && isempty(getenv('TG_DEMO_MARKER'));
NTURNS  = 3;  if ~INTERACTIVE, NTURNS = 1; end
macos.init(MODEL);

% =========================================================================
beat(1, 'BUILD -- the whole bench, and the coating it installs', INTERACTIVE);
% =========================================================================
%  'pbs','cube' is the only change from v1.  It puts a cemented MacNeille
%  cube where the plate was: entrance face -> coated 45-deg diagonal ->
%  exit face, with the SAME glass either side of the coating, so the
%  transmitted chief is not displaced at all and the four ports are
%  symmetric.  That last point is why there is no compensator in this rig:
%  every traversal is a/2 -> diagonal -> a/2 whichever port you enter by.
PBS  = macos.design.pbs_macneille();               % (1/2 H  L  1/2 H)^4
PBSq = macos.design.pbs_macneille('design','qw');  % the trap, for beat 3
fprintf('  MacNeille condition n_g*sin45 = nH*nL/sqrt(nH^2+nL^2)\n');
fprintf('    ZnS 2.35 / cryolite 1.35  ->  prism index %.6f (a dense flint)\n', PBS.n_glass);
fprintf('    internal angles %.3f / %.3f deg -- Brewster at every H/L interface\n', ...
    PBS.theta_H, PBS.theta_L);
fprintf('    %d layers, quarter waves AT ANGLE: ', size(PBS.layers,1));
fprintf('%.1f ', PBS.thk*1e6); fprintf('nm\n');
fprintf('  faces get a single-layer MgF2 quarter wave (%.1f nm)\n\n', LAM*1e6/(4*1.38));

mkrig = @(gf) macos.design.twyman_green( ...
    'pbs',          'cube', ...  % <-- the whole v2 change
    'polarizing',   true,  ...   % the cube IS the split; this is required
    'ngridpts',     NGRID, ...
    'qwp_ret',      QWP,   ...
    'to_grid_file', gf, 'to_grid_n', N_G, 'to_grid_dx', DX_G, TAIL{:});

macos.write_grid_file('demo2_flat.txt', zeros(N_G));
G = mkrig('demo2_flat.txt');
G.bt.emit('demo2_test.in');  G.br.emit('demo2_ref.in');
print_train('TEST ARM  (transmits out, reflects back)', G.bt);
print_train('REF  ARM  (reflects out, transmits back)', G.br);
fprintf('\n  -> demo2_test.in (%d elements), demo2_ref.in (%d elements)\n', ...
    numel(G.bt.E), numel(G.br.E));
fprintf('  arm glass paths differ by %.2e mm -- NO compensator plate\n', ...
    abs(G.bt.path_len - G.br.path_len));
fprintf('  every arm azimuth is a DESIGN CONSTANT: pol %g, arm QWPs %g/%g, out QWP %g\n', ...
    G.P.pol_in_deg, G.P.qwp_test_deg, G.P.qwp_ref_deg, G.P.out_qwp_deg);

%  DRAW IT.  Every beat has to put something on the screen -- a beat that
%  only prints is a dead beat in front of a room.  Beat 1's two pictures are
%  the two things its text claims: the stack that was just designed, and how
%  the two arms actually get through the cube.  Neither costs a ray trace.
f1 = figure('Color','w', 'Position',[80 80 1250 470], 'Visible','off');
tl1 = tiledlayout(f1, 1, 2, 'TileSpacing','compact', 'Padding','compact');
nexttile;                                   % index vs depth through the joint
zt = [0; cumsum(PBS.thk(:))*1e6];           % layer boundaries, nm
nn = PBS.layers(:,1);
pad = 0.25*zt(end);
stairs([-pad; zt], [PBS.n_glass; nn; PBS.n_glass], 'LineWidth',2); hold on;
plot([zt(end) zt(end)+pad], PBS.n_glass*[1 1], 'LineWidth',2, 'Color',[0 0.447 0.741]);
yline(PBS.n_glass, ':', sprintf('prism n = %.4f', PBS.n_glass));
xlim([-pad, zt(end)+pad]); grid on;
xlabel('depth through the cemented joint (nm)'); ylabel('refractive index');
title(sprintf('%d layers, ZnS / cryolite, quarter waves AT ANGLE', numel(nn)));
nexttile;                                   % the four ports
hold on; axis equal off; xlim([-0.55 0.75]); ylim([-0.55 0.75]);
rectangle('Position',[0 0 1 1]*0+[-0.35 -0.35 0.7 0.7], 'FaceColor',[.93 .95 .99], ...
          'EdgeColor',[.4 .4 .4], 'LineWidth',1.2);
plot([-0.35 0.35], [0.35 -0.35], 'k-', 'LineWidth',2.5);
text(0.20, 0.22, 'coating', 'FontSize',9, 'Rotation',-45, 'HorizontalAlignment','center');
q = @(x1,y1,x2,y2,c) quiver(x1,y1,x2-x1,y2-y1, 0, 'Color',c, 'LineWidth',2, 'MaxHeadSize',0.5);
q(-0.52,0, -0.05,0, [.2 .2 .2]);           text(-0.52,0.06,'IN  45\circ','FontSize',9);
q(0.05,0, 0.52,0, [0.85 0.33 0.10]);        text(0.30,0.06,'TEST (DM)','FontSize',9,'Color',[0.85 0.33 0.10]);
q(0,-0.05, 0,-0.52, [0 0.447 0.741]);       text(0.03,-0.42,'REF','FontSize',9,'Color',[0 0.447 0.741]);
q(0,0.05, 0,0.52, [0.47 0.67 0.19]);        text(0.03,0.45,'OUT','FontSize',9,'Color',[0.47 0.67 0.19]);
title({'one cemented interface, four ports', ...
       'test TRANSMITS out and REFLECTS back; reference does the reverse'}, ...
      'FontSize',9);
title(tl1, 'The polarizing cube, before a single ray is traced');
print(f1, 'demo2_beat1_stack.png', '-dpng', '-r150');
fprintf('  saved demo2_beat1_stack.png\n');

AT = arm_desc('demo2_test.in', G.bt, G.T, G.P.qwp_test_deg);
AR = arm_desc('demo2_ref.in',  G.br, G.R, G.P.qwp_ref_deg);
AT = set_pol_align(AT, G.P.qwp_test_deg, G.P.out_qwp_deg);
AR = set_pol_align(AR, G.P.qwp_ref_deg,  G.P.out_qwp_deg);

% =========================================================================
beat(2, 'LAYOUT -- look at it, and follow the light', INTERACTIVE);
% =========================================================================
macos.load_rx('demo2_test.in');
%  TWO NAMED VIEWS.  view_std's four defaults are built for a folded
%  telescope; this bench lives in the optical-table PLANE, so the panel that
%  actually shows the routing is the one looking straight DOWN on it.  Both
%  cameras are set explicitly (the recipe is view_std's own) so each panel
%  can be labelled for what it is -- a "side view" caption over a top-down
%  picture is the kind of small lie that costs you the room.
%  The up-vector must not be parallel to the view direction: for the
%  top-down panel that rules out the beam-frame vertical, so use the
%  BROADSIDE axis, which puts the light left-to-right the way an optical
%  layout is always drawn.  (Using the beam direction as up works too and
%  runs the light up the page -- correct, but a tall thin picture in a wide
%  panel.)
d0 = G.bt.src_dir(:);  [~, i0] = min(abs(d0));
xb = zeros(3,1);  xb(i0) = 1;  xb = xb - dot(xb,d0)*d0;  xb = xb/norm(xb);
yb = cross(d0, xb);
ai = deg2rad([-35 22]);
VW = { -yb, xb, 'TABLE PLANE -- looking down on the bench' ; ...
       cos(ai(2))*(cos(ai(1))*xb + sin(ai(1))*d0) + sin(ai(2))*yb, yb, 'ISO view' };
f = figure('Color','w', 'Position',[40 40 1700 560], 'Visible','off');
tl2 = tiledlayout(f, 1, 2, 'Padding','tight', 'TileSpacing','tight');
for q = 1:size(VW,1)
    ax = nexttile(tl2);
    macos.view_rx('ax', ax, 'title', VW{q,3});
    axis(ax, 'equal');
    xl = xlim(ax);  yl = ylim(ax);  zl = zlim(ax);
    tgt = [mean(xl); mean(yl); mean(zl)];
    dd  = 3*max([diff(xl), diff(yl), diff(zl)]);
    set(ax, 'CameraTarget', tgt.', 'CameraPosition', (tgt - dd*VW{q,1}).', ...
            'CameraUpVector', VW{q,2}.', 'Projection', 'orthographic');
    camva(ax, 'auto');  camzoom(ax, 1.45);  axis(ax, 'off');
end
title(tl2, 'MacNeille-cube polarization-PSI Twyman-Green, test arm');
print(f, 'demo2_beat2_layout.png', '-dpng', '-r150');
fprintf('  source -> baffle -> L1 -> polarizer@45 -> CUBE ->\n');
fprintf('     TEST arm: the cube TRANSMITS p out to the DM; the double-passed\n');
fprintf('               QWP turns p into s; the cube REFLECTS s to the output\n');
fprintf('     REF  arm: the cube REFLECTS s out to the flat; the QWP turns s\n');
fprintf('               into p; the cube TRANSMITS p to the SAME output\n');
fprintf('  -> recomb -> output QWP -> rotating analyzer -> L2 -> focal mask ->\n');
fprintf('     field lens -> detector at the DM pupil image\n');
fprintf('  Both returns leave by the output port -- that is what a PBS buys,\n');
fprintf('  and it is why the cube delivers ~2.3x the light the plate rig does.\n');
fprintf('  saved demo2_beat2_layout.png\n');
%  Point at the PRESCRIPTIONS.  Everything on screen came out of two text
%  files, and an audience that models optics wants to see them.  Give the
%  paths, say how an element is found in one (each is a block starting
%  "iElt= N"), name the handful that matter, and then show the coating
%  block VERBATIM -- that is the whole of what the engine was told about
%  the beamsplitter.
dk = {'demo2_test.in', 'demo2_ref.in'};
fprintf('\n  the two prescriptions the engine actually reads:\n');
for q = 1:2
    L = regexp(fileread(dk{q}), '\n', 'split');
    fprintf('    %s   (%d elements, %d lines)\n', ...
            fullfile(pwd, dk{q}), numel(G.bt.E), numel(L));
end
fprintf('  each element is a block beginning "iElt= N".  The ones to look at:\n');
fprintf('    iElt=  4   PolIn        both arms, 45 deg in\n');
fprintf('    iElt=  6   PBStxf / PBSrff   THE SPLIT: same coating, transmitted / reflected\n');
fprintf('    iElt=  9   TestOptic / PZT   the DM (GridData) / the reference flat\n');
fprintf('    iElt= 12   PBSrfr / PBStxr   the same coating again, on the way back\n');
fprintf('    iElt= 16   Analyzer     the only thing that moves in the whole rig\n');
%  There are SEVERAL Coating= blocks in the deck -- the cube faces carry a
%  single AR layer -- so take the one whose layer count matches the
%  diagonal's, not simply the first.  Print exactly that many rows after it.
Lt = regexp(fileread('demo2_test.in'), '\n', 'split');
nL = size(PBS.layers,1);
kall = find(~cellfun(@isempty, regexp(Lt, '^\s*Coating=', 'once')));
kc = [];
for q = kall
    if str2double(strtrim(extractAfter(strtrim(Lt{q}), 'Coating='))) == nL
        kc = q;  break;
    end
end
if ~isempty(kc)
    fprintf('  and this is the beamsplitter coating, verbatim from demo2_test.in\n');
    fprintf('  (n, extinction, optical thickness in waves; the faces carry a\n');
    fprintf('   separate single-layer AR block, which is why we match on count):\n');
    for q = kc : min(kc + nL, numel(Lt))
        fprintf('    %s\n', strtrim(Lt{q}));
    end
end

% =========================================================================
beat(3, 'COATING -- the engine stack vs the textbook, and the trap', INTERACTIVE);
% =========================================================================
%  Probe the diagonal with a PURE s and a PURE p input and take the amplitude
%  ratio across it.  The reference is macos.design.thinfilm_rt -- Macleod's
%  characteristic matrix, written from the textbook and never transcribed
%  from the engine (an "analytic" copied out of elemsub.F is circular in
%  exactly the coefficient it should check).
E = probe_rt('demo2_test.in', G.T.iPBSf, 'demo2_ref.in', G.R.iPBSf);
A = PBS.rt;
fprintf('               engine        Macleod\n');
fprintf('    R_s   %12.8f   %12.8f\n', E.Rs, A.Rs);
fprintf('    T_p   %12.8f   %12.8f\n', E.Tp, A.Tp);
fprintf('    T_s   %12.3e   %12.3e\n', E.Ts, A.Ts);
fprintf('    R_p   %12.3e   %12.3e   <-- the MacNeille p-null\n', E.Rp, A.Rp);
fprintf('    R+T = %.12f (s) / %.12f (p) -- the two arms'' decks, one coating\n', ...
    E.Rs+E.Ts, E.Rp+E.Tp);
fprintf('    extinction T_p/T_s = %.0f : 1\n\n', E.Tp/E.Ts);
%  And the design lesson.  Brewster at the H/L interfaces equalizes the
%  tilted p admittances, so for p the whole stack is ONE HOMOGENEOUS SLAB --
%  but its two boundaries with the PRISM are not Brewster, and what happens
%  there depends only on the slab's total p phase thickness.
Gq = macos.design.twyman_green('pbs','cube','polarizing',true,'ngridpts',NGRID, ...
        'qwp_ret',QWP, 'pbs_coat', PBSq.layers, ...
        'to_grid_file','demo2_flat.txt','to_grid_n',N_G,'to_grid_dx',DX_G, TAIL{:});
Gq.bt.emit('demo2_qw_test.in');  Gq.br.emit('demo2_qw_ref.in');
Eq = probe_rt('demo2_qw_test.in', Gq.T.iPBSf, 'demo2_qw_ref.in', Gq.R.iPBSf);
fprintf('  SAME Brewster condition, ONE layer''s worth of termination different:\n');
fprintf('    (1/2H L 1/2H)^4  = %2.0f quarter waves (EVEN, half-wave absentee): R_p %.2e\n', ...
    PBS.qw_total, E.Rp);
fprintf('    H(LH)^4          = %2.0f quarter waves (ODD, quarter-wave layer):  R_p %.2e\n', ...
    PBSq.qw_total, Eq.Rp);
fprintf('  Both are textbook MacNeille designs.  One is a polarizer.\n');
fprintf('  (and the odd one costs %.1f%% of the transmitted p light)\n\n', 100*(1-Eq.Tp));
%  So there is nothing to align.  v1 had to solve a +3.768-degree waveplate
%  clock here because its splitter rotated the test arm 7.479 degrees; the
%  cube puts each arm ON a coating eigenaxis, where a diattenuator cannot
%  rotate anything.
az_t = arm_azimuth(AT, QWP, AT.qwp_deg);
az_r = arm_azimuth(AR, QWP, AR.qwp_deg);
fprintf('  arms leave at %+10.6f and %+10.6f deg: %.2e deg from ORTHOGONAL\n', ...
    az_t, az_r, abs(mod(az_t-az_r+90,180)-90)-90);
st = arm_state(AT, QWP, AT.iOQ);  sr = arm_state(AR, QWP, AR.iOQ);
fprintf('  after the output QWP (a DESIGN constant, not a solved one):\n');
fprintf('    test |b/a| %.5f arg %+7.2f | ref |b/a| %.5f arg %+7.2f | <t|r> %.1e\n', ...
    abs(st(2)/st(1)), rad2deg(angle(st(2)/st(1))), ...
    abs(sr(2)/sr(1)), rad2deg(angle(sr(2)/sr(1))), ...
    abs(st'*sr)/(norm(st)*norm(sr)));
fprintf('  v1 needed an alignment beat here.  This rig has no knob to turn.\n');
%  DRAW IT.  The table above is two numbers per stack; the picture is the
%  whole story -- how R_p behaves either side of the design angle, with the
%  engine's own measurements dropped on top of the textbook curves.  Pure
%  analytic sweep, no ray tracing, so it costs nothing.
aoi = linspace(35, 55, 121);
Rp1 = zeros(size(aoi));  Rs1 = Rp1;  Rp2 = Rp1;  Rs2 = Rp1;
L1 = [PBS.layers(:,1)  - 1i*PBS.layers(:,2),  PBS.thk(:)];
L2 = [PBSq.layers(:,1) - 1i*PBSq.layers(:,2), PBSq.thk(:)];
for q = 1:numel(aoi)
    a = macos.design.thinfilm_rt(L1, PBS.n_glass,  PBS.n_glass,  aoi(q), LAM);
    b = macos.design.thinfilm_rt(L2, PBSq.n_glass, PBSq.n_glass, aoi(q), LAM);
    Rp1(q) = a.Rp;  Rs1(q) = a.Rs;  Rp2(q) = b.Rp;  Rs2(q) = b.Rs;
end
f3 = figure('Color','w', 'Position',[80 80 1250 470], 'Visible','off');
tl3 = tiledlayout(f3, 1, 2, 'TileSpacing','compact', 'Padding','compact');
for pn = 1:2
    nexttile;
    if pn == 1, Rp = Rp1; Rs = Rs1; Ee = E;  ttl = '(1/2H L 1/2H)^4  -- EVEN, absentee';
    else,       Rp = Rp2; Rs = Rs2; Ee = Eq; ttl = 'H(LH)^4  -- ODD, quarter-wave layer'; end
    semilogy(aoi, max(Rs,1e-16), 'LineWidth',1.8); hold on;
    semilogy(aoi, max(Rp,1e-16), 'LineWidth',1.8);
    semilogy(45, max(Ee.Rs,1e-16), 'o', 'MarkerSize',10, 'LineWidth',2, 'Color',[0 0.447 0.741]);
    semilogy(45, max(Ee.Rp,1e-16), 's', 'MarkerSize',10, 'LineWidth',2, 'Color',[0.85 0.33 0.10]);
    xline(45, ':', 'design angle');
    grid on; ylim([1e-14 2]); xlabel('angle of incidence in the prism (deg)');
    ylabel('reflectance');  title(ttl, 'FontSize',10);
    legend({'R_s (textbook)','R_p (textbook)','R_s (engine)','R_p (engine)'}, ...
           'Location','east', 'FontSize',8);
end
title(tl3, 'Same MacNeille condition, one layer of termination apart -- lines are Macleod, markers are the engine');
print(f3, 'demo2_beat3_coating.png', '-dpng', '-r150');
fprintf('  saved demo2_beat3_coating.png\n');

% =========================================================================
beat(4, 'NULL -- flat DM, and nothing was aligned', INTERACTIVE);
% =========================================================================
%  THREE traces per arm, and then every analyzer angle is free: the detector
%  field is bilinear in the analyzer axis, so E(t) = c^2 A + c s B + s^2 C.
t0 = tic;
Sr    = analyzer_basis(AR, QWP, []);
Sflat = analyzer_basis(AT, QWP, []);
fprintf('  6 traces in %.2f s -- from here the analyzer costs nothing\n', toc(t0));
fprintf('  (v1 spent 7 more traces before this point, solving the arm clock)\n');
I_null = frame(Sflat, Sr, 0);
msk = I_null > 0.1*max(I_null(:));
p_null = fourstep(Sflat, Sr, THETAS);
fprintf('  fringe field is FLAT: recovered phase %.3e rad rms in the pupil\n', ...
    std(p_null(msk) - median(p_null(msk))));
fprintf('  = %.4f nm of surface -- the null of a compensated Twyman-Green\n', ...
    1e6*std(p_null(msk)-median(p_null(msk)))*LAM/(4*pi));
%  SHOW the null.  A beat that prints a number and draws nothing is a dead
%  beat in front of a room -- and this is the most important number so far,
%  so it earns a picture.  The point of the picture is that there is nothing
%  IN it: a featureless fringe field, and a surface map whose full colour
%  range is a few hundredths of a nanometre.  Both panels are the honest
%  autoscale, NOT a fixed scale that would flatter the result.
h_null = (p_null - median(p_null(msk))) * LAM/(4*pi) * 1e6;    % nm of surface
b4 = beam_box(msk, 6);
f4 = figure('Color','w', 'Position',[80 80 1180 470], 'Visible','off');
tl4 = tiledlayout(f4, 1, 2, 'TileSpacing','compact', 'Padding','compact');
nexttile; imagesc(sub(I_null, b4)); axis image off; colorbar;
title('the fringe field, flat DM');
show(h_null, msk, size(h_null,1), b4, [], ...
     sprintf('recovered surface: %.4f nm rms', std(h_null(msk))));
title(tl4, sprintf(['nothing to measure -- and the instrument agrees ' ...
                    '(%.3f nm rms, %.0f pm)'], std(h_null(msk)), ...
                   1e3*std(h_null(msk))));
print(f4, 'demo2_beat4_null.png', '-dpng', '-r150');
fprintf('  saved demo2_beat4_null.png -- the picture is that there is nothing in it\n');

% =========================================================================
beat(5, 'POKE -- drive one actuator', INTERACTIVE);
% =========================================================================
%  A LIVE poke: rewrite the DM grid in the loaded model (macos.set_elt_grid,
%  which invalidates the cached trace for you) rather than re-emitting a
%  deck.  This is the same call an actuator command loop would make.
M1 = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                      'poke',150e-6, 'pattern','single');
S1 = analyzer_basis(AT, QWP, M1);
I_poke = frame(S1, Sr, 0);
fprintf('  one actuator at 150 nm -> the fringes bend over it\n');
d1 = angle(exp(1i*(fourstep(S1,Sr,THETAS) - p_null)));
fprintf('  local surface change %.1f nm peak (injected %.1f nm)\n', ...
    1e6*max(abs(d1(msk)))*LAM/(4*pi), 150);
fprintf('  (a few percent low: the detector''s pupil image of the DM smooths a\n');
fprintf('   SINGLE actuator, which is the sharpest thing the DM can make)\n');
b5 = beam_box(msk, 6);  cl5 = [0 max([I_null(msk); I_poke(msk)])];
f5 = figure('Color','w','Position',[80 80 1200 460]);
subplot(1,2,1); imagesc(sub(I_null,b5)); axis image off; clim(cl5); colorbar;
title('flat DM: the null (a dark fringe)');
subplot(1,2,2); imagesc(sub(I_poke,b5)); axis image off; clim(cl5); colorbar;
title('one actuator at 150 nm');
print(f5, 'demo2_beat5_poke.png', '-dpng', '-r150');
fprintf('  saved demo2_beat5_poke.png\n');

% =========================================================================
beat(6, 'SWEEP -- rotate the analyzer, no re-trace', INTERACTIVE);
% =========================================================================
%  Nothing in the interferometer moves.  The phase steps come from rotating
%  a polarizer in the output leg -- and because the field is bilinear in the
%  analyzer axis, all of these frames come out of the six traces of beat 4.
nsw = 36;  ths = (0:nsw-1)/nsw*180;
t0 = tic;  Isw = zeros([size(I_null) nsw]);
for k = 1:nsw, Isw(:,:,k) = frame(S1, Sr, ths(k)); end
fprintf('  %d analyzer frames synthesized in %.3f s -- ZERO traces\n', nsw, toc(t0));
pk = find(msk);  [~,ip] = max(abs(d1(pk)));  [pr,pc] = ind2sub(size(I_null), pk(ip));
tr = squeeze(Isw(pr,pc,:));
b6 = beam_box(msk, 6);  cl6 = [0 max(Isw(:))];
f6 = figure('Color','w','Position',[80 80 1500 700]);
for k = 1:4
    subplot(2,4,k); imagesc(sub(frame(S1,Sr,THETAS(k)),b6)); axis image off;
    clim(cl6);          % ONE scale for all four: the whole field modulates
    title(sprintf('analyzer %d\\circ  (step %d/4)', THETAS(k), k));
end
subplot(2,4,[5 6]);
plot(ths, tr/max(tr), 'b-', 'LineWidth',1.5); hold on;
plot(THETAS, interp1([ths 180],[tr;tr(1)]/max(tr),THETAS), 'ro','MarkerFaceColor','r');
grid on; xlabel('analyzer angle \theta (deg)'); ylabel('normalized intensity');
title('one pixel: I(\theta) = A + B cos2\theta + C sin2\theta');
legend({'synthesized','the four steps'}, 'Location','south');
subplot(2,4,[7 8]);
%  the whole sweep as ONE static image: a cut across the beam through the
%  poked actuator, stacked against analyzer angle.  This is what the live
%  animation shows, in a form that survives being a PNG on a slide.
kymo = squeeze(Isw(pr, b6(3):b6(4), :)).';
imagesc(1:size(kymo,2), ths, kymo);  clim(cl6);
xlabel('across the beam (px)'); ylabel('analyzer angle \theta (deg)');
title(sprintf('the fringes walk: one cut through the actuator, all %d angles', nsw));
print(f6, 'demo2_beat6_sweep.png', '-dpng', '-r150');
fprintf('  saved demo2_beat6_sweep.png\n');
%  The sweep as an ANIMATED GIF.  MATLAB's own animation loop below needs a
%  MATLAB window; a GIF plays in any viewer, which is what a driven session
%  (or a slide) actually has.  Same 36 synthesized frames, same fixed scale
%  as the panels above -- if the scale floated, the fringes would look like
%  they were breathing when only the phase is moving.
gname = 'demo2_beat6_sweep.gif';
if exist(gname, 'file'), delete(gname); end
cmg = parula(256);  ghi = max(Isw(:));
for k = 1:nsw
    A = sub(Isw(:,:,k), b6);
    ix = uint8(255*max(0, min(1, A/max(ghi, eps))));
    if k == 1
        imwrite(ix, cmg, gname, 'gif', 'LoopCount',Inf, 'DelayTime',0.06);
    else
        imwrite(ix, cmg, gname, 'gif', 'WriteMode','append', 'DelayTime',0.06);
    end
end
fprintf('  saved %s -- the sweep, playable outside MATLAB\n', gname);
if ANIMATE                          % the live animation
    fprintf('  animating the sweep (close the window to stop)...\n');
    fa = figure('Color','w','Position',[200 200 620 620]);
    ax = axes(fa);  im = imagesc(ax, sub(Isw(:,:,1),b6));  axis(ax,'image','off');
    cl = [min(Isw(:)) max(Isw(:))];  clim(ax, cl);
    for turn = 1:NTURNS
        for k = 1:nsw
            if ~isvalid(fa), break; end
            set(im, 'CData', sub(Isw(:,:,k),b6));
            title(ax, sprintf('analyzer %5.1f\\circ   (fringe phase 2\\theta)', ths(k)));
            drawnow;  if INTERACTIVE, pause(0.04); end
        end
        if ~isvalid(fa), break; end
    end
    if isvalid(fa)
        print(fa, 'demo2_beat6_animation_frame.png', '-dpng', '-r120');
        fprintf('  animation ran %d turns x %d frames\n', NTURNS, nsw);
    end
end

% =========================================================================
beat(7, 'RECOVER -- four-step PSI against the truth', INTERACTIVE);
% =========================================================================
%  The full DM command now, and the differential protocol a real gauge uses:
%  measure flat, measure poked, subtract in the complex domain.  Every static
%  term cancels and nothing needs unwrapping.
[Mdm, dminfo] = dm_influence_map(N_G, DX_G, 'nact',NACT, 'pitch',PITCH, ...
                                 'poke',50e-6, 'pattern','checker');
Sdm  = analyzer_basis(AT, QWP, Mdm);
dphi = angle(exp(1i*(fourstep(Sdm,Sr,THETAS) - p_null)));
h    = dphi * LAM/(4*pi);           % surface height, mm (sign gated in the example)
fprintf('  %dx%d actuators, checkerboard at 50 nm: %.2f nm rms, %.1f nm PtV injected\n', ...
    NACT, NACT, 1e6*std(Mdm(:)), 1e6*(max(Mdm(:))-min(Mdm(:))));
[best, map] = register_to_dm(AT, G.T, Mdm, N_G, DX_G, h, msk);
%  Score the interior: the outermost ring resamples the truth with rays on one
%  side only, so its residual is an artefact of the comparison, not the gauge.
mskin = erode_disc(msk, 0.92);
res  = best.hm(mskin) - best.ht(mskin);
fprintf('  the detector images the DM at mag %.3f with %.3f%% anamorphic stretch\n', ...
    map.mag, map.anam_pct);
fprintf('  and %.4f mm of nonlinear distortion -- measured from the trace, not assumed\n', ...
    map.nonlin_mm);
fprintf('\n  RECOVERED %.2f nm rms   TRUTH %.2f nm rms   RESIDUAL %.3f nm rms (%.1f%%)\n\n', ...
    1e6*std(best.hm(msk)), 1e6*std(best.ht(msk)), 1e6*std(res), ...
    100*std(res)/std(best.ht(mskin)));
NN = size(h,1);  box = beam_box(msk, 6);
cl = [-1 1]*1e6*max(abs(best.ht(msk)));
f7 = figure('Color','w','Position',[80 80 1560 420]);
tl = tiledlayout(f7,1,4,'TileSpacing','compact','Padding','compact');
nexttile; imagesc(sub(frame(Sdm,Sr,0),box)); axis image off; colorbar; title('interferogram');
show(1e6*best.hm, msk, NN, box, cl, sprintf('MEASURED (%.2f nm rms)', 1e6*std(best.hm(msk))));
show(1e6*best.ht, msk, NN, box, cl, sprintf('TRUTH (%.2f nm rms)', 1e6*std(best.ht(msk))));
show(1e6*(best.hm-best.ht), msk, NN, box, [-1 1]*4e6*std(res), ...
     sprintf('residual %.3f nm rms interior', 1e6*std(res)));
title(tl, 'Polarization-PSI Twyman-Green: a DM measured, and the map it was given');
print(f7, 'demo2_beat7_recovery.png', '-dpng', '-r150');
fprintf('  saved demo2_beat7_recovery.png\n');

% =========================================================================
beat(8, 'CALIBRATE -- and find the edge of the calibration', INTERACTIVE);
% =========================================================================
%  Beat 7 left 0.18 nm rms on the table, and it is NOT noise -- it correlates
%  with the truth, so it is a deterministic function of what went in and can
%  in principle be calibrated out.  This beat does that properly, which means
%  three things a single scale factor does not:
%
%   (1) CALIBRATE ON A BASIS, not on one shape.  Inject each of a set of
%       Zernike modes on the DM, one at a time, and build the matrix G that
%       takes true mode coefficients to measured ones.  Its DIAGONAL is the
%       per-mode gain; its OFF-DIAGONAL is cross-talk, which a scalar gain
%       cannot see and a shift-invariant transfer function cannot fix.
%   (2) CHECK IT IS LINEAR IN AMPLITUDE over the range you care about.  A
%       gain measured at one amplitude is worthless if the gauge is not
%       linear; measure it rather than assume it.
%   (3) VALIDATE OUT OF SAMPLE.  Scoring a calibration on the data that
%       fitted it measures nothing.  Two held-out tests are run: a random
%       combination INSIDE the calibrated span, and the beat-7 checkerboard,
%       which is deliberately OUTSIDE it.  The contrast between those two is
%       the answer to "over what range of aberrations does this help".
%
%  The aperture fraction is MEASURED, not taken from the nominal design: the
%  registration of beat 7 already gives every detector pixel's position on the
%  DM (map.Xt/map.Yt), so the illuminated radius is just the largest of those
%  inside the beam mask.  Reading it off the design would bake in exactly the
%  kind of assumption this beat exists to avoid.
MODES = [4 5 6 7 8 9 10 13];
NAMES = {'astig45','defocus','astig0','trefoil-y','coma-y','coma-x','trefoil-x','spherical'};
AMP   = 20e-6;                                   % 20 nm rms per mode
apf   = max(hypot(map.Xt(msk), map.Yt(msk))) / (((N_G-1)/2)*DX_G);
Bz    = macos.zernike_grid_basis(N_G, MODES, apf);
axs   = ((1:N_G)-(N_G+1)/2)*DX_G;
nM    = numel(MODES);
fprintf('  basis: %d Zernike modes at %g nm rms, aperture fraction %.4f (measured)\n', ...
        nM, AMP*1e6, apf);

Tt = zeros(nnz(msk), nM);  Mm = zeros(nnz(msk), nM);
t8 = tic;
for k = 1:nM
    Mk = AMP * Bz(:,:,k);
    hk = meas_surface(AT, QWP, Mk, Sr, p_null, THETAS, LAM);
    tk = interpn(axs, axs, Mk, map.Xt, map.Yt, 'spline', 0);
    Mm(:,k) = hk(msk) - mean(hk(msk));
    Tt(:,k) = tk(msk) - mean(tk(msk));
end
G = Tt \ Mm;                        % measured coeffs = G * true coeffs
fprintf('  %d modes x 3 traces in %.1f s\n', nM, toc(t8));
fprintf('  %-11s %8s %11s\n', 'mode', 'gain', 'cross-talk');
for k = 1:nM
    x = G(:,k);  x(k) = 0;
    fprintf('  %-11s %8.4f %11.4f\n', NAMES{k}, G(k,k), norm(x));
end

%  (2) amplitude linearity, on one mode
ki = find(MODES == 5, 1);            % defocus
amps = [2 10 30 50]*1e-6;  rec = zeros(size(amps));
for q = 1:numel(amps)
    hq = meas_surface(AT, QWP, amps(q)*Bz(:,:,ki), Sr, p_null, THETAS, LAM);
    c  = Tt \ (hq(msk) - mean(hq(msk)));
    rec(q) = c(ki)*AMP;              % Tt columns carry the AMP scaling
end
pf = polyfit(amps, rec, 1);
nl = max(abs(rec - polyval(pf, amps)))/max(rec);
fprintf('  amplitude linearity on defocus, %g to %g nm rms:\n', amps(1)*1e6, amps(end)*1e6);
fprintf('     injected  '); fprintf('%8.1f', amps*1e6);  fprintf('  nm\n');
fprintf('     recovered '); fprintf('%8.2f', rec*1e6);   fprintf('  nm\n');
fprintf('     slope %.5f, departure from a straight line %.2f%% of full scale\n', ...
        pf(1), 100*nl);

%  (3) out-of-sample validation: inside the span, and outside it
rng_w = [0.31 -0.62 0.44 0.18 -0.27 0.51 -0.12 0.35];   % fixed, not random --
w = rng_w(:)/norm(rng_w) * (20e-6/AMP);                 % a demo must repeat
Min = zeros(N_G);  for k = 1:nM, Min = Min + w(k)*AMP*Bz(:,:,k); end
val = {'in-span mixture', Min; 'beat-7 checkerboard', Mdm};
vb = zeros(1,2);  va = zeros(1,2);  rcr = zeros(1,2);
for v = 1:2
    hv = meas_surface(AT, QWP, val{v,2}, Sr, p_null, THETAS, LAM);
    tv = interpn(axs, axs, val{v,2}, map.Xt, map.Yt, 'spline', 0);
    hv = hv(msk) - mean(hv(msk));   tv = tv(msk) - mean(tv(msk));
    cc = G \ (Tt \ hv);             % de-calibrate into true coefficients
    hc = hv - Tt*(Tt\hv) + Tt*cc;    % correct only the part the basis spans
    vb(v) = 1e6*std(hv - tv);   va(v) = 1e6*std(hc - tv);
    %  Report the residual as a FRACTION of the input too.  The two test
    %  shapes are not the same size (20 nm rms vs 6.4), so comparing their
    %  absolute residuals would say more about the inputs than the gauge.
    %  And correlate what is LEFT against the truth: if it still correlates,
    %  there is linear signal the basis did not reach.
    rc = corrcoef(hc - tv, tv);  rcr(v) = rc(1,2);
    fprintf('  %-20s %.4f -> %.4f nm rms   (%.2f%% of a %.2f nm input)  resid/truth corr %+.2f\n', ...
            val{v,1}, vb(v), va(v), 100*va(v)/(1e6*std(tv)), 1e6*std(tv), rc(1,2));
end
%  THE HONEST READING, and it is not the one this beat was built expecting.
gm = mean(diag(G));  gs = std(diag(G));
xt = G - diag(diag(G));
fprintf('  ------------------------------------------------------------------\n');
fprintf('  The gauge''s LINEAR response is essentially ideal over these\n');
fprintf('  aberrations: gain %.4f +- %.4f across all %d modes, cross-talk\n', ...
        gm, gs, nM);
fprintf('  below %.1f%%, and linear in amplitude to %.2f%% from %g to %g nm.\n', ...
        100*max(abs(xt(:))), 100*nl, amps(1)*1e6, amps(end)*1e6);
fprintf('  So there is almost nothing to calibrate -- correcting it buys ~%.0f%%.\n', ...
        100*(1 - va(1)/vb(1)));
%  The two correlations LOCALISE what is left, which is the real deliverable
%  here: they say WHERE more calibration would pay and where it would not.
fprintf('  And the last column says where the leftover error lives.  After\n');
fprintf('  correction the in-span residual is UNCORRELATED with the truth\n');
fprintf('  (%+.2f) -- over low-order aberrations this gauge is AT ITS FLOOR and\n', rcr(1));
fprintf('  no calibration will help it.  The checkerboard''s residual is still\n');
fprintf('  %+.2f correlated, so at ACTUATOR spatial frequencies there IS signal\n', rcr(2));
fprintf('  left on the table -- above what an 8-mode basis can reach.\n');
fprintf('  The answer to "over what range of aberrations": over the low-order\n');
fprintf('  range it is already at the floor; to do better you widen the BASIS,\n');
fprintf('  not the gain.\n');

f8 = figure('Color','w', 'Position',[80 80 1500 430], 'Visible','off');
tl8 = tiledlayout(f8, 1, 3, 'TileSpacing','compact', 'Padding','compact');
nexttile; imagesc(G); axis image; colorbar; clim([-1 1]*max(abs(G(:))));
set(gca,'XTick',1:nM, 'XTickLabel',NAMES, 'YTick',1:nM, 'YTickLabel',NAMES);
xtickangle(45);  title('G: true coefficients -> measured');
nexttile; plot(amps*1e6, rec*1e6, 'o-', 'LineWidth',1.6); grid on; hold on;
plot(amps*1e6, polyval(pf,amps)*1e6, 'k--');
xlabel('injected (nm rms)'); ylabel('recovered (nm rms)');
title(sprintf('linearity: slope %.4f, %.2f%% departure', pf(1), 100*nl));
nexttile; bar([vb; va].'); grid on;
set(gca,'XTickLabel',{'in-span','out-of-span'});
ylabel('residual (nm rms)'); legend({'raw','calibrated'}, 'Location','northwest');
title('held-out validation');
title(tl8, 'Calibrating the gauge -- and the edge of the calibration');
print(f8, 'demo2_beat8_calibration.png', '-dpng', '-r150');
fprintf('  saved demo2_beat8_calibration.png\n');

beat(0, '', false);
fprintf('\n=== demo complete.  Artefacts: demo2_*.in, demo2_beat*.png ===\n');

% =========================================================================
%  LOCAL FUNCTIONS
% =========================================================================
function beat(n, ttl, interactive)
%BEAT  Banner, optional pause, and SELF-TIMING.  The rehearsal question for a
%   live demo is "does any beat lose the room", so measure it rather than
%   estimate: the timer stops on the NEXT banner, and the total prints at the
%   end (call beat(0,...) to close the last one).
%
%   If TG_DEMO_MARKER names a directory, each finished beat also drops an
%   empty beat<N>.done file there.  A driver outside MATLAB needs a
%   completion signal it can trust, and stdout to a redirected file is
%   block-buffered -- the filesystem is not.  The marker is written BEFORE
%   the next beat's input() blocks, so it means "beat N is finished and the
%   script is waiting", which is exactly what the driver has to know.
    persistent t0 last
    if ~isempty(t0) && ~isempty(last)
        fprintf('\n   [beat %d took %.2f s]\n', last, toc(t0));
        md = getenv('TG_DEMO_MARKER');
        if ~isempty(md)
            fid = fopen(fullfile(md, sprintf('beat%d.done', last)), 'w');
            if fid > 0, fclose(fid); end
        end
    end
    if n == 0, t0 = [];  last = [];  return; end
    fprintf('\n');
    fprintf('#########################################################\n');
    fprintf('#  BEAT %d -- %s\n', n, ttl);
    fprintf('#########################################################\n');
    if interactive && n > 1
        input('   [enter to run this beat] ', 's');
    end
    t0 = tic;  last = n;
end

function print_train(ttl, b)
    fprintf('\n  %s\n', ttl);
    for k = 1:numel(b.E)
        e = b.E(k);
        mark = '';
        if ismember(e.element, {'TrPolarizer','WavePlate'}), mark = '   <-- polarization'; end
        if ~isempty(e.gridfile), mark = '   <-- the DM'; end
        fprintf('   %2d  %-11s %-9s %-10s s=%8.3f mm%s\n', ...
            k, e.name, e.element, e.surface, e.s, mark);
    end
end

function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
    if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end

function A = set_pol_align(A, qwp_deg, oq_deg)
    A.qwp_deg = qwp_deg;  A.oq_deg = oq_deg;
end

function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end

function load_arm(A, QWP, an_deg, grid)
%LOAD_ARM  Load the deck, optionally rewrite the DM grid IN THE LOADED MODEL
%   (the live poke), then set every polarizing element.
    macos.load_rx(A.rx);  b = A.b;
    if nargin >= 4 && ~isempty(grid)
        macos.set_elt_grid(A.iTO, macos.get_elt_grid_spacing(A.iTO), grid);
    end
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end

function E = arm_field(A, QWP, an_deg, grid)
    load_arm(A, QWP, an_deg, grid);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP, grid)
    E0  = arm_field(A, QWP,  0, grid);
    E45 = arm_field(A, QWP, 45, grid);
    E90 = arm_field(A, QWP, 90, grid);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function R = probe_rt(rx_t, iT, rx_r, iR)
%PROBE_RT  The engine's R and T of the cemented diagonal, per polarization.
%   The amplitude ratio ACROSS the diagonal -- the field just after the
%   entrance face vs just after the diagonal.  The glass between is lossless,
%   so the propagation phase has unit modulus and drops out of |ratio|.  The
%   source frame for the +x chief is xGrid = yhat (the diagonal's p axis) and
%   yGrid = zhat (its s axis); PolIn is set to the probe axis so it passes
%   the state instead of projecting it.
    P = {rx_t, iT, 'T'; rx_r, iR, 'R'};  Q = struct();
    for k = 1:2
        for pol = {'s','p'}
            macos.load_rx(P{k,1});
            if strcmp(pol{1},'s'), ax = [0 0 1];  Ex = [0 0];  Ey = [1 0];
            else,                  ax = [0 1 0];  Ex = [1 0];  Ey = [0 0]; end
            ip = P{k,2};
            macos.polarizer(ip(1)-1, 'axis', ax);
            macos.polarization('on','Ex',Ex,'Ey',Ey);
            macos.vector_diffraction(true);
            macos.trace(ip(1));  f1 = macos.ray_field(ip(1));
            macos.trace(ip(2));  f2 = macos.ray_field(ip(2));
            ok = (f1.status == 0) & (f2.status == 0);
            a1 = sqrt(abs(f1.Ex).^2 + abs(f1.Ey).^2 + abs(f1.Ez).^2);
            a2 = sqrt(abs(f2.Ex).^2 + abs(f2.Ey).^2 + abs(f2.Ez).^2);
            Q.([P{k,3} pol{1}]) = median(a2(ok)./a1(ok))^2;
        end
    end
    R = struct('Ts',Q.Ts, 'Tp',Q.Tp, 'Rs',Q.Rs, 'Rp',Q.Rp);
end

function h = meas_surface(A, QWP, M, Sr, p_null, THETAS, LAM)
%MEAS_SURFACE  Put shape M on the DM, run the four-step measurement against
%   the reference arm, and return the recovered SURFACE in mm.  Differential
%   against the flat-DM phase, so every static term cancels.  Three traces.
    d = angle(exp(1i*(fourstep(analyzer_basis(A, QWP, M), Sr, THETAS) - p_null)));
    h = d * LAM/(4*pi);
end

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end

function e = arm_state(A, QWP, iElt)
    load_arm(A, QWP, 0, []);
    macos.trace(iElt);  f = macos.ray_field(iElt);
    ok = f.status == 0;
    psi = A.b.E(iElt).psi(:);
    u1 = macos.design.Bench.perp(psi);  u2 = cross(psi, u1);
    e1 = f.Ex*u1(1) + f.Ey*u1(2) + f.Ez*u1(3);
    e2 = f.Ex*u2(1) + f.Ey*u2(2) + f.Ez*u2(3);
    r  = e2(ok)./e1(ok);  a = median(abs(e1(ok)));
    e  = [a; a*(median(real(r)) + 1i*median(imag(r)))];
end

function az = arm_azimuth(A, QWP, qwp_deg)
    A.qwp_deg = qwp_deg;
    e = arm_state(A, QWP, A.iRC);
    az = 0.5*atan2d(2*real(conj(e(1))*e(2)), abs(e(1))^2 - abs(e(2))^2);
end

function [best, map] = register_to_dm(A, ix, Mdm, N_G, DX_G, h, msk)
%REGISTER_TO_DM  The instrument's pupil mapping, measured from the trace: one
%   (DM position, detector position) pair per surviving ray.
    macos.load_rx(A.rx);
    s1 = macos.trace(ix.iTO);   ito  = macos.get_ray_info(s1.nRays);
    s2 = macos.trace(ix.iDET);  idet = macos.get_ray_info(s2.nRays);
    okr = ito.ok_trace(:) & ito.ok_pass(:) & idet.ok_trace(:) & idet.ok_pass(:);
    psi1 = macos.get_elt_psi(ix.iTO);  vpt1 = macos.get_elt_vpt(ix.iTO);
    u1 = macos.design.Bench.perp(psi1);  v1 = cross(psi1, u1);
    xy_to = [u1.'; v1.'] * (ito.pos - vpt1);
    psi2 = macos.get_elt_psi(ix.iDET);
    u2 = macos.design.Bench.perp(psi2);  v2 = cross(psi2, u2);
    xy_d = [u2.'; v2.'] * (idet.pos - idet.pos(:,1));
    xy_to = xy_to(:,okr);  xy_d = xy_d(:,okr);
    Aaf = [xy_d.' ones(nnz(okr),1)] \ xy_to.';
    Lm  = Aaf(1:2,:).';
    [~,Ss,~] = svd(Lm);  sm = diag(Ss);
    nl  = xy_to - (Lm*xy_d + Aaf(3,:).');
    map = struct('mag', sqrt(abs(det(Lm))), 'anam_pct', 100*(sm(1)/sm(2)-1), ...
                 'nonlin_mm', sqrt(mean(sum(nl.^2,1))));
    Fx = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(1,:).', 'linear','linear');
    Fy = scatteredInterpolant(xy_d(1,:).', xy_d(2,:).', xy_to(2,:).', 'linear','linear');
    N = size(h,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = sum(cg(msk))/nnz(msk);  cy = sum(rg(msk))/nnz(msk);
    dxp = macos.dx_at(ix.iDET, 'mm');
    a1 = (cg-cx)*dxp;  a2 = (rg-cy)*dxp;
    c_d = mean(xy_d, 2);
    axs = ((1:N_G)-(N_G+1)/2)*DX_G;
    hm = h - mean(h(msk));
    cands = {a1,a2; a1,-a2; -a1,a2; -a1,-a2; a2,a1; a2,-a1; -a2,a1; -a2,-a1};
    best = struct('c',-inf, 'i',1);
    for c = 1:size(cands,1)
        [cc, ht] = reg_corr([0 0 0 0], cands{c,1}, cands{c,2}, c_d, Fx, Fy, axs, Mdm, hm, msk);
        if cc > best.c, best = struct('c',cc, 'ht',ht, 'i',c); end
    end
    A1 = cands{best.i,1};  A2 = cands{best.i,2};
    p = fminsearch(@(q) -reg_corr(q,A1,A2,c_d,Fx,Fy,axs,Mdm,hm,msk), [0 0 0 0], ...
                   optimset('TolX',1e-7,'TolFun',1e-10,'Display','off'));
    [c2, ht2, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk);
    if c2 > best.c, best.c = c2;  best.ht = ht2; end
    best.hm = hm;
    map.Xt = Xt;  map.Yt = Yt;     % the reusable resampler (see reg_corr)
end

function [c, ht, Xt, Yt] = reg_corr(p, A1, A2, c_d, Fx, Fy, axs, Mdm, hm, msk)
%   XT/YT are each detector pixel's position ON THE DM, in mm.  Returned so
%   the solved registration can be REUSED to resample any other shape (beat
%   8 injects a dozen of them) instead of re-solving it every time.
    s = exp(p(4));  ct = cos(p(3));  st = sin(p(3));
    X = s*(ct*A1 - st*A2) + c_d(1) + p(1);
    Y = s*(st*A1 + ct*A2) + c_d(2) + p(2);
    Xt = Fx(X,Y);  Yt = Fy(X,Y);
    ht = interpn(axs, axs, Mdm, Xt, Yt, 'spline', 0);
    ht = ht - mean(ht(msk));
    cc = corrcoef(hm(msk), ht(msk));  c = cc(1,2);
end

function show(Z, msk, N, box, cl, ttl)
    q = nan(N);  q(msk) = Z(msk);  q = sub(q, box);
    nexttile; imagesc(q, 'AlphaData', ~isnan(q)); axis image off;
    if ~isempty(cl), clim(cl); end
    colorbar; title(ttl);
end

function b = beam_box(msk, pad)
%BEAM_BOX  Padded bounding box of the illuminated pixels -- the beam is a
%   small disc on the padded diffraction array.
    [rr, cc] = find(msk);  N = size(msk,1);
    b = [max(1,min(rr)-pad) min(N,max(rr)+pad) max(1,min(cc)-pad) min(N,max(cc)+pad)];
end

function Z = sub(Z, b), Z = Z(b(1):b(2), b(3):b(4)); end

function m = erode_disc(msk, frac)
%ERODE_DISC  Keep the inner FRAC of the illuminated disc (no toolbox needed).
    N = size(msk,1);  [cg, rg] = meshgrid(1:N, 1:N);
    cx = mean(cg(msk));  cy = mean(rg(msk));
    r  = sqrt((cg-cx).^2 + (rg-cy).^2);
    m  = msk & (r <= frac*max(r(msk)));
end
