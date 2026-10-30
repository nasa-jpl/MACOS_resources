% tma_3plus1.m  (mmacos/templates/10_telescopes/ -- an adaptable design study)
% =====================================================================
%  3+1 CORONAGRAPH FRONT END -- unobscured conic TMA + M4 pupil relay
%  (the j18-geometry DEMO; see tma_3plus1_aoi_search.m /
%  tma_3plus1_optimize.m for the polarization-safe sibling)
% =====================================================================
%  ARCHITECTURE.  Three conic mirrors (j18mono-geometry Korsch, 6.6 m
%  f/20, unobscured eccentric-pupil section) form the SCIENCE FOCUS --
%  the shared focal plane where instrument pickoffs live (the 3-mirror
%  holds a 5'-diameter circular field at ~0.095 waves @ 1 um; see
%  ../tma_centered).  A 4th FIELD MIRROR just past that focus relays
%  the CORONAGRAPH CHANNEL (a ~30 arcsec patch) to FP2 and -- the
%  point -- relays the exit pupil to an accessible plane and FLATTENS
%  it (the bare TMA pupil carries +0.7 mm defocus / 1.1 mm astig,
%  macos.pupil_quality; the relayed pupil comes out ~10x flatter with
%  M4 varying in the patch balance).  A relay cannot carry the wide
%  field: at EFL=132 m a 2.5' half-field walks +-96 mm across M4 --
%  wide field stays at the TMA focus, the relay serves the small-field
%  channel.
%
%  BUILD MECHANICS (the parts that were hard to get right):
%   - seidel cannot seed a relay-past-focus 4-mirror chain (the
%     degenerate-reimager regime): optimize the 3-mirror TMA first and
%     CARRY its conics via add_mirror(...,'conic',K); M4 seeds K=0.
%   - M4 must exceed its focal length past the focus (DPAST > f4) or
%     the relay conjugate is VIRTUAL and the derive places FP2 behind
%     M4.
%   - hold M4 out of the AXIAL image solves ('elts',1:3) but let it
%     vary in the PATCH balance -- the intermediate image walks across
%     M4 with field, so it has real field leverage (and flattens the
%     pupil almost for free).
%   - packaging (Dave): clear with a body-scaled margin -- the 0.05*D
%     default demands 330 mm around a 25 mm relay mirror and drives
%     the section to 1.5*D; 0.01*D keeps it at 0.84*D.
%
%  KNOWN OUT-OF-PREFERENCE (kept as a demo; see the _aoi_search
%  sibling): coronagraph polarization prefers the per-mirror AOI
%  SPREAD across the beam < 15 deg.  The j18 parent is an f/1.2
%  primary, whose convergence alone puts ~21 deg on M1 and ~24 deg on
%  M2 (spread ~ D/R1 -- decenter barely moves it).  Meeting 15 deg
%  needs a slower primary / longer PM-SM separation: that trade study
%  is tma_3plus1_aoi_search.m.
%
%  Run:  >> run('.../templates/10_telescopes/tma_3plus1/tma_3plus1.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
D      = 6.605;                          % aperture (m) -- j18mono parent
R      = [15.879722 1.778913 3.016227];  % |radii| M1 / M2(convex) / M3
TBET   = [7.169041556 7.965313479];      % spacings (m)
LAM    = 1.0e-6;                         % score at 1 um
DPAST  = 1.2;                            % M4 past the TMA focus (m); > f4!
R4     = 1.5;                            % M4 radius (concave; f4 = 0.75 m)
PATCH_RAD_ASEC = 15;                     % coronagraph patch RADIUS (arcsec)
SECT_MARGIN    = 0.01;                   % clearance margin (fraction of D)
% =====================================================================

optF = macos.design.field_ring(PATCH_RAD_ASEC/60, 'units','arcmin');

fprintf('====================================================================\n');
fprintf(' 3+1 coronagraph front end | D=%.2f m f/20 | %g" patch | j18 parent\n', ...
        D, 2*PATCH_RAD_ASEC);
fprintf('====================================================================\n');

%% -- [1] the 3-mirror TMA, centered, conic-optimized ------------------
[tA, rA] = tma_conic_recipe('D',D,'R',R,'spacings',TBET,'lambda',LAM, ...
        'section',false,'fields',macos.design.field_ring(0.5,'units','arcmin'));
K3  = rA.conics;
bfd = norm(tA.spec.elt(end).Vpt - tA.spec.elt(3).Vpt);
fprintf('\n[1] TMA: axial %.4f waves; K=[%.4f %.4f %.4f]; M3->focus %.3f m\n', ...
        rA.wfe_axial, K3, bfd);

%% -- [2] the 3+1 chain: conics carried, M4 sphere ---------------------
t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',256,'wavelength_m',LAM,'grid_npts',41);
t.add_mirror('M1','radius_m',R(1),'spacing_after_m',TBET(1),'conic',K3(1));
t.add_mirror('M2','radius_m',R(2),'spacing_after_m',TBET(2),'convex',true,'conic',K3(2));
t.add_mirror('M3','radius_m',R(3),'spacing_after_m',bfd+DPAST,'conic',K3(3));
t.add_mirror('M4','radius_m',R4,'spacing_after','derive','conic',0);
t.add_focal_plane('FP2');
t.build();  nE = numel(t.spec.elt);
macos.trace(nE);  W = macos.opd();  v = W(isfinite(W)&W~=0);
fprintf('[2] 3+1 coaxial, conics carried: %.4f waves on-axis at FP2\n', std(v)/LAM);
t.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1], ...
           'elts',1:3,'max_iters',80);

%% -- [3] COMPACT unobscured section (packaging rule) ------------------
dy = t.set_offaxis('all','margin',SECT_MARGIN);
t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1],'elts',1:4,'max_iters',100);
t.set_offaxis('none');
macos.trace(nE);  W = macos.opd();  v = W(isfinite(W)&W~=0);
fprintf('[3] compact section (%.2f D) + axial refigure: %.4f waves\n', dy/D, std(v)/LAM);

%% -- [4] coronagraph-patch balance (M4 varying) -----------------------
rb = t.optimize('fields',optF,'dofs',[1 1 0 0 1 0 1 1],'elts',1:4,'max_iters',200);
t.set_offaxis('none');
fprintf('[4] %g"-dia patch balance: worst %.4f waves\n', ...
        2*PATCH_RAD_ASEC, max(rb.wfe_after)/LAM);

%% -- [5] clearance + the relayed pupil --------------------------------
t.realize_apertures('fields',[0 0; optF],'margin',0.05,'quiet',true);
rep = t.check_clipping('noload',true,'quiet',true);
fprintf('[5] clearance: %d/%d clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','** OBSCURED'));

t.add_pupil(numel(t.spec.elt));
try
    pq = macos.pupil_quality(numel(t.spec.elt)-1);
    fprintf(['[6] relayed pupil: dia %.1f mm, defocus %+.3f mm, astig %.3f mm' ...
             '   (bare TMA: +0.7 / 1.1)\n'], ...
            pq.diameter*1e3, pq.defocus*1e3, max(abs(pq.astig))*1e3);
catch ME, fprintf('[6] pupil_quality failed: %s\n', ME.message); end

%% -- [7] AOI report (polarization: SPREAD < 15 deg preferred) ---------
aoi = aoi_report(t, 'fields', optF);
fprintf(['[7] worst AOI spread %.1f deg -- OUT of the 15-deg preference on\n' ...
         '    M1/M2 (f/1.2 parent; see tma_3plus1_aoi_search.m).\n'], ...
        max([aoi.aoi_spread_deg]));

%% -- [8] deliverables -------------------------------------------------
rxfile = fullfile(exdir,'tma_3plus1.in');  matfile = fullfile(exdir,'tma_3plus1.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[8] saved: %s\n           + %s\n', rxfile, matfile);
try
    f1 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f1, fullfile(exdir,'tma_3plus1_layout.png'));
    fprintf('    layout: tma_3plus1_layout.png\n');
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
