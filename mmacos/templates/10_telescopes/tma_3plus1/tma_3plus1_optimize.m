% tma_3plus1_optimize.m  (mmacos/templates/10_telescopes/tma_3plus1/)
% =====================================================================
%  POLARIZATION-SAFE 3+1: full optimization at the geometry found by
%  tma_3plus1_aoi_search.m (run that FIRST -- it writes
%  tma_3plus1_geometry.mat).
% =====================================================================
%  Same staged flow as the j18-geometry demo (tma_3plus1.m), at the
%  slower-primary / longer PM-SM geometry that meets the < 15 deg
%  per-mirror AOI-spread preference: 3-mirror conic optimize -> carry
%  conics into the 4-mirror chain -> compact unobscured section ->
%  coronagraph-patch balance with M4 varying -> clearance + relayed
%  pupil + AOI verification -> deliverables
%  (tma_3plus1_polsafe.in/.mat + layout figure).
%
%  Run:  >> run('.../tma_3plus1/tma_3plus1_optimize.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

gfile = fullfile(exdir,'tma_3plus1_geometry.mat');
if ~exist(gfile,'file')
    error('tma_3plus1_optimize:nogeom', ...
        'run tma_3plus1_aoi_search.m first (no %s).', gfile);
end
G = load(gfile);  g = G.chosen;

% ====================  USER DESIGN CHOICES  ==========================
DPAST  = 1.2;            % M4 past the TMA focus (m); MUST exceed f4.
                         % Overridden by the search's chosen geometry when
                         % present (the finder verifies clearance on the
                         % FULL 4-mirror chain -- M4 lives inside the
                         % M2->M3 corridor, so its clearance is mm-level
                         % and geometry-specific).
R4     = 1.5;            % M4 radius (concave; f4 = 0.75 m)
PATCH_RAD_ASEC = 15;     % coronagraph patch RADIUS (arcsec)
% =====================================================================
D = g.D;  LAM = g.lambda;
if isfield(g,'dpast'), DPAST = g.dpast; end
if isfield(g,'r4'),    R4    = g.r4;    end
optF = macos.design.field_ring(PATCH_RAD_ASEC/60, 'units','arcmin');

fprintf('====================================================================\n');
fprintf(' polarization-safe 3+1 | D=%.2f m f/%g | primary f/%.1f (t1=%.2f m)\n', ...
        D, g.sys_fnum, g.pf, g.t(1));
fprintf('====================================================================\n');

%% -- [1] the 3-mirror TMA at the chosen geometry ----------------------
[tA, rA] = tma_conic_recipe('D',D,'R',g.R,'spacings',g.t,'lambda',LAM, ...
        'section',false,'fields',macos.design.field_ring(0.5,'units','arcmin'));
K3  = rA.conics;
bfd = norm(tA.spec.elt(end).Vpt - tA.spec.elt(3).Vpt);
fprintf('\n[1] TMA: axial %.4f waves; K=[%.4f %.4f %.4f]; M3->focus %.3f m\n', ...
        rA.wfe_axial, K3, bfd);

%% -- [2] the 3+1 chain ------------------------------------------------
t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',256,'wavelength_m',LAM,'grid_npts',41);
t.add_mirror('M1','radius_m',g.R(1),'spacing_after_m',g.t(1),'conic',K3(1));
t.add_mirror('M2','radius_m',g.R(2),'spacing_after_m',g.t(2),'convex',true,'conic',K3(2));
t.add_mirror('M3','radius_m',g.R(3),'spacing_after_m',bfd+DPAST,'conic',K3(3));
t.add_mirror('M4','radius_m',R4,'spacing_after','derive','conic',0);
t.add_focal_plane('FP2');
t.build();  nE = numel(t.spec.elt);
macos.trace(nE);  W = macos.opd();  v = W(isfinite(W)&W~=0);
fprintf('[2] 3+1 coaxial, conics carried: %.4f waves at FP2\n', std(v)/LAM);
t.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1], ...
           'elts',1:3,'max_iters',80);

%% -- [3] compact section + axial refigure -----------------------------
dy = t.set_offaxis('all','margin',g.sect_margin);
t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1],'elts',1:4,'max_iters',100);
t.set_offaxis('none');
macos.trace(nE);  W = macos.opd();  v = W(isfinite(W)&W~=0);
fprintf('[3] compact section (%.2f D) + refigure: %.4f waves\n', dy/D, std(v)/LAM);

%% -- [4] coronagraph-patch balance ------------------------------------
rb = t.optimize('fields',optF,'dofs',[1 1 0 0 1 0 1 1],'elts',1:4,'max_iters',200);
t.set_offaxis('none');
fprintf('[4] %g"-dia patch balance: worst %.4f waves\n', ...
        2*PATCH_RAD_ASEC, max(rb.wfe_after)/LAM);

%% -- [5] verify: clearance, pupil, AOI --------------------------------
t.realize_apertures('fields',[0 0; optF],'margin',0.05,'quiet',true);
rep = t.check_clipping('noload',true,'quiet',false);
fprintf('[5] clearance: %d/%d clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','** OBSCURED'));

t.add_pupil(numel(t.spec.elt));
try
    pq = macos.pupil_quality(numel(t.spec.elt)-1);
    fprintf('[6] relayed pupil: dia %.1f mm, defocus %+.3f mm, astig %.3f mm\n', ...
            pq.diameter*1e3, pq.defocus*1e3, max(abs(pq.astig))*1e3);
catch ME, fprintf('[6] pupil_quality failed: %s\n', ME.message); end

pk = packaging_report(t);
fprintf('[6b] shroud: %.2f x D diameter, %.1f m long (launch-shroud metric)\n', ...
        pk.shroud_over_D, pk.length_m);

aoi = aoi_report(t, 'fields', optF);
fprintf('[7] worst AOI spread %.1f deg -> %s (limit 15)\n', ...
        max([aoi.aoi_spread_deg]), ...
        ternary(max([aoi.aoi_spread_deg]) <= 15,'MEETS the preference','** over'));

%% -- [8] deliverables --------------------------------------------------
rxfile = fullfile(exdir,'tma_3plus1_polsafe.in');
matfile = fullfile(exdir,'tma_3plus1_polsafe.mat');
t.save(rxfile);  t.save_spec(matfile);
fprintf('[8] saved: %s\n           + %s\n', rxfile, matfile);
try
    f1 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f1, fullfile(exdir,'tma_3plus1_polsafe_layout.png'));
    fprintf('    layout: tma_3plus1_polsafe_layout.png\n');
catch ME, fprintf('    layout skipped (%s)\n', ME.message); end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
