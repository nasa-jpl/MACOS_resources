% tma_3plus1_aoi_search.m  (mmacos/design/examples/tma_3plus1/)
% =====================================================================
%  CONSTRAINT FINDER: step the PM-SM separation (primary f/#) until
%  the coronagraph polarization preference is met.
% =====================================================================
%  THE CONSTRAINT (Dave, 2026-07-04): to keep polarization coupling
%  manageable a coronagraph front end prefers the per-mirror AOI
%  SPREAD across the rays of the beam < 15 deg.  The spread at M1 is
%  the primary's own convergence, ~ D/R1 -- independent of decenter --
%  so the ONLY real knob is a SLOWER primary, i.e. a LONGER PM-SM
%  separation.  That acts directly against packaging (compactness),
%  so this script reports BOTH metrics per step and stops at the
%  first (most compact) geometry that meets the AOI preference.
%
%  Per step: tma_layout(D, pf, 20) derives a consistent Korsch
%  (Cassegrain feed + M3 relay, j18 form) -> conic-optimize -> carry
%  the conics into the FULL 4-mirror chain (M4 relay) -> compact
%  unobscured section -> verify ALL THREE constraints: per-mirror AOI
%  spread, body-in-beam clearance (M4 lives inside the M2->M3
%  corridor -- mm-level, so only the full chain can judge it), and
%  the launch-shroud envelope.  The chosen geometry is saved to
%  tma_3plus1_geometry.mat for tma_3plus1_optimize.m.
%
%  Run:  >> run('.../tma_3plus1/tma_3plus1_aoi_search.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
D          = 6.605;            % aperture (m)
SYS_FNUM   = 20;               % system f/#
PF_STEPS   = 1.2:0.2:3.2;      % primary f/# ladder (PM-SM ~ pf*D)
AOI_LIMIT  = 15;               % max per-mirror AOI spread (deg)
LAM        = 1.0e-6;
SECT_MARGIN= 0.01;             % compact clearance margin (fraction of D)
DPAST      = 1.2;              % M4 past the TMA focus (m); > f4
R4         = 1.5;              % M4 radius (concave, f4=0.75 m)
% =====================================================================

fprintf('====================================================================\n');
fprintf(' AOI-spread constraint finder | D=%.2f m f/%g | limit %g deg\n', ...
        D, SYS_FNUM, AOI_LIMIT);
fprintf('====================================================================\n');
fprintf(' %6s %9s %9s %9s | %6s %6s %6s %6s | %5s %8s | %s\n', ...
        'prim f/#','t1(m)','dy/D','shroud/D', ...
        'sprM1','sprM2','sprM3','sprM4','clear','patchWFE','verdict');

chosen = [];
hist = struct('pf',{},'R',{},'t',{},'dy',{},'spread',{}, ...
              'shroud_over_D',{},'clear',{},'ok',{});
for pf = PF_STEPS
    try
        [R, tt, info] = macos.design.tma_layout(D, pf, SYS_FNUM);
        % 3-mirror conic optimize (the conic source)
        t3 = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
                'model_size',256,'wavelength_m',LAM,'grid_npts',41);
        t3.add_mirror('M1','radius_m',R(1),'spacing_after_m',tt(1));
        t3.add_mirror('M2','radius_m',R(2),'spacing_after_m',tt(2),'convex',true);
        t3.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
        t3.add_focal_plane('FP');
        t3.build();
        t3.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1], ...
                    'max_iters',120);
        K3  = [t3.spec.elt(1).Kc t3.spec.elt(2).Kc t3.spec.elt(3).Kc];
        bfd = norm(t3.spec.elt(end).Vpt - t3.spec.elt(3).Vpt);

        % the FULL 4-mirror chain (M4 relay carried in)
        t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
                'model_size',256,'wavelength_m',LAM,'grid_npts',41);
        t.add_mirror('M1','radius_m',R(1),'spacing_after_m',tt(1),'conic',K3(1));
        t.add_mirror('M2','radius_m',R(2),'spacing_after_m',tt(2),'convex',true,'conic',K3(2));
        t.add_mirror('M3','radius_m',R(3),'spacing_after_m',bfd+DPAST,'conic',K3(3));
        t.add_mirror('M4','radius_m',R4,'spacing_after','derive','conic',0);
        t.add_focal_plane('FP2');
        t.build();
        dy = t.set_offaxis('all','margin',SECT_MARGIN);
        t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1], ...
                   'elts',1:4,'max_iters',80);
        t.set_offaxis('none');
        optP = macos.design.field_ring(0.25,'units','arcmin');
        rb = t.optimize('fields',optP,'dofs',[1 1 0 0 1 0 1 1], ...
                        'elts',1:4,'max_iters',120);
        t.set_offaxis('none');
        wpatch = max(rb.wfe_after)/LAM;

        t.realize_apertures('fields',[0 0; optP],'margin',0.05,'quiet',true);
        rep = t.check_clipping('noload',true,'quiet',true);
        okclear = all([rep.ok]);
        aoi = aoi_report(t, 'quiet',true);
        pk  = packaging_report(t, 'quiet',true);
        spr = [aoi.aoi_spread_deg];
        ok  = all(spr <= AOI_LIMIT) && okclear;
        hist(end+1) = struct('pf',pf,'R',R,'t',tt,'dy',dy, ...
                             'spread',spr,'shroud_over_D',pk.shroud_over_D, ...
                             'clear',okclear,'ok',ok);  %#ok<AGROW>
        fprintf([' %6.1f %9.3f %9.2f %9.2f | %6.1f %6.1f %6.1f %6.1f | ' ...
                 '%5s %8.4f | %s\n'], ...
            pf, tt(1), dy/D, pk.shroud_over_D, ...
            spr(1), spr(2), spr(3), spr(min(4,numel(spr))), ...
            ternary(okclear,'yes','NO'), wpatch, ...
            ternary(ok,'MEETS','over'));
        if ok && isempty(chosen)
            chosen = struct('pf',pf,'R',R,'t',tt,'dy',dy,'spread',spr, ...
                            'shroud_over_D',pk.shroud_over_D, ...
                            'D',D,'sys_fnum',SYS_FNUM,'lambda',LAM, ...
                            'sect_margin',SECT_MARGIN,'dpast',DPAST, ...
                            'r4',R4,'info',info);
            break;                       % first (most compact) compliant step
        end
    catch ME
        fprintf(' %6.1f   FAILED: %s\n', pf, ME.message);
    end
end

if isempty(chosen)
    fprintf('\nNO step met the %g-deg spread -- extend PF_STEPS.\n', AOI_LIMIT);
else
    gfile = fullfile(exdir,'tma_3plus1_geometry.mat');
    save(gfile,'chosen','hist');
    fprintf(['\nCHOSEN: primary f/%.1f (t1 = %.2f m, %.1fx the j18 7.17 m) --\n' ...
             '  spreads [%.1f %.1f %.1f] deg, decenter %.2f D.\n' ...
             '  Geometry saved: %s\n' ...
             '  Next: run tma_3plus1_optimize.m\n'], ...
            chosen.pf, chosen.t(1), chosen.t(1)/7.169, ...
            chosen.spread(1), chosen.spread(2), chosen.spread(3), ...
            chosen.dy/D, gfile);
end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
