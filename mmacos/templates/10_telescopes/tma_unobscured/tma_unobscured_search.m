% tma_unobscured_search.m  (mmacos/templates/10_telescopes/tma_unobscured/)
% =====================================================================
%  CONSTRAINT FINDER: the visible-band UNOBSCURED TMA front end --
%  how slow does M1 need to be?
% =====================================================================
%  THE BRIEF (Dave, 2026-07-06): an unobscured TMA for a VISIBLE
%  (500 nm) telescope -- the front end for a coronagraph, plus imager
%  and spectrometer.  Two rules drive the geometry:
%   1. SLOWER M1 than the centered j18 (f/1.2): a slower primary makes
%      the eccentric-pupil section's per-mirror AOI SPREAD gentler --
%      the coronagraph polarization preference is < 15 deg across the
%      beam on every mirror -- and shrinks the clearing decenter.  The
%      price is tube length (t1 grows with f1).
%   2. KEEP M2 CLOSE TO THE SOURCE->M1 BEAM (packaging): the whole
%      train must fit a cylindrical launch shroud.  set_offaxis('all')
%      already finds the MINIMAL clearing decenter (M2 as close to the
%      incoming beam as the margin allows); slower M1 shrinks it.
%
%  The finder walks the primary-f/# ladder, builds each candidate from
%  the closed-form Korsch layout (macos.design.tma_layout), solves the
%  conics, extracts the minimal unobscured section, and judges each on
%  the full 3-D clearance + per-mirror AOI spread + shroud -- the
%  slower-M1 trade table.  The design point is F1_PICK = f/2.5 (Dave
%  2026-07-06), verified MEETS and saved to tma_unobscured_geometry.mat
%  for tma_unobscured.m.
%
%  Run:  >> run('.../tma_unobscured/tma_unobscured_search.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
D          = 6.605;             % aperture (m)
LAM        = 0.5e-6;            % center wavelength: 500 nm (visible)
SYS_FNUM   = 20;                % system f/# (EFL = 132 m; lambda*F = 10 um
                                % -> Nyquist at ~5 um detector pixels)
F1_STEPS   = [1.5 2.0 2.5 3.0]; % the slower-M1 ladder (primary f/#)
F1_PICK    = 2.5;               % the design point (Dave 2026-07-06:
                                % "go with f/2.5" -- comfortably under
                                % the AOI rule with margin for the
                                % coronagraph, at acceptable tube)
FEED_FNUM  = 10;                % Cassegrain FEED f/# (= f1 * m2), held
                                % constant across the ladder (j18: 1.2 x
                                % 8 = 9.6).  Holding m2 instead drives
                                % the feed toward the system f/# as M1
                                % slows -- the M3 relay degenerates to
                                % 1:1 and the clearing decenter blows up
                                % (seen at m2=8: f/2.5 needs > 2.5*D).
FIELD_RAD  = 0.5;               % science-field RADIUS (arcmin) for the
                                % conic solve + the demo's FOV ladder
AOI_MAX    = 15;                % per-mirror AOI SPREAD preference (deg,
                                % coronagraph polarization rule)
MARGIN     = 0.05;              % section clearance margin (x D)
% =====================================================================

fprintf('====================================================================\n');
fprintf(' Unobscured visible TMA finder | D=%.2f m | %g nm | system f/%g\n', ...
        D, LAM*1e9, SYS_FNUM);
fprintf('====================================================================\n');
fprintf(['\n slower M1 -> gentler section AOI + smaller clearing decenter\n' ...
         ' (M2 stays close to the source->M1 beam), but a longer tube.\n\n']);
fprintf(' %4s | %6s %7s %7s | %9s %8s %8s | %s\n', 'f/1', 't1 (m)', ...
        'dy (m)', 'dy/D', 'AOIspread', 'shroud*D', 'len (m)', 'verdict');

best = [];
for f1 = F1_STEPS
    m2 = FEED_FNUM / f1;                 % constant-feed secondary mag
    [R, tt, info] = macos.design.tma_layout(D, f1, SYS_FNUM, ...
                        'secondary_mag', m2);
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
            'model_size',256,'wavelength_m',LAM,'grid_npts',41);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',tt(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',tt(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP');
    t.build();
    t.optimize('fields_arcmin',[FIELD_RAD/2 FIELD_RAD], ...
               'dofs',[0 0 0 0 0 0 0 1],'max_iters',80);
    % AOI-safe sections are shroud-expensive: the clearing decenter can
    % exceed the solver's default 1.5*D bisection bound -- raise it and
    % let the trade show honestly in the shroud column.
    dy = t.set_offaxis('all', 'margin', MARGIN, 'max_dist', 2.5*D);

    aoi = aoi_report(t, 'quiet', true);
    pow = abs([t.spec.elt.Kr]) < 1e21;
    spr = max([aoi([aoi.elt] <= numel(pow) & pow([aoi.elt])).aoi_spread_deg]);
    pk  = packaging_report(t, 'quiet', true);
    cc  = t.check_clipping('noload', true, 'quiet', true);
    ok  = all([cc.obstructs] == 0);
    meets = ok && spr < AOI_MAX;
    fprintf(' %4.1f | %6.2f %7.3f %7.3f | %8.1f%s %8.2f %8.2f | %s\n', ...
            f1, tt(1), dy, dy/D, spr, ternary(spr<AOI_MAX,' ','*'), ...
            pk.shroud_over_D, pk.length_m, ...
            ternary(meets,'MEETS', ternary(ok,'clear, AOI high','OBSCURED')));
    if abs(f1 - F1_PICK) < 1e-9        % the design point (Dave)
        if ~meets
            error(['the f/%.1f design point does not meet ' ...
                   '(clear=%d, AOI spread %.1f deg).'], f1, ok, spr);
        end
        K = [t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc];
        best = struct('D',D, 'lambda',LAM, 'sys_fnum',SYS_FNUM, ...
            'f1',f1, 'sec_mag',m2, 'feed_fnum',FEED_FNUM, ...
            'R',R, 'TBET',tt, 'K',K, ...
            'decenter',dy, 'margin',MARGIN, 'field_rad',FIELD_RAD, ...
            'aoi_spread',spr, 'shroud_over_D',pk.shroud_over_D, ...
            'info',info);
    end
end

if isempty(best)
    error('F1_PICK=%.1f is not in F1_STEPS.', F1_PICK);
end
gfile = fullfile(exdir,'tma_unobscured_geometry.mat');
chosen = best;  save(gfile, 'chosen');
fprintf(['\n chosen: f/%.1f primary (design point, AOI spread %.1f deg ' ...
         '< %g), decenter %.3f m\n (%.2f x D); shroud %.2f x D -- the ' ...
         'AOI-safe section''s honest packaging price.\n saved -> %s\n'], ...
        best.f1, best.aoi_spread, AOI_MAX, best.decenter, best.decenter/D, ...
        best.shroud_over_D, gfile);

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
