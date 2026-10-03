function T = strict_stage_table(map_n)
%STRICT_STAGE_TABLE  Score ALL FOUR committed stage solves under the strict
%   metric, at EPD 4060.  Pure evaluation -- no rebuild, no optimizer.
%
%   T = STRICT_STAGE_TABLE()    map_n = 9 (the packet's box sampling)
%
%   For each stage N, the committed deck `rodgers1_epd4060_stageN.in` is
%   scored by STRICT_WFE_DECK over its own 0.2 deg box, with that stage's
%   OWN solved FPA (the deck's FocalPlane) held frozen -- Rodgers'
%   procedure, stage by stage.  Stage 1's box sits about 0 deg (its deck's
%   chief ray is on-axis); stages 2-4 sit about their deck's +0.5 deg bias.
%   The bias is read back FROM the deck, so there is no bias to double.
%
%   Cross-validation, run first and asserted: stage 2 must reproduce the
%   Addendum-3 §D gate-3 number (429.6 / 246.8 nm), which was obtained by
%   the independent Telescope-object path (STRICT_WFE).  If it does not,
%   nothing below is trustworthy.
%
%   Writes rodgers1_epd4060_strict_stages.mat and one field map per stage,
%   rodgers1_epd4060_stageN_strict.png.

    if nargin < 1, map_n = 9; end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common();
    lam_nm = P.lambda_m*1e9;

    GT = {P.gt.s1_onaxis_box, P.gt.s2_box, P.gt.s3_box, P.gt.s4_box};
    Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');
    T = struct('map_n',map_n,'lambda_nm',lam_nm,'stage',cell(1,4));

    % a throwaway Telescope, used only as the renderer for view_field_map
    trend = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_mm', 4060, 'wavelength_m', P.lambda_m, ...
                'model_size', P.model_size);
    trend.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    trend.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    trend.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    trend.build();          % resolve the spec so view_field_map's title is complete

    for st = 1:4
        deck = fullfile(here, sprintf('rodgers1_epd4060_stage%d.in', st));
        banner('STAGE %d -- %s', st, deck);
        s = strict_wfe_deck(deck, Frel);
        T(st).stage = st;  T(st).scan = s;  T(st).gt = GT{st};
        w = s.wfe_m(isfinite(s.wfe_m))*1e9;
        T(st).min = min(w);  T(st).max = max(w);  T(st).avg = mean(w);
        T(st).nfin = numel(w);  T(st).ntot = numel(s.wfe_m);
        fprintf('  deck bias   : [%+.4f %+.4f] deg   FPA Vpt [%.6f %.6f %.6f]\n', ...
                s.bias_deg, s.detector.Vpt);
        fprintf('  FPA psi     : [%.9f %.9f %.9f]\n', s.detector.psi);
        fprintf('  rays/field  : %d..%d   fields %d/%d finite\n', ...
                min(s.nrays), max(s.nrays), T(st).nfin, T(st).ntot);
        fprintf('  STRICT (nm) : min %9.3f  max %9.3f  avg %9.3f\n', ...
                T(st).min, T(st).max, T(st).avg);
        fprintf('  Rodgers (nm): min %9.3f  max %9.3f  avg %9.3f\n', ...
                GT{st}(1)*lam_nm, GT{st}(2)*lam_nm, GT{st}(3)*lam_nm);

        if st == 2
            d = abs(T(st).max - 429.627)/429.627;
            fprintf('  CROSS-CHECK vs Addendum-3 §D (429.627 nm max): %.3e relative\n', d);
            assert(d < 5e-3, ...
                'strict_stage_table:crosscheck', ...
                ['stage-2 deck path does not reproduce the §D gate-3 number ' ...
                 '(%.4f vs 429.627 nm) -- do not trust the other stages.'], T(st).max);
        end

        % Field axes ABSOLUTE (deck bias added back), matching the committed
        % *_stageN_{global,refsphere}.png so the four panels overlay Mike's
        % slide sequence.  Keep the metric label SHORT -- view_field_map's
        % title is a single line on a 620 px figure and a long label clips
        % the family prefix off the left.
        png = fullfile(here, sprintf('rodgers1_epd4060_stage%d_strict.png', st));
        Fabs = s.fields*180/pi*60 + s.bias_deg*60;
        scan = struct('fields', Fabs, 'wfe', s.wfe(:), 'metric','strict');
        fig = trend.view_field_map(scan,'kind','contour','save',png,'visible',false);
        close(fig);
        T(st).png = png;
    end

    banner('STRICT METRIC ACROSS THE STUDY  (EPD 4060, nm @ %g nm, %dx%d box)', ...
           lam_nm, map_n, map_n);
    fprintf('  stage | strict max/avg  | Rodgers max/avg | max x | avg x\n');
    for st = 1:4
        g = GT{st};
        fprintf('  S%-4d | %7.1f/%-7.1f | %7.1f/%-7.1f | %5.2f | %5.2f\n', st, ...
            T(st).max, T(st).avg, g(2)*lam_nm, g(3)*lam_nm, ...
            T(st).max/(g(2)*lam_nm), T(st).avg/(g(3)*lam_nm));
    end

    save(fullfile(here,'rodgers1_epd4060_strict_stages.mat'),'T');
    fprintf('\nsaved rodgers1_epd4060_strict_stages.mat + 4 stage maps\n');
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
