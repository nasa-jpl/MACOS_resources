% s1_fov_sweep.m  (mmacos/design/examples/e2e2/ -- stage-1 diagnostic)
% =====================================================================
%  HOW WIDE A FIELD WILL THIS TELESCOPE ACTUALLY HOLD?
% =====================================================================
%  Stage 1 lands at 0.638 nm RMS over the 0.2 deg box -- 56x inside the
%  500 nm diffraction-limit bar.  That headroom is an invitation to ask
%  for more field, and this answers the question by measurement rather
%  than by extrapolating the residual.
%
%  TWO CURVES, and the difference between them is the whole point:
%
%    AS-SOLVED   the committed 0.2 deg design, scored out to +-0.3 deg.
%                What the current telescope delivers if you simply point
%                further off its own axis.  Falls off fast -- the conics
%                were balanced over a box 3x smaller.
%
%    RE-SOLVED   a fresh solve AT each candidate box, scored on that box.
%                What the ARCHITECTURE can do.  Three conics null
%                spherical, coma and astigmatism about the axis, so the
%                residual they cannot reach is higher order plus the
%                Petzval field curvature, and re-balancing buys real
%                field.  This is the curve that answers "what fraction
%                of 0.6 deg is achievable".
%
%  Solving AT the target field, not scoring a design solved elsewhere,
%  is the standing rule (conic-basin path dependence).  Each candidate
%  therefore gets the full stage-1 recipe from scratch: build at K = 0,
%  on-axis conic seed, add_pupil, joint solve over that candidate's own
%  3x3 field set, score on its own uniform 9x9.
%
%  Reports the largest candidate meeting BOTH bars (RMS <= P.dl_rms_m at
%  the verdict rung, Strehl >= P.strehl_min) and writes
%  s1_fov_sweep.{mat,png} + s1_fov_sweep.txt.
%
%    >> run('.../design/examples/e2e2/s1_fov_sweep.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
mmroot = fileparts(fileparts(fileparts(exdir)));
run(fullfile(mmroot,'mmacos_setup.m'));
addpath(exdir);

P   = e2e2_params();
LAM = P.lambda_m;
HALF_DEG = [0.10 0.15 0.20 0.25 0.30];      % candidate HALF-fields

% Section [1] scores a committed deck before anything is built, so the
% engine has to be brought up explicitly -- s1_axial.m gets this for free
% from Telescope/build.
macos.init(P.model_size);

fprintf('\n====================================================================\n');
fprintf(' e2e2 stage-1 FIELD SWEEP | D=%.2f m f/%.4g | %g nm | DL bar %.2f nm RMS\n', ...
        P.D_m, P.system_fnum, LAM*1e9, P.dl_rms_m*1e9);
fprintf('====================================================================\n');

[R, tsp] = macos.design.tma_layout(P.D_m, P.primary_fnum, P.system_fnum, ...
        'secondary_mag', P.secondary_mag, 'int_focus_m', P.int_focus_m, ...
        'm3_behind_m',   P.m3_behind_m);

%% -- [1] AS-SOLVED: the committed 0.2 deg design, out to +-0.3 deg -----
deck0 = fullfile(exdir,'s1_axial.in');
S = struct('half_deg',HALF_DEG, 'as_solved',[], 're_solved',[]);
if exist(deck0,'file') == 2
    fprintf('\n[1] AS-SOLVED -- the committed 0.2 deg design, pointed further off axis\n');
    fprintf('    %9s %10s %10s %10s %12s\n', ...
            'half[deg]','max nm','avg nm','minStrehl','vs DL bar');
    as = nan(numel(HALF_DEG),3);
    for i = 1:numel(HALF_DEG)
        F = macos.design.field_grid(HALF_DEG(i)*60, P.score_n, 'units','arcmin');
        [L, info] = strict_ladder_deck(deck0, F, 'lambda', LAM);
        ok = all(isfinite(L),2);
        as(i,:) = [max(L(ok,P.score_rung)), mean(L(ok,P.score_rung)), ...
                   min(info.strehl(ok,P.score_rung))];
        fprintf('    %9.2f %10.3f %10.3f %10.4f %12s\n', HALF_DEG(i), ...
                as(i,1)*1e9, as(i,2)*1e9, as(i,3), ...
                verdict_(as(i,1) <= P.dl_rms_m && as(i,3) >= P.strehl_min));
    end
    S.as_solved = as;
else
    fprintf('\n[1] AS-SOLVED skipped -- s1_axial.in not built\n');
end

%% -- [2] RE-SOLVED: a fresh stage-1 solve AT each candidate box --------
fprintf(['\n[2] RE-SOLVED -- full stage-1 recipe at each candidate box\n' ...
         '    (solve AT the target field; a design solved elsewhere and ' ...
         'merely re-scored\n     understates what the architecture can do)\n']);
fprintf('    %9s %10s %10s %10s %12s   %s\n', ...
        'half[deg]','max nm','avg nm','minStrehl','vs DL bar','conics K');
rs = nan(numel(HALF_DEG),3);
Kk = nan(numel(HALF_DEG),3);
decks = cell(1,numel(HALF_DEG));
for i = 1:numel(HALF_DEG)
    h = HALF_DEG(i);
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_m', P.D_m, 'wavelength_m', LAM, ...
            'model_size', P.model_size, 'grid_npts', P.grid_npts);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',tsp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',tsp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP','ap_r',P.fp_body_r);
    t.set_hole('M1', P.M1_hole_m);
    t.build();
    t.optimize('fields_arcmin', [], 'dofs', P.dofs_conic, ...
               'max_iters', P.max_iters);            % on-axis conic seed
    t.add_pupil();
    Fs = macos.design.field_grid(h*60, P.solve_n, 'units','arcmin', ...
                                 'origin', false);
    t.optimize('fields', Fs, 'dofs', P.dofs_conic, ...
               'fpa_dofs', P.fpa_dofs, 'max_iters', P.max_iters);
    pm = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, ...
                       t.spec.elt));
    Kk(i,:) = arrayfun(@(k) t.spec.elt(k).Kc, pm);
    decks{i} = fullfile(exdir, sprintf('s1_fov_%03.0fmdeg.in', h*1000));
    t.save(decks{i});
    F = macos.design.field_grid(h*60, P.score_n, 'units','arcmin');
    [L, info] = strict_ladder_deck(decks{i}, F, 'lambda', LAM);
    ok = all(isfinite(L),2);
    rs(i,:) = [max(L(ok,P.score_rung)), mean(L(ok,P.score_rung)), ...
               min(info.strehl(ok,P.score_rung))];
    fprintf('    %9.2f %10.3f %10.3f %10.4f %12s   [%.5f %.5f %.5f]\n', ...
            h, rs(i,1)*1e9, rs(i,2)*1e9, rs(i,3), ...
            verdict_(rs(i,1) <= P.dl_rms_m && rs(i,3) >= P.strehl_min), Kk(i,:));
end
S.re_solved = rs;  S.conics = Kk;  S.decks = decks;

%% -- [3] the answer ----------------------------------------------------
pass = (rs(:,1) <= P.dl_rms_m) & (rs(:,3) >= P.strehl_min);
ilast = find(pass, 1, 'last');
lines = {};
lines{end+1} = sprintf(['ACHIEVABLE FIELD at %g nm, RMS <= %.2f nm and ' ...
                        'Strehl >= %.2f, rung "+LS tip/tilt":'], ...
                       LAM*1e9, P.dl_rms_m*1e9, P.strehl_min);
if isempty(ilast)
    lines{end+1} = '  NO candidate meets both bars -- the architecture is the wall.';
else
    hb = HALF_DEG(ilast);
    lines{end+1} = sprintf(['  half-field %.2f deg (FULL BOX %.2f deg) -- ' ...
                            '%.0f%% of the 0.6 deg asked for,'], ...
                           hb, 2*hb, 100*(2*hb)/0.6);
    lines{end+1} = sprintf('  at %.3f nm max / %.3f nm avg, Strehl %.4f.', ...
                           rs(ilast,1)*1e9, rs(ilast,2)*1e9, rs(ilast,3));
    if ilast < numel(HALF_DEG)
        lines{end+1} = sprintf(['  The next candidate, %.2f deg half-field, ' ...
                                'reads %.3f nm -- %.1fx the bar.'], ...
                               HALF_DEG(ilast+1), rs(ilast+1,1)*1e9, ...
                               rs(ilast+1,1)/P.dl_rms_m);
    end
end
% the growth law, measured rather than assumed
g = polyfit(log(HALF_DEG(:)), log(rs(:,1)), 1);
lines{end+1} = sprintf(['Measured growth of the re-solved residual: RMS ~ ' ...
                        'theta^%.2f (a pure Petzval/astig wall would be 2).'], g(1));
if ~isempty(S.as_solved)
    lines{end+1} = sprintf(['Re-solving is worth %.1fx at 0.30 deg half-field ' ...
                            '(%.1f nm as-solved -> %.1f nm re-solved).'], ...
                           S.as_solved(end,1)/rs(end,1), ...
                           S.as_solved(end,1)*1e9, rs(end,1)*1e9);
end
txt = sprintf('%s\n', lines{:});
fprintf('\n[3] %s', txt);

%% -- [4] artifacts ------------------------------------------------------
try
    f = figure('Visible','off','Position',[100 100 760 520]);
    hold on; box on; grid on;
    if ~isempty(S.as_solved)
        plot(HALF_DEG, S.as_solved(:,1)*1e9, 'o--', 'LineWidth',1.4, ...
             'DisplayName','as-solved (0.2 deg design)');
    end
    plot(HALF_DEG, rs(:,1)*1e9, 's-', 'LineWidth',1.8, ...
         'DisplayName','re-solved at each box');
    yline(P.dl_rms_m*1e9, 'k-', 'LineWidth',1.2, ...
          'DisplayName', sprintf('DL bar %.1f nm', P.dl_rms_m*1e9));
    set(gca,'YScale','log');
    xlabel('half-field  [deg]');
    ylabel(sprintf('max RMS WFE over the box  [nm]  (+LS tip/tilt, %g nm)', LAM*1e9));
    title('e2e2 s1: how much field does the axial TMA hold?');
    legend('Location','northwest');
    saveas(f, fullfile(exdir,'s1_fov_sweep.png'));  close(f);
    fprintf('    figure: s1_fov_sweep.png\n');
catch ME
    fprintf('    figure skipped (%s)\n', ME.message);
end
fid = fopen(fullfile(exdir,'s1_fov_sweep.txt'),'w');
fprintf(fid, ['e2e2 stage-1 FIELD SWEEP\n' ...
              'D %.3f m | f/%.4g | %g nm | DL bar %.2f nm RMS, Strehl %.2f\n\n'], ...
        P.D_m, P.system_fnum, LAM*1e9, P.dl_rms_m*1e9, P.strehl_min);
fprintf(fid, '%9s | %-32s | %-32s\n','half[deg]', ...
        'AS-SOLVED (0.2 deg design)','RE-SOLVED at that box');
fprintf(fid, '%9s | %10s %10s %10s | %10s %10s %10s\n','', ...
        'max nm','avg nm','Strehl','max nm','avg nm','Strehl');
for i = 1:numel(HALF_DEG)
    if isempty(S.as_solved), a = nan(1,3); else, a = S.as_solved(i,:); end
    fprintf(fid, '%9.2f | %10.3f %10.3f %10.4f | %10.3f %10.3f %10.4f\n', ...
            HALF_DEG(i), a(1)*1e9, a(2)*1e9, a(3), ...
            rs(i,1)*1e9, rs(i,2)*1e9, rs(i,3));
end
fprintf(fid, '\n%s', txt);
fclose(fid);
save(fullfile(exdir,'s1_fov_sweep.mat'),'S','P','HALF_DEG');
fprintf('    saved s1_fov_sweep.{txt,mat}\n');

function s = verdict_(c), if c, s = 'PASS'; else, s = 'over'; end, end
