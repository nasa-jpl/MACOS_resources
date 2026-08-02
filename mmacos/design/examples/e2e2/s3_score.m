% s3_score.m  (mmacos/design/examples/e2e2/ -- stage 3 of 3, FINAL)
% =====================================================================
%  E2E2 STAGE 3 -- FINAL SCORING AND DOCUMENTATION
% =====================================================================
%  The telescope closes out at stage 2 (Dave 2026-08-02: "close out the
%  telescope flow at S2, relay separately").  This stage does not design
%  anything.  It scores the delivered design properly and writes down
%  what it is, which is the part a reader actually needs.
%
%  FOUR THINGS, and the order matters:
%
%    [1] THE LADDER, on a DENSER grid than the solve stages used.  All
%        four references, every time, because a single number is not a
%        result unless its convention is named -- the four spread by 2.4x
%        on this design and reporting only the friendliest would be a
%        choice disguised as a measurement.
%
%    [2] COMA, as the centroid-minus-chief displacement map.  This is
%        what makes the four references disagree in the first place: a
%        sphere on the chief leaves a tilt wherever the chief is not the
%        centroid, and that is exactly where there is coma.  Mapping it
%        explains the ladder instead of just tabulating it.
%
%    [3] DISTORTION, the centroid grid against ideal f.theta.  Reported
%        ALONGSIDE the wavefront, never inside it: distortion is chief-ray
%        MAPPING error, no detector pose corrects it, and none of the DOFs
%        in this flow target it.  It calibrates; blur does not.
%
%    [4] PARAMETER PROVENANCE across all stages.  The parameter table IS
%        the design -- WFE only scores it, and WFE is reproducible from
%        the parameters while the parameters are not reproducible from it.
%
%  Then the handoff: the emitted .in feeds the EXISTING segmentation ->
%  sensitivities -> MET -> compare -> simulator pipeline unchanged.
%
%    >> run('.../design/examples/e2e2/s3_score.m')
% =====================================================================
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
mmroot = fileparts(fileparts(fileparts(exdir)));
run(fullfile(mmroot,'mmacos_setup.m'));
addpath(exdir);

P   = e2e2_params();
LAM = P.lambda_m;   D = P.D_m;   h = P.fov_half_deg;
S1  = load(fullfile(exdir,'s1_axial.mat'));
S2  = load(fullfile(exdir,'s2_fold.mat'));
deck = fullfile(exdir,'s2_fold.in');
assert(exist(deck,'file')==2, 'e2e2:s3:deck', 's2_fold.in not built');

fprintf('\n====================================================================\n');
fprintf(' E2E2 stage 3: FINAL SCORE | D=%.2f m f/%.4g | %g nm | %g deg box\n', ...
        D, P.system_fnum, LAM*1e9, 2*h);
fprintf('====================================================================\n');

macos.init(P.model_size);

%% -- [1] the ladder, on a DENSE uniform grid ---------------------------
% Denser than the solve stages scored on: the solve set had 8 points and
% the stage verdicts used 9x9, so a finer grid is the check that those
% numbers were sampling-limited, not luck.  Statistics NEVER come from
% the solve set (an edge-weighted sampling biases the average ~8% at an
% identical max -- rodgers1 dense_field_check).
Fsc = macos.design.field_grid(P.fov_arcmin, P.final_n, 'units','arcmin');
fprintf('\n[1] reference ladder on a UNIFORM %dx%d grid (%d points)\n', ...
        P.final_n, P.final_n, size(Fsc,1));
[L, info] = strict_ladder_deck(deck, Fsc, 'lambda', LAM);
ok = all(isfinite(L),2);
nm = {'strict-chief','strict-centroid','+best focus','+LS tip/tilt'};
fprintf('    %-16s %10s %10s %11s %11s %10s\n', ...
        'rung','max nm','avg nm','max waves','min Strehl','vs bar');
pass = false(1,4);
for r = 1:4
    mx = max(L(ok,r));   av = mean(L(ok,r));   st = min(info.strehl(ok,r));
    pass(r) = (mx <= P.dl_rms_m) && (st >= P.strehl_min);
    fprintf('    %-16s %10.3f %10.3f %11.4f %11.4f %10s%s\n', nm{r}, ...
            mx*1e9, av*1e9, mx/LAM, st, tern_(pass(r),'PASS','FAIL'), ...
            tern_(r==P.score_rung,'   <- verdict',''));
end
fprintf(['    bar: %.2f nm RMS (lambda/%g) and Strehl %.2f\n'], ...
        P.dl_rms_m*1e9, 1/P.dl_waves, P.strehl_min);
spread = max(L(ok,1))/max(L(ok,4));
fprintf(['    the four references spread by %.2fx on this design -- which is why ' ...
         'every\n    number in this flow names its rung\n'], spread);

%% -- [2] coma: the centroid-minus-chief displacement map ---------------
% The quantity that MAKES the references disagree.  strict_wfe_deck
% returns it per field alongside both WFEs.
fprintf('\n[2] coma tracker: centroid-minus-chief displacement on the detector\n');
sw = strict_wfe_deck(deck, Fsc, 'reference','strict-centroid');
dc = sw.dcen_m(:)*1e6;                       % microns
good = isfinite(dc);
fprintf('    min %.3f  max %.3f  mean %.3f um over %d fields\n', ...
        min(dc(good)), max(dc(good)), mean(dc(good)), nnz(good));
fprintf(['    (on axis this is ~0; it grows linearly with field where the ' ...
         'aberration is\n     coma, and it is the tilt the chief-referenced ' ...
         'sphere carries and the\n     centroid-referenced one does not)\n']);

%% -- [3] distortion: the centroid grid vs ideal f.theta ----------------
% Three readings, most literal first: (a) against the ideal f.theta grid
% with the scale HELD at the traced EFL, only the detector's arbitrary
% placement and clocking removed; (b) after also fitting a uniform scale
% -- the fitted scale IS the local magnification, which off axis is not
% f; (c) after a full affine fit, i.e. what no linear map can absorb.
fprintf('\n[3] distortion: centroid grid vs ideal f.theta\n');
% THE CENTROID'S POSITION ON THE DETECTOR, not its offset from the chief.
% strict_wfe_deck returns both, and they differ by five orders of
% magnitude: .dcen_2d is centroid-MINUS-CHIEF (the coma tracker of [2],
% a few um) while distortion needs where the image actually LANDS
% (+-0.2 m here).  Using the former fits a scale of 8e-6 and reports 0.3 m
% of "distortion" that is nothing but the wrong quantity.
Vd = sw.detector.Vpt(:);   Nd = sw.detector.psi(:);   Nd = Nd/norm(Nd);
f1 = [1;0;0] - Nd*dot([1;0;0],Nd);
if norm(f1) < 1e-8, f1 = [0;1;0] - Nd*dot([0;1;0],Nd); end
f1 = f1/norm(f1);   f2 = cross(Nd, f1);
Q   = sw.c_centroid;                          % 3 x K, global
cok = all(isfinite(Q),1).' & good;
th  = Fsc(cok,:);                             % field offsets, rad
obs = [ (f1.'*(Q(:,cok) - Vd)).', (f2.'*(Q(:,cok) - Vd)).' ];   % K x 2, m
EFLt = S2.rpt.EFL_m;
ideal = EFLt * tan(th);                       % ideal f.theta, m
[da, db, dcn, scl, rot, par] = distortion_(obs, ideal);
fprintf(['    traced EFL %.3f m | frame rotation %+.2f deg | parity %+d ' ...
         '(%s)\n'], EFLt, rot, round(par), ...
        tern_(par < 0, 'INVERTED -- odd powered reflections, as it must be', ...
              'upright'));
fprintf('    (a) vs f.theta, scale held : max %8.1f  rms %8.1f um\n', da*1e6);
fprintf('    (b) after a uniform scale  : max %8.1f  rms %8.1f um   (local mag %+.4f%%)\n', ...
        db*1e6, (abs(scl)-1)*100);
fprintf('    (c) after a full affine    : max %8.1f  rms %8.1f um   <- genuinely nonlinear\n', ...
        dcn*1e6);
fprintf(['    reported ALONGSIDE the wavefront, never inside it: this is ' ...
         'chief-ray mapping\n    error, no detector pose corrects it, and no ' ...
         'DOF in this flow targets it\n']);

%% -- [4] parameter provenance across the whole flow --------------------
fprintf('\n[4] parameter provenance\n');
t = macos.design.Telescope.load_spec(fullfile(exdir,'s2_fold.mat'));
t.build('', 'init', false);
pt = param_table(t, 'prev', S1.pt, ...
     'title', 'E2E2 FINAL -- PARAMETER PROVENANCE (delta vs the axial start)', ...
     'held', {'R1,R2,R3 + spacings (first-order layout)', ...
              sprintf('field bias %g'' (stage 2 clearance frontier)', S2.BIAS), ...
              sprintf('M1 hole %.4f m (measured, floored at the M2 shadow)', S2.r_hole)});

%% -- [5] views + maps ---------------------------------------------------
try
    fv = macos.view_std('args', {'show','beam'}, 'visible', false, ...
            'title', sprintf('e2e2: D=%.1f m f/%.0f folded Korsch TMA, %g deg box', ...
                             D, P.system_fnum, 2*h), ...
            'save', fullfile(exdir,'s3_views.png'));
    close(fv);  fprintf('\n[5] standard views: s3_views.png\n');
catch ME, fprintf('\n[5] view_std skipped (%s)\n', ME.message); end
try
    f1 = figure('Visible','off','Position',[80 80 1180 420]);
    ax = Fsc(:,1)*180*60/pi;  ay = Fsc(:,2)*180*60/pi;
    n  = P.final_n;
    subplot(1,3,1);
    contourf(reshape(ax,n,n), reshape(ay,n,n), ...
             reshape(L(:,P.score_rung)*1e9,n,n), 14, 'LineColor','none');
    axis equal tight; colormap(parula); cb=colorbar; cb.Label.String='RMS WFE [nm]';
    xlabel('\theta_x [arcmin]'); ylabel('\theta_y [arcmin]');
    title(sprintf('%s', nm{P.score_rung}));
    subplot(1,3,2);
    contourf(reshape(ax,n,n), reshape(ay,n,n), reshape(dc,n,n), 14, 'LineColor','none');
    axis equal tight; cb=colorbar; cb.Label.String='centroid - chief [\mum]';
    xlabel('\theta_x [arcmin]'); title('coma tracker');
    subplot(1,3,3);
    rd = nan(size(Fsc,1),1);
    rr = sqrt(sum((obs - ideal).^2,2))*1e6;
    rd(cok) = rr;
    contourf(reshape(ax,n,n), reshape(ay,n,n), reshape(rd,n,n), 14, 'LineColor','none');
    axis equal tight; cb=colorbar; cb.Label.String='f\cdot\theta departure [\mum]';
    xlabel('\theta_x [arcmin]'); title('distortion (scale held)');
    saveas(f1, fullfile(exdir,'s3_maps.png'));  close(f1);
    fprintf('    field maps: s3_maps.png\n');
catch ME, fprintf('    field maps skipped (%s)\n', ME.message); end

%% -- [6] the report -----------------------------------------------------
rpt = design_report(t, 'rings_arcmin', [P.fov_arcmin/4, P.fov_arcmin/2, ...
                    P.fov_arcmin], 'dl_waves', P.dl_waves);
lad = sprintf('%-16s %10.3f %10.3f %11.4f %10s\n', ...
      'rung','max nm','avg nm','minStrehl','verdict');
for r = 1:4
    lad = [lad sprintf('%-16s %10.3f %10.3f %11.4f %10s\n', nm{r}, ...
           max(L(ok,r))*1e9, mean(L(ok,r))*1e9, min(info.strehl(ok,r)), ...
           tern_(pass(r),'PASS','FAIL'))]; %#ok<AGROW>
end
add = { ...
 '==================== E2E2 FINAL ===================='
 sprintf('  D %.3f m | f/%.4g | %g nm | %g x %g deg used box | bias %g''', ...
         D, P.system_fnum, LAM*1e9, 2*h, 2*h, S2.BIAS)
 sprintf('  traced EFL %.3f m (f/%.3f) | M1 hole %.4f m (%.2f%% of the area)', ...
         rpt.EFL_m, rpt.fno_fp, S2.r_hole, 100*(2*S2.r_hole/D)^2)
 ''
 sprintf('  LADDER on a uniform %dx%d grid (%d points), bar %.2f nm / Strehl %.2f', ...
         P.final_n, P.final_n, nnz(ok), P.dl_rms_m*1e9, P.strehl_min)
 lad
 sprintf('  the four references spread %.2fx -- every number names its rung', spread)
 ''
 sprintf('  coma  : centroid-chief %.3f .. %.3f um (mean %.3f)', ...
         min(dc(good)), max(dc(good)), mean(dc(good)))
 sprintf('  distortion vs f.theta : %.1f um rms held / %.1f uniform-scaled / %.1f nonlinear', ...
         da(2)*1e6, db(2)*1e6, dcn(2)*1e6)
 sprintf('           local magnification %+.4f%% of the traced EFL (parity %+d)', ...
         (abs(scl)-1)*100, round(par))
 '           reported alongside the wavefront, never inside it'
 ''
 '  HOW IT GOT HERE'
 sprintf('    S1 axial        %8.3f nm  (%s)', ...
         S1.sc.uniform.max_m(P.score_rung)*1e9, 'conics + FPA, joint')
 sprintf('    S2 folded       %8.3f nm  (fold + %g'' bias on the clearance frontier)', ...
         S2.sc.uniform.max_m(P.score_rung)*1e9, S2.BIAS)
 '    relay           parked as follow-on: the telescope does not need it'
 '                    (relay_followon/README.md -- two failed attempts recorded)'
 ''
 '  HANDOFF: this .in feeds the EXISTING pipeline unchanged --'
 '    run_segmentation -> run_sensitivities -> run_met -> run_compare -> run_simulator'
 '    (design/runners/; worked end-to-end in ../e2e/).  Not duplicated here.'
 '===================================================='};
addtxt = sprintf('%s\n', add{:});
fprintf('\n[6] %s', addtxt);
fid = fopen(fullfile(exdir,'s3_report.txt'),'w');
fprintf(fid, '%s%s%s', rpt.text, pt.text, addtxt);   fclose(fid);
save(fullfile(exdir,'s3_score.mat'), 'P','L','info','Fsc','sw','dc', ...
     'da','db','dcn','scl','rot','par','pt','rpt','pass','spread');
fprintf('    report: s3_report.txt   artifacts: s3_score.mat, s3_maps.png\n');
fprintf('\nE2E2 COMPLETE.  Telescope closed out at stage 2; relay is follow-on.\n');

% ---- helpers --------------------------------------------------------
function [da, db, dc, scl, rot, par] = distortion_(obs, ideal)
%DISTORTION_  Centroid grid against ideal f.theta, three readings.
%   (a) scale HELD: remove only the detector's arbitrary offset and
%       clocking -- an ORTHOGONAL fit, reflections ALLOWED.
%   (b) + a uniform scale: the fitted scale IS the local magnification.
%   (c) + a full affine: what no linear map absorbs = genuinely
%       nonlinear distortion.  Each returned as [max rms], metres.
%
%   REFLECTIONS MUST BE ALLOWED.  An imaging train with an ODD number of
%   powered reflections inverts the image, and this one also folds the
%   beam into +x, so the detector frame is both inverted and rotated 90
%   deg from the field frame.  Forcing a PROPER rotation (det = +1) on
%   that mapping cannot fit it: reading (a) came back with 0.3 m of
%   "distortion" and a fitted scale of -100.0003%, which is not
%   distortion at all -- it is the parity, misread.  So the orthogonal
%   fit here keeps whatever determinant the data wants, and the parity is
%   REPORTED rather than silently absorbed.
    o = obs - mean(obs,1);   q = ideal - mean(ideal,1);
    [U,~,V] = svd(q.'*o);
    R   = V*U.';                       % orthogonal, reflection allowed
    par = det(R);                      % +1 proper, -1 = inverted image
    ra = o - q*R.';
    da = [max(vecnorm(ra,2,2)), sqrt(mean(sum(ra.^2,2)))];
    % atan2d(R(2,1),R(1,1)) is a ROTATION angle only for a proper R.  When
    % the fit comes back improper (det < 0, the odd-reflection case) that
    % expression is the angle of the reflection AXIS doubled, not a
    % rotation, so report it against the proper part and let `par` carry
    % the parity.  (Fable review, 2026-08-02.)
    Rp = R;   if par < 0, Rp = R*diag([1 -1]); end
    rot = atan2d(Rp(2,1), Rp(1,1));
    % + uniform scale.  Minimising ||o - s*q*R'|| gives
    %   s = trace(o'*q*R') / trace(q'*q),  and trace(o'*q*R') = trace(R*q'*o),
    % i.e. R and NOT R'.  Getting that transpose wrong makes (b) come back
    % LARGER than (a) -- which is impossible, since (b) has strictly more
    % freedom, and is the check that caught it.
    scl = trace(R*(q.'*o)) / max(trace(q.'*q), eps);
    rb = o - scl*(q*R.');
    db = [max(vecnorm(rb,2,2)), sqrt(mean(sum(rb.^2,2)))];
    assert(db(2) <= da(2)*(1+1e-9), ...
        ['distortion_: the uniform-scale fit (%g) is worse than the ' ...
         'scale-held fit (%g), which is impossible -- it has strictly more ' ...
         'freedom.  The scale formula is wrong.'], db(2), da(2));
    % + full affine
    A  = [q, ones(size(q,1),1)] \ o;
    rc = o - [q, ones(size(q,1),1)]*A;
    dc = [max(vecnorm(rc,2,2)), sqrt(mean(sum(rc.^2,2)))];
    % Same impossibility check as (b) vs (a), one rung further: a full
    % affine has strictly more freedom than scale+rotation, so it cannot
    % fit worse.  (Fable review, 2026-08-02.)
    assert(dc(2) <= db(2)*(1+1e-9), ...
        ['distortion_: the affine fit (%g) is worse than the ' ...
         'uniform-scale fit (%g), which is impossible -- it has strictly ' ...
         'more freedom.'], dc(2), db(2));
end

function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
