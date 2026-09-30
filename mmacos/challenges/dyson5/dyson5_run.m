function OUT = dyson5_run(over)
%DYSON5_RUN  The dyson5 challenge runner: one user-editable entry point.
%
%   OUT = DYSON5_RUN() runs every stage in P.stages at the default spec
%   (dyson5_params).  OUT = DYSON5_RUN(OVER) applies overrides first.
%   Every number in the challenge record is produced THROUGH this runner.
%
%   Stages (the ladder grows beat by beat; each stage names its convention
%   before it prints a number):
%     s0  SCALING LAW of the classical concentric Dyson at the spec: the
%         Dyson condition R_g = n r/(n-1) verified by exact trace, then
%         the block radius the spec's slit corner needs for a blur budget
%         of P.blur_px pixels -- the pre-registered question "can a single
%         block meet F/1.8 at a 54 mm slit?" answered before any element
%         is added.  Engine-free (pure geometry).
%     s1  deck emission (Dyson + Offner) -- beat 2
%     s2  spectrometer_score: field-angle + wavelength maps, smile,
%         keystone, SRF/XRF, radiometric chain -- beat 2
%     s3  native optimize -- beat 3
%
%   Artifacts (P.outdir): <tag>_s0_scaling.txt, <tag>_s0_scaling.mat,
%   <tag>_s0_scaling.png.
%
%   See also DYSON5_PARAMS, dyson_layout, dyson_scaling.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here, '..', '..', 'mmacos_setup.m'));
    addpath(here);
    P = dyson5_params(over);
    if isempty(P.outdir), P.outdir = here; end
    if ~exist(P.outdir, 'dir'), mkdir(P.outdir); end
    tag = fullfile(P.outdir, P.tag);
    OUT.P = P;

    for k = 1:numel(P.stages)
        switch P.stages{k}
            case 's0', OUT.s0 = stage_s0_(P, tag);
            case {'s1','s2','s3'}
                fprintf('dyson5_run: stage %s is queued for the next beat (BRIEF_to_dyson5 build order).\n', P.stages{k});
            otherwise
                error('dyson5_run:stage', 'unknown stage %s', P.stages{k});
        end
    end
end

% =====================================================================
function S = stage_s0_(P, tag)
%STAGE_S0_  The concentric-Dyson scaling law at the spec.
    n = sellmeier_(P.glass, P.lambda_ref_m);
    slit_len = P.npix(1)*P.pixel_m;
    fpa_h    = P.npix(2)*P.pixel_m;
    h_max    = hypot(slit_len/2, P.y_offset_m);       % slit CORNER off the axis
    blur_max = P.blur_px*P.pixel_m;

    % (1) the condition, verified at a mid-size block
    D = dyson_layout(0.1, n, 'fno', P.Fno);
    % (2) the sweep: blur at the slit corner vs block radius
    Ssc = dyson_scaling(n, P.Fno, h_max, blur_max, 'r_grid', P.r_grid_m);

    S.n = n;  S.h_max = h_max;  S.blur_max = blur_max;  S.slit_len = slit_len;
    S.fpa_h = fpa_h;  S.spectral_sampling_m = diff(P.band_m)/P.npix(2);
    S.condition = D;  S.scaling = Ssc;

    % -- the record ------------------------------------------------------
    fid = fopen([tag '_s0_scaling.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});   % console + record
    pr('dyson5 s0 -- classical concentric Dyson at the spec (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: block index n(%s) at %.2f um = %.6f; F/# = air-equivalent image-space,\n', P.glass, P.lambda_ref_m*1e6, n);
    pr('  in-glass marginal half-angle u = asin(1/(2 n F)) = %.3f deg; blur = transverse rms spot of a\n', D.u_glass*180/pi);
    pr('  full F/%.1f cone from a point on the flat face, imaged back to the flat face (1:1), grating\n', P.Fno);
    pr('  traced as a mirror (order 0); field height h measured from the Dyson axis on the flat face.\n');
    pr('SPEC: slit %.1f mm (%d px x %.0f um), FPA %.1f mm spectral (%d px), band %.0f-%.0f nm -> %.2f nm/px;\n', ...
        slit_len*1e3, P.npix(1), P.pixel_m*1e6, fpa_h*1e3, P.npix(2), P.band_m*1e9, S.spectral_sampling_m*1e9);
    pr('  slit centre offset %.1f mm -> slit CORNER at h_max = %.2f mm; blur budget %.2f px = %.1f um rms.\n', ...
        P.y_offset_m*1e3, h_max*1e3, P.blur_px, blur_max*1e6);
    pr('\n(1) DYSON CONDITION, r = 100 mm: R_g = n r/(n-1) = %.3f mm, gap %.3f mm, D_g(axial) = %.1f mm\n', ...
        D.R_g*1e3, D.gap*1e3, D.D_g_axial*1e3);
    pr('    blur (um rms) vs R_g factor: ');  pr('%.2f:%.2f  ', [D.sweep.factor; D.sweep.blur*1e6]);  pr('\n');
    pr('    blur (um rms) vs h (mm):     ');  pr('%.0f:%.2f  ', [D.h*1e3; D.blur_rms*1e6]);
    pr('  -> h-exponent %.2f (fifth-order residual)\n', D.h_exponent);
    pr('    centroid shift image_y + h (um) vs h (mm): ');  pr('%.0f:%.3f  ', [D.h*1e3; D.distortion*1e6]);  pr('  (distortion, same order)\n');
    pr('\n(2) SCALING at the slit corner h_max = %.2f mm:\n', h_max*1e3);
    pr('    r (mm):        ');  pr('%8.0f', Ssc.r_grid*1e3);  pr('\n');
    pr('    blur (um rms): ');  pr('%8.2f', Ssc.blur*1e6);    pr('\n');
    pr('    fitted r-exponent %.2f (law: blur ~ h^4 / r^3)\n', Ssc.law_exponent);
    if isnan(Ssc.r_required)
        pr('    -> NO block on the grid meets %.1f um at the corner.\n', blur_max*1e6);
    else
        pr('    -> r_required = %.1f mm  (R_g = %.1f mm, gap %.1f mm, grating clear diam >= %.1f mm axial + field)\n', ...
            Ssc.r_required*1e3, Ssc.R_g_required*1e3, (Ssc.R_g_required-Ssc.r_required)*1e3, ...
            2*Ssc.R_g_required*sin(D.u_glass)*1e3);
    end
    pr('VERDICT: the single-block concentric Dyson meets the corner blur only at r >= %.0f mm -- report the law,\n', Ssc.r_required*1e3);
    pr('  do not add elements yet (BRIEF_to_dyson5 build order, pre-registered null).\n');
    fclose(fid);

    % -- figure ----------------------------------------------------------
    f = figure('Visible', 'off', 'Position', [100 100 640 420]);
    loglog(Ssc.r_grid*1e3, Ssc.blur*1e6, 'o-', 'LineWidth', 1.5);  hold on
    yline(blur_max*1e6, '--', sprintf('%.2f px budget', P.blur_px));
    if ~isnan(Ssc.r_required), xline(Ssc.r_required*1e3, ':', sprintf('r = %.0f mm', Ssc.r_required*1e3)); end
    grid on;  xlabel('block radius r (mm)');  ylabel('rms blur at the slit corner (um)');
    title(sprintf('Concentric Dyson, F/%.1f, %s, slit corner h = %.1f mm: blur ~ r^{%.2f}', ...
        P.Fno, P.glass, h_max*1e3, Ssc.law_exponent));
    print(f, [tag '_s0_scaling.png'], '-dpng', '-r110');  close(f);
    save([tag '_s0_scaling.mat'], 'S', 'P');
    fprintf('dyson5 s0: wrote %s_s0_scaling.{txt,mat,png}\n', tag);
end

function dualprint_(fid, varargin)
%DUALPRINT_  fprintf to the console and the record file (a plain function:
%   fprintf inside cellfun's uniform-output context errors).
    fprintf(1, varargin{:});  fprintf(fid, varargin{:});
end

function n = sellmeier_(glass, lam_m)
%SELLMEIER_  Index for the layout, from the engine table rows (um^2 C's).
%   The engine applies the same rows at trace time (gate tGlassDispersion).
    switch glass
        case 'Silica', B = [0.6961663 0.4079426 0.8974794];  C = [0.004679148 0.01351206 97.934];
        case 'CaF2',   B = [0.5675888 0.4710914 3.8484723];  C = [0.050263605^2 0.1003909^2 34.649040^2];
        otherwise, error('dyson5_run:glass', 'no Sellmeier row for %s in this runner', glass);
    end
    L2 = (lam_m*1e6)^2;
    n = sqrt(1 + sum(B .* L2 ./ (L2 - C)));
end
