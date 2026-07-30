function G = strict_rung_gates(map_n)
%STRICT_RUNG_GATES  The three acceptance gates for the strict metric.
%
%   G = STRICT_RUNG_GATES()      map_n = 9  (the packet's box sampling)
%   G = STRICT_RUNG_GATES(5)     coarser, for a quick pass
%
%   Gate 1  displaced-detector discriminator.  Stage 2, detector frozen by
%           Rodgers' procedure, then displaced +10 mm (and +627 mm) along
%           its own normal.  The strict metric must grow by the ANALYTIC
%           sphere-difference amount.  A flat response would mean the
%           metric is self-referencing.
%   Gate 2  on-axis anchor.  Stage 1 (no field bias) over its 0.2 deg box
%           under the strict metric, against Rodgers' on-axis field map
%           (0.446 / 1.463 / 0.606 nm min/max/avg).
%   Gate 3  the experiment.  Stage 2 (frozen verbatim conics, +0.5 deg
%           offset), EPD 4060, his-procedure detector frozen, strict metric
%           over the bias-relative 9x9 box, against 374.6 / 199.9 nm.
%
%   All at EPD = 4060 mm (Dave's measured aperture), lambda = 1000 nm.
%   Writes rodgers1_epd4060_strict_gates.mat next to this file.

    if nargin < 1, map_n = 9; end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common();  P.EPD_mm = 4060;
    lam_nm = P.lambda_m*1e9;
    G = struct('EPD_mm',P.EPD_mm,'lambda_nm',lam_nm,'map_n',map_n);

    Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');

    % ================================================================
    banner('GATE 1 -- displaced-detector discriminator (stage 2)');
    % ================================================================
    t2 = build_tma(P, P.K_nom, P.offset_deg);
    fp2 = t2.align_focal_plane('grid',5,'span_arcmin',6);
    nE  = numel(t2.spec.elt);
    V0  = t2.spec.elt(nE).Vpt(:);  N0 = t2.spec.elt(nE).psi(:); N0 = N0/norm(N0);
    fprintf('  detector frozen by align_focal_plane: tilt %.4f deg vs arriving chief,\n', fp2.tilt_deg);
    fprintf('  Vpt = [%.6f %.6f %.6f] m,  psi = [%.9f %.9f %.9f]\n', V0, N0);
    G.s2_fpa = fp2;

    G.g1 = gate1(t2, nE, V0, N0);

    % ================================================================
    banner('GATE 2 -- on-axis anchor (stage 1)');
    % ================================================================
    t1 = build_tma(P, P.K_nom, 0);
    fp1 = t1.align_focal_plane('grid',5,'span_arcmin',6);
    fprintf('  S1 FPA: tilt %.4f deg, focus z = %.3f mm\n', fp1.tilt_deg, fp1.fp_vpt(3)*1e3);
    s1 = strict_wfe(t1, Frel);
    G.s1 = s1;
    report('S1 on-axis box, STRICT', s1, P.gt.s1_onaxis_box, lam_nm);
    fprintf('  Rodgers on-axis map: min %.3f  max %.3f  avg %.3f nm\n', ...
            P.gt.s1_onaxis_box(1)*lam_nm, P.gt.s1_onaxis_box(2)*lam_nm, ...
            P.gt.s1_onaxis_box(3)*lam_nm);

    % ================================================================
    banner('GATE 3 -- the experiment: stage 2, frozen, bias-relative box');
    % ================================================================
    % NOTE the field convention: trace_at_field ADDS spec.field_bias, so the
    % box passed here is BOX-RELATIVE (unlike realize_apertures('fields',..),
    % which is the one branch that does NOT add the bias and therefore takes
    % the absolute box -- the §D `biasbox` helper).  Passing the absolute box
    % here would double the bias and evaluate a +1.0 deg box.
    fprintf('  field bias = %.4f deg; box is +/-%.2f deg about it\n', ...
            t2.spec.field_bias*180/pi, P.fov_half_deg);
    s2 = strict_wfe(t2, Frel);
    G.s2 = s2;
    report('S2 bias-rel box, STRICT', s2, P.gt.s2_box, lam_nm);
    fprintf('  Rodgers S2 box:      min %.3f  max %.3f  avg %.3f nm\n', ...
            P.gt.s2_box(1)*lam_nm, P.gt.s2_box(2)*lam_nm, P.gt.s2_box(3)*lam_nm);
    w = s2.wfe_m(isfinite(s2.wfe_m))*1e9;
    G.verdict = struct('max_ratio', max(w)/(P.gt.s2_box(2)*lam_nm), ...
                       'avg_ratio', mean(w)/(P.gt.s2_box(3)*lam_nm));
    fprintf('\n  VERDICT: max x%.3f   avg x%.3f   (gate: within ~1.5x)\n', ...
            G.verdict.max_ratio, G.verdict.avg_ratio);

    % ---- field maps (suffixed-parallel; nothing committed is overwritten)
    save_map(t1, s1, fullfile(here,'rodgers1_epd4060_stage1_strict.png'));
    save_map(t2, s2, fullfile(here,'rodgers1_epd4060_stage2_strict.png'));

    save(fullfile(here,'rodgers1_epd4060_strict_gates.mat'),'G');
    fprintf('\nsaved rodgers1_epd4060_strict_gates.mat + _strict.png maps\n');
end

% =====================================================================
function g = gate1(t, nE, V0, N0)
%GATE1  Harvest the box-centre field ONCE, then evaluate the strict metric
%   against detector planes displaced along the detector normal, and
%   compare the growth with the closed-form sphere difference.
    t.trace_at_field([0 0]);
    s  = macos.trace(nE);
    ri = macos.get_ray_info(s.nRays);
    ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
    P  = ri.pos(:,ok);  D = ri.dir(:,ok);  L = ri.opl(ok);
    p1 = ri.pos(:,1);   d1 = ri.dir(:,1);

    % exit pupil (FEX-style two-probe chief crossing)
    t.trace_at_field([1e-5 0]);  macos.trace(nE);
    rp = macos.get_ray_info(s.nRays);
    X  = fex_cross(p1, d1, rp.pos(:,1), rp.dir(:,1));
    t.trace_at_field([]);
    fprintf('  exit pupil (probe crossing) = [%.6f %.6f %.6f] m\n', X);

    cosg = abs(dot(N0, d1));                 % detector normal vs chief
    cosT = sum(D .* d1, 1).';                % ray obliquity about the chief
    fprintf('  detector-normal/chief angle = %.4f deg\n', acosd(cosg));
    fprintf('  ray half-cone               = %.4f deg  (f/%.2f)\n', ...
            acosd(min(cosT)), 1/(2*tan(acos(min(cosT)))));
    fprintf('  std(cos theta)              = %.6e   <- the defocus coefficient\n', std(cosT));

    DZ = [0 0.001 0.010 0.627];
    g  = struct('dz',DZ,'wfe_nm',nan(size(DZ)),'growth_nm',nan(size(DZ)), ...
                'predicted_nm',nan(size(DZ)),'ray_resid_nm',nan(size(DZ)), ...
                'xp',X,'std_cos',std(cosT),'tilt_deg',acosd(cosg));
    W0 = [];
    for j = 1:numel(DZ)
        Vd  = V0 + N0*DZ(j);
        c   = p1 + d1*(dot(N0, Vd - p1)/dot(N0, d1));
        R   = norm(X - c);
        W   = strict_sphere_opl(P, D, L, c, R);
        g.wfe_nm(j) = std(W)*1e9;
        if j == 1
            W0 = W;  R0 = R;  c0 = c;                                   %#ok<NASGU>
        else
            % --- closed-form sphere difference, per ray ---------------
            % moving the sphere centre from c0 to c (a shift of delta
            % along the chief) and the radius from R0 to R changes the
            % OPL to the sphere by, EXACTLY for a stigmatic bundle,
            %     delta*cos(theta) - sqrt(R^2 - delta^2 sin^2 theta) + R0
            delta = dot(c - c0, d1);
            pred  = delta*cosT - sqrt(R^2 - delta^2*(1-cosT.^2)) + R0;
            meas  = W - W0;
            g.growth_nm(j)    = std(meas)*1e9;
            g.predicted_nm(j) = std(pred)*1e9;
            g.ray_resid_nm(j) = std(meas - pred)*1e9;
            fprintf(['  dz = %+8.1f mm : strict %12.4g nm   growth %12.4g nm' ...
                     '   ANALYTIC %12.4g nm   per-ray resid %10.4g nm\n'], ...
                    DZ(j)*1e3, g.wfe_nm(j), g.growth_nm(j), ...
                    g.predicted_nm(j), g.ray_resid_nm(j));
        end
    end
    fprintf('  dz =      0.0 mm : strict %12.4g nm  (baseline)\n', g.wfe_nm(1));
    fprintf('  engine FP rmsWFE at dz=0    = %.6e m  = %.4g nm\n', ...
            s.rmsWFE, s.rmsWFE*1e9);
end

function X = fex_cross(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function save_map(t, s, path)
%SAVE_MAP  Render a strict-metric field map through the standard viewer.
%   view_field_map wants .fields in ARCMIN and .wfe in WAVES.
    scan = struct('fields', s.fields*180/pi*60, 'wfe', s.wfe(:), ...
                  'metric', 'strict (chief-tied sphere on the frozen detector, piston only)');
    fig = t.view_field_map(scan, 'kind','contour', 'save',path, 'visible',false);
    close(fig);
end

function report(tag, s, gt, lam_nm)
    w = s.wfe_m(isfinite(s.wfe_m))*1e9;
    fprintf('  %-26s min %9.3f  max %9.3f  avg %9.3f nm   (%d/%d fields)\n', ...
            tag, min(w), max(w), mean(w), numel(w), numel(s.wfe_m));
    e = s.eng_rms_m(isfinite(s.eng_rms_m))*1e9;
    fprintf('  %-26s min %9.3f  max %9.3f  avg %9.3f nm   [engine FP rmsWFE, for contrast]\n', ...
            '', min(e), max(e), mean(e));
end

function t = build_tma(P, K, bias_deg)
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',K(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',K(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',K(3),'spacing_after','derive');
    if bias_deg ~= 0, t.set_field_bias(bias_deg*60); end
    t.build();
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
