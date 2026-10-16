function OUT = dyson5_tma_step1(over)
%DYSON5_TMA_STEP1  Round 4 step 1: the coaxial three-mirror anastigmat parent.
%
%   OUT = DYSON5_TMA_STEP1() builds the f=330 mm, F/1.8 coaxial TMA (Jim's
%   numbers: 30 m GSD at ~550 km, 18 um pixels -> D=183 mm) with the design
%   layer (macos.design.tma_layout + Telescope, the tma_onaxis idiom), the stop
%   on the SECONDARY, and solves the CONICS with the native optimizer (CALIB)
%   AT a field bias (scanned 3-6 deg ALONG-track), over the 9.4 deg CROSS-track
%   strip as the field set (the 3k module: +-4.7 deg -> +-27 mm = the 54 mm
%   slit length).  Conic-only DOFs preserve the first-order layout, so the EFL
%   stays 330 mm (freeing ROC would drift f/# off 1.8).  Each bias is engine-
%   scored on 7 strip fields: rms spot radius (um) and energy in an 18 um pixel.
%   The least bias that images best is rendered (view_orthoviews).
%
%   OUT = DYSON5_TMA_STEP1(OVER) overrides: .biases_deg, .primary_fnum,
%   .secondary_mag, .strip_half_deg (4.7 = 3k; 2.35 = 1.5k), .nfield,
%   .max_iters, .areal_density (kg/m^2 for the mirror-mass estimate), .quiet.
%
%   Records dyson5_tma_step1.{txt,mat} and dyson5_tma_step1_layout.png.
%   NEW FILE (round 4); TO's tms_*/t4 files are theirs.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    opt = struct('EFL_m', 0.330, 'Fsys', 1.8, 'primary_fnum', 1.0, 'secondary_mag', 3.5, ...
                 'int_focus_D', -0.125, 'm3_behind_D', 0.6, 'lam', 633e-9, 'model', 256, ...
                 'biases_deg', [1 2 3 4 5 6], 'strip_half_deg', 4.7, 'nfield', 7, ...
                 'pixel_m', 18e-6, 'max_iters', 150, 'areal_density', 25, 'quiet', false, 'tag', 'dyson5_tma_step1');
    fn = fieldnames(over);
    for k = 1:numel(fn), if ~isfield(opt, fn{k}), error('unknown option %s', fn{k}); end, opt.(fn{k}) = over.(fn{k}); end
    D = opt.EFL_m/opt.Fsys;

    addpath('/Users/dcr/dev/MACOS_resources/mmacos/src');  macos.init(opt.model);
    fid = fopen(fullfile(here, [opt.tag '.txt']), 'w');  pr = @(varargin) dp_(fid, opt.quiet, varargin{:});
    pr('dyson5 round 4 step 1 -- coaxial TMA parent (%s)\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('f=%.0f mm, F/%.1f, D=%.1f mm; primary f/%.2f, secondary mag %.2f; conics solved AT the bias (CALIB); entrance stop\n', ...
       opt.EFL_m*1e3, opt.Fsys, D*1e3, opt.primary_fnum, opt.secondary_mag);
    pr('  (engine stop_info_set rejects an M2 stop on the Telescope deck -- same D beam; the M2/exit-pupil stop is step 3).\n');
    pr('  conic-only DOFs (EFL held at 330 mm); field set = the 9.4 deg cross-track strip (+-%.1f deg -> the 54 mm slit).\n', opt.strip_half_deg);
    pr('  spot = transverse rms radius about the centroid (perp to the chief) at the FP, um; EE = energy in an 18 um pixel.\n\n');

    % ---- first-order Korsch layout, CALIBRATED so the EXACT-traced EFL = 330 mm
    % (beat5's caution: seidel_seed's n-flip paraxial EFL is unreliable for a convex
    % PNP reimager -- the K=0 seed built from the layout traces to a different focus,
    % so the layout's requested system f/# is NOT the as-built f/#.  Iterate the
    % requested system f/# until the exact trace delivers EFL = 330 mm at D = 183 mm.)
    fsys_req = opt.Fsys;  R = []; t = []; lay = [];
    pr('[layout calibration] holding D=%.1f mm, primary f/%.2f, secondary mag %.2f; EFL by exact trace:\n', D*1e3, opt.primary_fnum, opt.secondary_mag);
    for it = 1:6
        [R, t, lay] = macos.design.tma_layout(D, opt.primary_fnum, fsys_req, 'secondary_mag', opt.secondary_mag, ...
                          'int_focus_m', opt.int_focus_D*D, 'm3_behind_m', opt.m3_behind_D*D);
        efl_trace = efl_by_trace_(R, t, D, opt.lam, opt.model);
        pr('  iter %d: requested f/%.3f (EFL_req %.4f) -> exact EFL %.4f m (F/%.3f)\n', it, fsys_req, lay.EFL, efl_trace, efl_trace/D);
        if abs(efl_trace - opt.EFL_m)/opt.EFL_m < 0.004, break; end
        fsys_req = fsys_req * (opt.EFL_m / efl_trace);
    end
    pr('[layout] R=[%.4f %.4f %.4f] m  t=[%.4f %.4f] m  exact EFL %.4f m, F/%.2f at D=%.1f mm\n\n', ...
       R(1),R(2),R(3), t(1),t(2), efl_trace, efl_trace/D, D*1e3);

    fx = linspace(-opt.strip_half_deg, opt.strip_half_deg, opt.nfield) * pi/180;   % cross-track field offsets (rad), about the bias
    rows = struct('bias_deg',{},'spot_um',{},'spot_max_um',{},'nsolved',{},'ee_min',{}, ...
                  'length_mm',{},'M2_mm',{},'M3_mm',{},'mass_kg',{},'efl_mm',{},'conics',{},'tel',{});
    pr('%6s  %-55s %8s %7s %7s %7s %7s %7s %7s\n','bias','rms spot per field across the strip, um (NaN=field not imaged)','maxspot','EEmin','len mm','M2 mm','M3 mm','mass kg','EFL mm');
    for b = opt.biases_deg
        try
            tel = build_tma_(R, t, D, opt.lam, opt.model);
            tel.set_field_bias(b*60);                      % along-track bias, ARCMIN
            % Solve the conics AT THE BIAS FIELD ONLY (the annular-field-anastigmat move,
            % project_fold_extraction): nulls 3rd-order spherical+coma+astig at the biased
            % chief.  Optimizing over the full +-4.7 deg strip loses rays in CALIB's trial
            % steps (runaway).  The strip edges' residual is what step 1 reports; step 4
            % adds freeform if the conics stall.
            tel.optimize('fields_arcmin', [], 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', opt.max_iters);
            nE = numel(tel.spec.elt);
            % NOTE: macos.stop(2) (stop on M2) is REJECTED by the engine's stop_info_set on a
            % Telescope-emitted deck (the entrance aperture is the emitted stop); telescope_score
            % succeeds only on a spectrometer_rx-style deck.  The parent spot is the same D beam
            % either way, so step 1 scores with the emitted entrance stop; the M2/exit-pupil stop
            % is step 3's (flagged for CC as a Telescope-veneer vs engine stop_info_set item).
            [spot, ee] = spot_per_field_(tel, nE, fx, opt.pixel_m);
            g = geom_(tel, nE, fx, opt.areal_density, D);
            ef = efl_trace;                                % conic-only solve does not change the paraxial EFL
            row = struct('bias_deg',b,'spot_um',spot,'spot_max_um',max(spot,[],'omitnan'),'nsolved',sum(~isnan(spot)),'ee_min',min(ee,[],'omitnan'), ...
                         'length_mm',g.len_mm,'M2_mm',g.M2_mm,'M3_mm',g.M3_mm, ...
                         'mass_kg',g.mass_kg,'efl_mm',ef*1e3,'conics',[tel.spec.elt(1).Kc tel.spec.elt(2).Kc tel.spec.elt(3).Kc],'tel',tel);
            rows(end+1) = row;   %#ok<AGROW>
            pr('%5.0f   %-55s %8.2f %8.3f %7.1f %7.1f %7.1f %7.2f %7.1f\n', b, spotstr_(spot), max(spot,[],'omitnan'), min(ee,[],'omitnan'), ...
               g.len_mm, g.M2_mm, g.M3_mm, g.mass_kg, ef*1e3);
        catch e
            pr('%5.0f   FAILED: %s\n', b, regexprep(e.message,'\s+',' '));
        end
    end
    assert(~isempty(rows), 'dyson5_tma_step1: every bias failed');
    ns=[rows.nsolved]; sm=[rows.spot_max_um]; score = -ns*1e9 + sm;  % most-solved first, then smallest worst spot
    [~, ib] = min(score);  best = rows(ib);
    pr('\nBEST bias = %g deg: %d/%d strip fields imaged; worst imaged-field spot %.2f um (%.2f px), EE_min %.3f, EFL %.1f mm\n', ...
       best.bias_deg, best.nsolved, opt.nfield, best.spot_max_um, best.spot_max_um*1e-6/opt.pixel_m, best.ee_min, best.efl_mm);
    pr('  conics K = [%.4f %.4f %.4f]; length %.1f mm, M2 %.1f mm, M3 %.1f mm; mirror mass %.2f kg at %.0f kg/m^2 areal density\n', ...
       best.conics(1),best.conics(2),best.conics(3), best.length_mm, best.M2_mm, best.M3_mm, best.mass_kg, opt.areal_density);

    % ---- render the best bias layout
    png = fullfile(here, [opt.tag '_layout.png']);
    try
        f1 = best.tel.view_orthoviews({'YZ','XZ'}, 'nrays', 11);
        saveas(f1, png);  if ishghandle(f1), close(f1); end
        pr('  layout rendered: %s\n', png);
    catch e
        pr('  layout render failed: %s\n', regexprep(e.message,'\s+',' '));
    end

    for i=1:numel(rows), rows(i).tel = []; end            % drop handles before save
    OUT = struct('rows', rows, 'best_bias_deg', best.bias_deg, 'opt', opt, 'D_m', D, 'layout', struct('R',R,'t',t,'lay',lay));
    fclose(fid);  save(fullfile(here,[opt.tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('dyson5_tma_step1: wrote %s.{txt,mat} + _layout.png\n', opt.tag); end
end

% ======================================================================
function t = build_tma_(R, t_sp, D, LAM, MODEL)
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',LAM,'model_size',MODEL);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',t_sp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',t_sp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP');
    t.build();
end

function [spot_um, ee] = spot_per_field_(tel, nE, fx, px)
%SPOT_PER_FIELD_  Per-field BEST-FOCUS geometric rms spot radius (um) and energy
%   in an 18 um pixel.  FP-placement-independent: for rays p_i + t d_i near focus,
%   the transverse rms is quadratic in the focus distance t, minimised in closed
%   form (t* = -Cov(a,b)/Var(b)).  A 6-7.6 deg off-axis field (bias + strip) is far
%   off the on-axis derived focus, so best-focus is the only honest spot metric.
    spot_um = nan(1,numel(fx));  ee = nan(1,numel(fx));
    for i = 1:numel(fx)
        tel.trace_at_field([fx(i) 0]);                     % cross-track field about the bias
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;
        n0 = nnz(ok);  if n0 < 20, continue; end           % beam lost at this field
        P = ri.pos(:,ok);  Dd = ri.dir(:,ok);
        ch = mean(Dd,2);  ch = ch/norm(ch);                % chief direction
        % sigma-clip stray rays that pass the ok mask but trace wild (coaxial-TMA
        % vignetting leaves a few rays metres off); keep the main beam.
        a0 = P - ch*(ch.'*P);  keep = true(1, size(P,2));
        for it = 1:5
            cen = mean(a0(:,keep),2);  rr = vecnorm(a0 - cen);
            keep = keep & (rr <= mean(rr(keep)) + 4*std(rr(keep)));
        end
        if nnz(keep) < 0.5*n0, continue; end               % mostly stray -> field not imaged
        P = P(:,keep);  Dd = Dd(:,keep);
        a = P  - ch*(ch.'*P);   a = a - mean(a,2);         % transverse position, centred
        b = Dd - ch*(ch.'*Dd);  b = b - mean(b,2);         % transverse slope, centred
        Vaa = mean(sum(a.^2,1));  Vbb = mean(sum(b.^2,1));  Vab = mean(sum(a.*b,1));
        tstar = -Vab/max(Vbb,eps);
        spot_um(i) = sqrt(max(Vaa - Vab^2/max(Vbb,eps), 0))*1e6;   % best-focus rms radius
        Q = a + tstar*b;                                   % transverse miss at best focus
        xh = cross([0;1;0], ch);  xh = xh/norm(xh);  yh = cross(ch, xh);
        u = xh.'*Q;  v = yh.'*Q;
        ee(i) = mean(abs(u - mean(u)) <= px/2 & abs(v - mean(v)) <= px/2);
    end
    tel.trace_at_field([]);
end

function g = geom_(tel, nE, fx, areal, D)
    e = tel.spec.elt;
    zz = arrayfun(@(k) e(k).Vpt(3), 1:nE);
    g.len_mm = (max(zz) - min(zz))*1e3;                    % full axial envelope (frontmost optic to FP)
    B = tel.ray_bundle('fields', [[0 0]; [fx(fx~=0).' zeros(sum(fx~=0),1)]]);   % strip incl. bias
    dia = @(k) footdia_(B, k);
    g.M2_mm = dia(2)*1e3;  g.M3_mm = dia(3)*1e3;  M1_m = D;   % M1 fills the aperture
    areas = pi*((M1_m/2)^2 + (g.M2_mm/2e3)^2 + (g.M3_mm/2e3)^2);
    g.mass_kg = areal*areas;
end

function d = footdia_(B, k)
    P = [];
    for f = 1:numel(B.pos)
        pk = B.pos{f}(:,:,k);  ok = B.ok{f}(:,k).';
        P = [P, pk(:,ok)];   %#ok<AGROW>
    end
    if isempty(P), d = NaN; return; end
    c = mean(P,2);  d = 2*max(vecnorm(P - c));
end

function ef = efl_by_trace_(R, t, D, LAM, MODEL)
    tel = build_tma_(R, t, D, LAM, MODEL);
    ef = efl_of_built_(tel, numel(tel.spec.elt), LAM);
end

function ef = efl_of_built_(tel, nE, ~)
    th = 0.02*pi/180;                                      % 0.02 deg probe field
    y = zeros(1,2);
    for j = 1:2
        tel.trace_at_field([ (2*j-3)*th, 0 ]);             % -th, +th cross-track
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;  c = mean(ri.pos(:,ok),2);  y(j) = c(1);
    end
    tel.trace_at_field([]);
    ef = abs(y(2)-y(1))/(2*th);                            % |d(image x)/d(theta)| = EFL
end

function w = rms_waves_(W, lam)
    v = W(isfinite(W) & W ~= 0);  if isempty(v), w = NaN; else, w = std(v)/lam; end
end
function s = spotstr_(v), s = strtrim(sprintf('%6.2f', v)); end
function dp_(fid, quiet, varargin), fprintf(fid, varargin{:}); if ~quiet, fprintf(varargin{:}); end, end
