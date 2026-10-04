function OUT = dyson5_tma_step2_linux(over)
%DYSON5_TMA_STEP2_LINUX  Round 4 step 2 (run on Linux by CC, 2026-10-03): the
%   UNOBSCURED eccentric-pupil section of the step-1 Korsch parent at Jim's
%   numbers (f = 330 mm, F/1.8, D = 183 mm), conics re-solved over the module's
%   cross-track STRIP, scored in the engine per field, clearance gated.
%
%   Step 1 (CCMac, dyson5_tma_step1.m) found the coaxial parent images only
%   +-1.5 deg of the 9.4 deg strip and cannot be biased clear of its own
%   image cone.  Step 2 removes the obscuration instead: Telescope.set_offaxis
%   ('all') decenters the pupil until every mirror clears the beam and emits
%   each mirror as a true off-axis SECTION of the unchanged parent conic
%   (VptElt = parent vertex, RptElt = section pole).  With no central
%   obstruction the along-track bias is a free knob, scanned small.  The
%   conics are then solved by CALIB in a ladder -- the section as is (the
%   parent's seed conics), the inner half-strip, the full strip -- each rung
%   its own row (CCMac's step 1 solved at the bias field only).  Both modules: 3k (+-4.7 deg) and 1.5k
%   (+-2.35 deg).  Every number is the ENGINE's (macos.trace on the emitted
%   deck); the spot is the best-focus transverse rms about the centroid, the
%   EE the energy in an 18 um pixel, both from step 1's helpers verbatim.
%
%   Records dyson5_tma_step2_linux.{txt,mat}, _layout_<module>.png and the
%   best deck per module dyson5_tma_step2_linux_<module>.in (for TO's join).
%   SEPARATE FILE from CCMac's step 2, if any (BRIEF_ccmac_dyson_size.md,
%   round 4 note of 2026-10-03 17:45).
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    opt = struct('EFL_m', 0.330, 'Fsys', 1.8, 'primary_fnum', 1.0, 'secondary_mag', 3.5, ...
                 'int_focus_D', -0.125, 'm3_behind_D', 0.6, 'lam', 633e-9, 'model', 256, ...
                 'biases_deg', [0 1 2], 'modules', struct('name', {'3k', '1k5'}, 'strip_half_deg', {4.7, 2.35}), ...
                 'nfield', 7, 'pixel_m', 18e-6, 'max_iters', 150, 'areal_density', 25, ...
                 'offaxis_margin', 0.05, 'decenter_m', 0.17, 'rungs', 0:2, 'recentre_bias_deg', 0, 'beam_wt', 1e-2, 'quiet', false, 'tag', 'dyson5_tma_step2_linux');
    fn = fieldnames(over);
    for k = 1:numel(fn), if ~isfield(opt, fn{k}), error('unknown option %s', fn{k}); end, opt.(fn{k}) = over.(fn{k}); end
    D = opt.EFL_m/opt.Fsys;

    macos.init(opt.model);
    fid = fopen(fullfile(here, [opt.tag '.txt']), 'w');  pr = @(varargin) dp_(fid, opt.quiet, varargin{:});
    pr('dyson5 round 4 step 2 (Linux, CC) -- unobscured eccentric section of the Korsch parent (%s)\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('f=%.0f mm, F/%.1f, D=%.1f mm; primary f/%.2f, secondary mag %.2f; set_offaxis(''none'', dist %.0f mm); conics by CALIB (WFE) in a ladder: as-is / inner half-strip / full strip,\n', ...
       opt.EFL_m*1e3, opt.Fsys, D*1e3, opt.primary_fnum, opt.secondary_mag, opt.decenter_m*1e3);
    pr('  conics only (FP not enrolled); rung ''strip+sp'' adds M2/M3 piston (spacing).  spot = best-focus rms radius about the centroid, um; EE = energy in an 18 um pixel.\n');
    pr('  clearance = Telescope.check_clipping on the emitted deck (body-in-beam / vignetting), reported per variant.\n\n');

    % ---- the step-1 calibrated layout (exact-traced EFL = 330 mm at D = 183 mm)
    fsys_req = opt.Fsys;  R = []; t = []; lay = [];  efl_trace = NaN;
    pr('[layout calibration] holding D=%.1f mm, primary f/%.2f, secondary mag %.2f; EFL by exact trace:\n', D*1e3, opt.primary_fnum, opt.secondary_mag);
    for it = 1:10
        [R, t, lay] = macos.design.tma_layout(D, opt.primary_fnum, fsys_req, 'secondary_mag', opt.secondary_mag, ...
                          'int_focus_m', opt.int_focus_D*D, 'm3_behind_m', opt.m3_behind_D*D);
        if opt.recentre_bias_deg > 0
            % step 2b: calibrate the TRACED plate scale on the SECTION at the working
            % bias and decenter (the eccentric section's local magnification off its
            % parent axis is not the parent's EFL: 375-450 mm at 2-3 deg in _d190)
            md1 = opt.modules(1);  fx1 = linspace(-md1.strip_half_deg, md1.strip_half_deg, opt.nfield)*pi/180;
            efl_trace = efl_on_section_(R, t, D, opt.lam, opt.model, opt.recentre_bias_deg, opt.decenter_m, ...
                                        [fx1(fx1~=0).' zeros(nnz(fx1~=0),1)], opt.max_iters);
        else
            efl_trace = efl_by_trace_(R, t, D, opt.lam, opt.model);
        end
        pr('  iter %d: requested f/%.3f (EFL_req %.4f) -> exact EFL %.4f m (F/%.3f)\n', it, fsys_req, lay.EFL, efl_trace, efl_trace/D);
        if abs(efl_trace - opt.EFL_m)/opt.EFL_m < 0.004 + 0.006*(opt.recentre_bias_deg > 0), break; end
        fsys_req = fsys_req * (opt.EFL_m / efl_trace);
    end
    pr('[layout] R=[%.4f %.4f %.4f] m  t=[%.4f %.4f] m  exact EFL %.4f m, F/%.2f at D=%.1f mm\n\n', ...
       R(1),R(2),R(3), t(1),t(2), efl_trace, efl_trace/D, D*1e3);

    rows = struct('module',{},'bias_deg',{},'decenter_mm',{},'solve',{},'spot_um',{},'spot_max_um',{},'nsolved',{},'ee_min',{}, ...
                  'clear_ok',{},'clear_note',{},'length_mm',{},'M2_mm',{},'M3_mm',{},'mass_kg',{},'conics',{},'tel',{});
    for m = 1:numel(opt.modules)
        md = opt.modules(m);
        fx = linspace(-md.strip_half_deg, md.strip_half_deg, opt.nfield) * pi/180;   % cross-track offsets (rad) about the bias
        pr('===== module %s: strip +-%.2f deg, %d fields =====\n', md.name, md.strip_half_deg, opt.nfield);
        pr('%5s %8s %-9s %-55s %8s %7s %6s %7s %7s %7s %7s\n','bias','decen mm','solve','rms spot per field across the strip, um (NaN=not imaged)','maxspot','EEmin','clear','len mm','M2 mm','M3 mm','mass kg');
        for b = opt.biases_deg
            % A ladder per bias, each rung its own row: R0 = the SECTION AS IS (the
            % parent's seed conics) at an EXPLICIT decenter.  Probed first (2026-10-03
            % 18:45, probe_sec.m): M2 clears the incoming beam at ~0.17 m (M1 at 0.10),
            % but the FP is pierced by the M2->M3 beam at EVERY decenter -- in a Korsch
            % the FP and the M1 hole are concentric, only the along-track BIAS separates
            % them -- so set_offaxis('all'|'M2') ran its bisection to the 1.5 D bound
            % (the first two Linux runs) and the as-is spot grew 11 um -> 0.16 / 0.27 /
            % 4.4 mm at d = 0 / 0.10 / 0.15 / 0.20 m (the seed is not a nulled anastigmat
            % over a sub-pupil; the solve must redo the conics on the section);
            % R1 = conics solved on the INNER fields (half the strip); R2 = the FULL
            % strip.  Conics only; the FP is not enrolled (a free detector tilt plus
            % three conics on a decentered section ran away in the first run).
            dec = NaN;
            for rung = opt.rungs
                try
                    tel = build_tma_(R, t, D, opt.lam, opt.model);
                    if b > 0, tel.set_field_bias(b*60); end                    % along-track bias, ARCMIN
                    dec = tel.set_offaxis('none', 'dist', opt.decenter_m);       % the section at an explicit decenter (probe: M2 clears at ~0.17 m)
                    tel.build();
                    fields_full = [fx(fx~=0).' zeros(nnz(fx~=0),1)];
                    fields_half = fields_full(abs(fields_full(:,1)) <= md.strip_half_deg*pi/180/2 + 1e-12, :);
                    switch rung
                        case 0, solve = 'as-is';
                        case 1, solve = 'inner';  tel.optimize('fields', fields_half, 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', opt.max_iters);
                        case 2, solve = 'strip';  tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', opt.max_iters);
                        case 3, solve = 'strip+sp';   % conics + M2/M3 piston (spacing); M1 conic only; EFL reported
                                tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1; 0 0 0 0 0 1 0 1; 0 0 0 0 0 1 0 1], 'max_iters', opt.max_iters);
                        case 4, solve = 'strip+ps';   % step 2b: conics + per-field IMAGE-POSITION rows pinning the plate scale to EFL_m
                                P = plate_targets_(tel, numel(tel.spec.elt), fields_full, opt.EFL_m);
                                tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', opt.max_iters, ...
                                             'beam_pos_fov', P, 'beam_wt', opt.beam_wt);
                    end
                    nE = numel(tel.spec.elt);
                    [spot, ee] = spot_per_field_(tel, nE, fx, opt.pixel_m);
                    g = geom_(tel, nE, fx, opt.areal_density, D);
                    [cok, cnote] = clearance_(tel);
                    efl_row = efl_of_built_(tel, nE, opt.lam);
                    row = struct('module',md.name,'bias_deg',b,'decenter_mm',dec*1e3,'solve',solve,'spot_um',spot, ...
                                 'spot_max_um',max(spot,[],'omitnan'),'nsolved',sum(~isnan(spot)),'ee_min',min(ee,[],'omitnan'), ...
                                 'clear_ok',cok,'clear_note',cnote,'length_mm',g.len_mm,'M2_mm',g.M2_mm,'M3_mm',g.M3_mm, ...
                                 'mass_kg',g.mass_kg,'conics',[tel.spec.elt(1).Kc tel.spec.elt(2).Kc tel.spec.elt(3).Kc],'tel',tel);
                    rows(end+1) = row;   %#ok<AGROW>
                    pr('%5.0f %8.1f %-9s %-55s %8.2f %7.3f %6s %7.1f %7.1f %7.1f %7.2f\n', b, dec*1e3, solve, spotstr_(spot), ...
                       max(spot,[],'omitnan'), min(ee,[],'omitnan'), tern_(cok,'PASS','FAIL'), g.len_mm, g.M2_mm, g.M3_mm, g.mass_kg);
                    pr('        K = [%.4f %.4f %.4f]; EFL %.1f mm; clearance: %s\n', row.conics, efl_row*1e3, cnote);
                catch e
                    pr('%5.0f %8.1f %-9s FAILED: %s\n', b, dec*1e3, sprintf('rung%d', rung), regexprep(e.message,'\s+',' '));
                end
            end
        end
        % best for this module: most fields imaged, then clearance, then smallest worst spot
        rm = rows(strcmp({rows.module}, md.name));
        if isempty(rm), pr('  module %s: every bias failed\n\n', md.name); continue; end
        score = -[rm.nsolved]*1e9 - [rm.clear_ok]*1e6 + [rm.spot_max_um];
        [~, ib] = min(score);  best = rm(ib);
        pr('\nBEST %s: bias %g deg, decenter %.1f mm (%s solve): %d/%d strip fields imaged; worst spot %.2f um (%.2f px), EE_min %.3f, clearance %s\n', ...
           md.name, best.bias_deg, best.decenter_mm, best.solve, best.nsolved, opt.nfield, best.spot_max_um, best.spot_max_um*1e-6/opt.pixel_m, ...
           best.ee_min, tern_(best.clear_ok,'PASS','FAIL'));
        pr('  conics K = [%.4f %.4f %.4f]; length %.1f mm, M2 %.1f mm, M3 %.1f mm; mirror mass %.2f kg at %.0f kg/m^2\n', ...
           best.conics(1),best.conics(2),best.conics(3), best.length_mm, best.M2_mm, best.M3_mm, best.mass_kg, opt.areal_density);
        deck = fullfile(here, sprintf('%s_%s.in', opt.tag, md.name));
        try, best.tel.save(deck);  pr('  deck of record: %s\n', deck); catch e, pr('  deck save failed: %s\n', regexprep(e.message,'\s+',' ')); end
        png = fullfile(here, sprintf('%s_layout_%s.png', opt.tag, md.name));
        try
            f1 = best.tel.view_orthoviews({'YZ','XZ'}, 'nrays', 11);
            saveas(f1, png);  if ishghandle(f1), close(f1); end
            pr('  layout rendered: %s\n\n', png);
        catch e
            pr('  layout render failed: %s\n\n', regexprep(e.message,'\s+',' '));
        end
    end
    for i=1:numel(rows), rows(i).tel = []; end
    OUT = struct('rows', rows, 'opt', opt, 'D_m', D, 'layout', struct('R',R,'t',t,'lay',lay));
    fclose(fid);  save(fullfile(here,[opt.tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('%s: wrote %s.{txt,mat} + layouts\n', opt.tag, opt.tag); end
end

% ======================================================================
function [ok, note] = clearance_(tel)
%CLEARANCE_  Telescope.check_clipping on the emitted deck: body-in-beam /
%   vignetting conflicts.  ok = no conflict; note = the report's summary.
    ok = false;  note = 'check_clipping unavailable';
    try
        rep = tel.check_clipping('quiet', true);      % per-element: .ok, .clearance (m), .obstructs
        ok = all([rep.ok]);
        [cmin, imin] = min([rep.clearance]);
        note = sprintf('min body-to-foreign-beam clearance %+.1f mm at %s; %d body/beam conflicts', ...
                       cmin*1e3, rep(imin).name, sum([rep.obstructs]));
    catch e
        note = regexprep(e.message,'\s+',' ');
    end
end

function s = tern_(c, a, b), if c, s = a; else, s = b; end, end

% ---- step-1 helpers, verbatim (dyson5_tma_step1.m, CCMac) ----------------
function t = build_tma_(R, t_sp, D, LAM, MODEL)
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',LAM,'model_size',MODEL);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',t_sp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',t_sp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP');
    t.build();
end

function [spot_um, ee] = spot_per_field_(tel, nE, fx, px)
    spot_um = nan(1,numel(fx));  ee = nan(1,numel(fx));
    for i = 1:numel(fx)
        tel.trace_at_field([fx(i) 0]);
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;
        n0 = nnz(ok);  if n0 < 20, continue; end
        P = ri.pos(:,ok);  Dd = ri.dir(:,ok);
        ch = mean(Dd,2);  ch = ch/norm(ch);
        a0 = P - ch*(ch.'*P);  keep = true(1, size(P,2));
        for it = 1:5
            cen = mean(a0(:,keep),2);  rr = vecnorm(a0 - cen);
            keep = keep & (rr <= mean(rr(keep)) + 4*std(rr(keep)));
        end
        if nnz(keep) < 0.5*n0, continue; end
        P = P(:,keep);  Dd = Dd(:,keep);
        a = P  - ch*(ch.'*P);   a = a - mean(a,2);
        b = Dd - ch*(ch.'*Dd);  b = b - mean(b,2);
        Vaa = mean(sum(a.^2,1));  Vbb = mean(sum(b.^2,1));  Vab = mean(sum(a.*b,1));
        tstar = -Vab/max(Vbb,eps);
        spot_um(i) = sqrt(max(Vaa - Vab^2/max(Vbb,eps), 0))*1e6;
        Q = a + tstar*b;
        xh = cross([0;1;0], ch);  xh = xh/norm(xh);  yh = cross(ch, xh);
        u = xh.'*Q;  v = yh.'*Q;
        ee(i) = mean(abs(u - mean(u)) <= px/2 & abs(v - mean(v)) <= px/2);
    end
    tel.trace_at_field([]);
end

function g = geom_(tel, nE, fx, areal, D)
    e = tel.spec.elt;
    zz = arrayfun(@(k) e(k).Vpt(3), 1:nE);
    g.len_mm = (max(zz) - min(zz))*1e3;
    B = tel.ray_bundle('fields', [[0 0]; [fx(fx~=0).' zeros(sum(fx~=0),1)]]);
    dia = @(k) footdia_(B, k);
    g.M2_mm = dia(2)*1e3;  g.M3_mm = dia(3)*1e3;  M1_m = D;
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

function P = plate_targets_(tel, nE, fields, EFL)
%PLATE_TARGETS_  Per-field image-position targets (3 x nfov, global, m) in CALIB
%   field order (field 1 = the nominal/bias chief, then FIELDS): the bias chief's
%   FP hit plus EFL*tan(theta) along the FP's in-plane field directions, taken
%   from the nominal trace (the chief's displacement for small +-x / +-y fields).
    th = 0.02*pi/180;
    tel.trace_at_field([0 0]);  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);  p0 = ri.pos(:,1);
    ux = unit_(hit_(tel, nE, [ th 0]) - hit_(tel, nE, [-th 0]));
    uy = unit_(hit_(tel, nE, [ 0 th]) - hit_(tel, nE, [0 -th]));
    tel.trace_at_field([]);
    P = zeros(3, 1 + size(fields,1));  P(:,1) = p0;
    for k = 1:size(fields,1)
        P(:,k+1) = p0 + EFL*tan(fields(k,1))*ux + EFL*tan(fields(k,2))*uy;
    end
end
function p = hit_(tel, nE, f), tel.trace_at_field(f); s = macos.trace(nE); ri = macos.get_ray_info(s.nRays); p = ri.pos(:,1); end
function u = unit_(v), u = v/norm(v); end

function ef = efl_on_section_(R, t, D, LAM, MODEL, bias_deg, dec_m, fields, max_iters)
%EFL_ON_SECTION_  The traced plate scale at the working bias of the SOLVED
%   section: the strip CALIB is inside the loop (the un-solved section's
%   scale is meaningless -- 530 mm and non-monotone in f_req, 2026-10-03).
    tel = build_tma_(R, t, D, LAM, MODEL);
    tel.set_field_bias(bias_deg*60);  tel.set_offaxis('none', 'dist', dec_m);  tel.build();
    tel.optimize('fields', fields, 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', max_iters);
    ef = efl_of_built_(tel, numel(tel.spec.elt), LAM);
end

function ef = efl_by_trace_(R, t, D, LAM, MODEL)
    tel = build_tma_(R, t, D, LAM, MODEL);
    ef = efl_of_built_(tel, numel(tel.spec.elt), LAM);
end

function ef = efl_of_built_(tel, nE, ~)
    th = 0.02*pi/180;
    y = zeros(1,2);
    for j = 1:2
        tel.trace_at_field([ (2*j-3)*th, 0 ]);
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;  c = mean(ri.pos(:,ok),2);  y(j) = c(1);
    end
    tel.trace_at_field([]);
    ef = abs(y(2)-y(1))/(2*th);
end

function s = spotstr_(v), s = strtrim(sprintf('%6.2f', v)); end
function dp_(fid, quiet, varargin), fprintf(fid, varargin{:}); if ~quiet, fprintf(varargin{:}); end, end
