function OUT = dyson5_tma_step4(over)
%DYSON5_TMA_STEP4  Round 4 step 4: aspheres to buy the plate scale AND the blur.
%
%   Restarts from CC's step-2b d205 unobscured section (decenter 205 mm, bias 3 deg
%   on the step-1 calibrated f=330/F1.8/D=183 Korsch parent) and adds EVEN-RADIAL
%   ASPHERES on M1/M2/M3 (CC's hook: Telescope.optimize 'asph_elts'/'asph_terms' ->
%   CALIB OptAsph=, engine macos 9fe033e) ON TOP of the three conics and the
%   plate-scale image-position rows ('beam_pos_fov'/'beam_wt', the per-field targets
%   = 330 mm*tan(theta)).  Three conics gave 8.9 px at a 447 mm plate scale and could
%   not pin the scale without wrecking the blur (CC step 2b); the aspheres are the
%   extra DOFs that may buy BOTH -- the 447->330 mm scale and a tighter strip -- and
%   image the 3k strip the conics could not.  Walks 'beam_wt' 1e-2 -> 1e-1 -> 1.
%
%   Modules (each its own telescope): 1k5 (strip +-2.35 deg) and 3k (+-4.7 deg).
%   Per (module, beam_wt): rms spot per field (best focus), traced plate scale
%   (efl_of_built_), clearance (check_clipping), length / M2,M3 dia / mirror mass.
%
%   OUT = DYSON5_TMA_STEP4(OVER) overrides: .modules, .beam_wts, .asph_elts,
%   .asph_terms, .decenter_m, .bias_deg, .max_iters, .quiet, etc.
%   Records dyson5_tma_step4.{txt,mat} + _layout_<module>.png.  NEW FILE (round 4).
%   Helpers are CC's step-2 / my step-1 helpers, verbatim (per-driver self-contained).
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    opt = struct('EFL_m', 0.330, 'Fsys', 1.8, 'primary_fnum', 1.0, 'secondary_mag', 3.5, ...
                 'int_focus_D', -0.125, 'm3_behind_D', 0.6, 'lam', 633e-9, 'model', 256, ...
                 'decenter_m', 0.205, 'bias_deg', 3, 'modules', struct('name',{'1k5','3k'},'strip_half_deg',{2.35,4.7}), ...
                 'nfield', 7, 'pixel_m', 18e-6, 'max_iters', 150, 'areal_density', 25, ...
                 'asph_elts', [1 2 3], 'asph_terms', [1 2], 'beam_wts', [1e-2 1e-1 1], ...
                 'quiet', false, 'tag', 'dyson5_tma_step4');
    fn = fieldnames(over);
    for k = 1:numel(fn), if ~isfield(opt, fn{k}), error('unknown option %s', fn{k}); end, opt.(fn{k}) = over.(fn{k}); end
    D = opt.EFL_m/opt.Fsys;

    addpath('/Users/dcr/dev/MACOS_resources/mmacos/src');  macos.init(opt.model);
    fid = fopen(fullfile(here,[opt.tag '.txt']),'w');  pr = @(varargin) dp_(fid, opt.quiet, varargin{:});
    pr('dyson5 round 4 step 4 (CCMac) -- aspheres on the d205 section: plate scale AND blur (%s)\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('restart = CC step-2b section (decenter %.0f mm, bias %g deg on the f=330/F1.8/D=183 Korsch); hook asph_elts=%s asph_terms=%s (h^4,h^6)\n', ...
       opt.decenter_m*1e3, opt.bias_deg, mat2str(opt.asph_elts), mat2str(opt.asph_terms));
    pr('on top of 3 conics + plate rows (beam_pos_fov = 330 mm*tan(theta)); walk beam_wt.  spot = best-focus rms radius, um; plate = traced EFL on the section.\n\n');

    % ---- step-1 layout, calibrated so the on-axis exact-traced EFL = 330 mm (reproduces CC's d205 parent on the fixed engine)
    fsys_req = opt.Fsys;  R=[]; t=[]; lay=[]; efl_trace=NaN;
    for it = 1:10
        [R, t, lay] = macos.design.tma_layout(D, opt.primary_fnum, fsys_req, 'secondary_mag', opt.secondary_mag, ...
                          'int_focus_m', opt.int_focus_D*D, 'm3_behind_m', opt.m3_behind_D*D);
        efl_trace = efl_by_trace_(R, t, D, opt.lam, opt.model);
        if abs(efl_trace - opt.EFL_m)/opt.EFL_m < 0.004, break; end
        fsys_req = fsys_req * (opt.EFL_m / efl_trace);
    end
    pr('[layout] R=[%.4f %.4f %.4f] m  t=[%.4f %.4f] m  on-axis exact EFL %.4f m, F/%.2f at D=%.1f mm\n\n', ...
       R(1),R(2),R(3), t(1),t(2), efl_trace, efl_trace/D, D*1e3);

    rows = struct('module',{},'variant',{},'beam_wt',{},'spot_um',{},'spot_max_um',{},'spot_ctr_um',{}, ...
                  'nsolved',{},'ee_min',{},'plate_mm',{},'clear_ok',{},'clear_note',{}, ...
                  'length_mm',{},'M2_mm',{},'M3_mm',{},'mass_kg',{},'conics',{},'tel',{});
    for m = 1:numel(opt.modules)
        md = opt.modules(m);
        fx = linspace(-md.strip_half_deg, md.strip_half_deg, opt.nfield)*pi/180;
        fields_full = [fx(fx~=0).' zeros(nnz(fx~=0),1)];
        pr('===== module %s: strip +-%.2f deg, %d fields =====\n', md.name, md.strip_half_deg, opt.nfield);
        pr('%-10s %7s %-55s %8s %8s %6s %7s %6s %7s %7s %7s %7s\n','variant','beam_wt','rms spot per field, um (NaN=not imaged)','maxspot','ctr','EEmin','plate','clear','len mm','M2 mm','M3 mm','mass kg');
        variants = {'conics', 0; 'asph(blur)', NaN};        % NaN wt = aspheres, NO plate-scale rows (blur alone)
        for w = opt.beam_wts(:).', variants(end+1,:) = {'asph+scale', w}; end   %#ok<AGROW>
        for vi = 1:size(variants,1)
            vname = variants{vi,1};  wt = variants{vi,2};
            try
                tel = build_tma_(R, t, D, opt.lam, opt.model);
                tel.set_field_bias(opt.bias_deg*60);
                tel.set_offaxis('none', 'dist', opt.decenter_m);
                tel.build();  nE = numel(tel.spec.elt);
                P = plate_targets_(tel, nE, fields_full, opt.EFL_m);
                if strcmp(vname,'conics')
                    tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], 'max_iters', opt.max_iters);
                elseif strcmp(vname,'asph(blur)')
                    tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], ...
                                 'asph_elts', opt.asph_elts, 'asph_terms', opt.asph_terms, 'max_iters', opt.max_iters);
                else
                    tel.optimize('fields', fields_full, 'dofs', [0 0 0 0 0 0 0 1], ...
                                 'asph_elts', opt.asph_elts, 'asph_terms', opt.asph_terms, ...
                                 'beam_pos_fov', P, 'beam_wt', wt, 'max_iters', opt.max_iters);
                end
                [spot, ee] = spot_per_field_(tel, nE, fx, opt.pixel_m);
                g = geom_(tel, nE, fx, opt.areal_density, D);
                [cok, cnote] = clearance_(tel);
                plate = efl_of_built_(tel, nE, opt.lam)*1e3;
                row = struct('module',md.name,'variant',vname,'beam_wt',wt,'spot_um',spot, ...
                             'spot_max_um',max(spot,[],'omitnan'),'spot_ctr_um',spot(ceil(numel(spot)/2)), ...
                             'nsolved',sum(~isnan(spot)),'ee_min',min(ee,[],'omitnan'),'plate_mm',plate, ...
                             'clear_ok',cok,'clear_note',cnote,'length_mm',g.len_mm,'M2_mm',g.M2_mm,'M3_mm',g.M3_mm, ...
                             'mass_kg',g.mass_kg,'conics',[tel.spec.elt(1).Kc tel.spec.elt(2).Kc tel.spec.elt(3).Kc],'tel',tel);
                rows(end+1) = row;   %#ok<AGROW>
                pr('%-10s %7.0e %-55s %8.1f %8.1f %6.3f %7.1f %6s %7.1f %7.1f %7.1f %7.2f\n', vname, wt, spotstr_(spot), ...
                   row.spot_max_um, row.spot_ctr_um, row.ee_min, plate, tern_(cok,'PASS','FAIL'), g.len_mm, g.M2_mm, g.M3_mm, g.mass_kg);
            catch e
                pr('%-10s %7.0e FAILED: %s\n', vname, wt, regexprep(e.message,'\s+',' '));
            end
        end
        pr('\n');
    end
    assert(~isempty(rows), 'dyson5_tma_step4: every variant failed');

    % best per module: clearance PASS, plate scale closest to 330, then smallest worst spot
    OUT = struct('rows', rows, 'opt', opt, 'D_m', D, 'layout', struct('R',R,'t',t));
    for m = 1:numel(opt.modules)
        md = opt.modules(m);  sel = find(strcmp({rows.module}, md.name));
        if isempty(sel), continue; end
        sc = arrayfun(@(i) (~rows(i).clear_ok)*1e9 + abs(rows(i).plate_mm-opt.EFL_m*1e3)*10 + rows(i).spot_max_um, sel);
        [~,bi] = min(sc);  best = rows(sel(bi));
        pr('BEST %s: %s (beam_wt %.0e): worst spot %.1f um (%.2f px), centre %.1f um (%.2f px), EE_min %.3f, plate %.1f mm, clearance %s, mass %.2f kg\n', ...
           md.name, best.variant, best.beam_wt, best.spot_max_um, best.spot_max_um*1e-6/opt.pixel_m, ...
           best.spot_ctr_um, best.spot_ctr_um*1e-6/opt.pixel_m, best.ee_min, best.plate_mm, tern_(best.clear_ok,'PASS','FAIL'), best.mass_kg);
        pr('   conics K=[%.4f %.4f %.4f]; %s\n', best.conics, best.clear_note);
        png = fullfile(here, sprintf('%s_layout_%s.png', opt.tag, md.name));
        try, f1 = best.tel.view_orthoviews({'YZ','XZ'},'nrays',11); saveas(f1,png); if ishghandle(f1), close(f1); end
             pr('   layout: %s\n', png); catch e, pr('   layout failed: %s\n', regexprep(e.message,'\s+',' ')); end
        OUT.(['best_' md.name]) = rmfield_tel_(best);
    end
    for i=1:numel(rows), rows(i).tel = []; end
    OUT.rows = rows;
    fclose(fid);  save(fullfile(here,[opt.tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('dyson5_tma_step4: wrote %s.{txt,mat} + layouts\n', opt.tag); end
end

% ======================================================================
function b = rmfield_tel_(b), b.tel = []; end

function [ok, note] = clearance_(tel)
    ok = false;  note = 'check_clipping unavailable';
    try
        rep = tel.check_clipping('quiet', true);
        ok = all([rep.ok]);  [cmin, imin] = min([rep.clearance]);
        note = sprintf('min clearance %+.1f mm at %s; %d conflicts', cmin*1e3, rep(imin).name, sum([rep.obstructs]));
    catch e, note = regexprep(e.message,'\s+',' '); end
end

function t = build_tma_(R, t_sp, D, LAM, MODEL)
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',LAM,'model_size',MODEL);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',t_sp(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',t_sp(2),'convex',true);
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive');
    t.add_focal_plane('FP');  t.build();
end

function [spot_um, ee] = spot_per_field_(tel, nE, fx, px)
    spot_um = nan(1,numel(fx));  ee = nan(1,numel(fx));
    for i = 1:numel(fx)
        tel.trace_at_field([fx(i) 0]);
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;  n0 = nnz(ok);  if n0 < 20, continue; end
        P = ri.pos(:,ok);  Dd = ri.dir(:,ok);  ch = mean(Dd,2);  ch = ch/norm(ch);
        a0 = P - ch*(ch.'*P);  keep = true(1, size(P,2));
        for it = 1:5, cen = mean(a0(:,keep),2);  rr = vecnorm(a0 - cen);  keep = keep & (rr <= mean(rr(keep)) + 4*std(rr(keep))); end
        if nnz(keep) < 0.5*n0, continue; end
        P = P(:,keep);  Dd = Dd(:,keep);
        a = P  - ch*(ch.'*P);   a = a - mean(a,2);  b = Dd - ch*(ch.'*Dd);  b = b - mean(b,2);
        Vaa = mean(sum(a.^2,1));  Vbb = mean(sum(b.^2,1));  Vab = mean(sum(a.*b,1));
        tstar = -Vab/max(Vbb,eps);  spot_um(i) = sqrt(max(Vaa - Vab^2/max(Vbb,eps), 0))*1e6;
        Q = a + tstar*b;  xh = cross([0;1;0], ch);  xh = xh/norm(xh);  yh = cross(ch, xh);
        u = xh.'*Q;  v = yh.'*Q;  ee(i) = mean(abs(u - mean(u)) <= px/2 & abs(v - mean(v)) <= px/2);
    end
    tel.trace_at_field([]);
end

function g = geom_(tel, nE, fx, areal, D)
    e = tel.spec.elt;  zz = arrayfun(@(k) e(k).Vpt(3), 1:nE);  g.len_mm = (max(zz) - min(zz))*1e3;
    B = tel.ray_bundle('fields', [[0 0]; [fx(fx~=0).' zeros(sum(fx~=0),1)]]);
    g.M2_mm = footdia_(B,2)*1e3;  g.M3_mm = footdia_(B,3)*1e3;
    areas = pi*((D/2)^2 + (g.M2_mm/2e3)^2 + (g.M3_mm/2e3)^2);  g.mass_kg = areal*areas;
end

function d = footdia_(B, k)
    P = [];
    for f = 1:numel(B.pos), pk = B.pos{f}(:,:,k);  ok = B.ok{f}(:,k).';  P = [P, pk(:,ok)]; end   %#ok<AGROW>
    if isempty(P), d = NaN; return; end
    c = mean(P,2);  d = 2*max(vecnorm(P - c));
end

function P = plate_targets_(tel, nE, fields, EFL)
    th = 0.02*pi/180;
    tel.trace_at_field([0 0]);  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);  p0 = ri.pos(:,1);
    ux = unit_(hit_(tel, nE, [ th 0]) - hit_(tel, nE, [-th 0]));
    uy = unit_(hit_(tel, nE, [ 0 th]) - hit_(tel, nE, [0 -th]));
    tel.trace_at_field([]);
    P = zeros(3, 1 + size(fields,1));  P(:,1) = p0;
    for k = 1:size(fields,1), P(:,k+1) = p0 + EFL*tan(fields(k,1))*ux + EFL*tan(fields(k,2))*uy; end
end
function p = hit_(tel, nE, f), tel.trace_at_field(f); s = macos.trace(nE); ri = macos.get_ray_info(s.nRays); p = ri.pos(:,1); end
function u = unit_(v), u = v/norm(v); end

function ef = efl_by_trace_(R, t, D, LAM, MODEL)
    tel = build_tma_(R, t, D, LAM, MODEL);  ef = efl_of_built_(tel, numel(tel.spec.elt), LAM);
end
function ef = efl_of_built_(tel, nE, ~)
    th = 0.02*pi/180;  y = zeros(1,2);
    for j = 1:2
        tel.trace_at_field([ (2*j-3)*th, 0 ]);  s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace & ri.ok_pass;  c = mean(ri.pos(:,ok),2);  y(j) = c(1);
    end
    tel.trace_at_field([]);  ef = abs(y(2)-y(1))/(2*th);
end

function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
function s = spotstr_(v), s = strtrim(sprintf('%6.1f', v)); end
function dp_(fid, quiet, varargin), fprintf(fid, varargin{:}); if ~quiet, fprintf(varargin{:}); end, end
