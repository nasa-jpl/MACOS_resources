function OUT = dyson5_tma_step5(over)
%DYSON5_TMA_STEP5  Round 4 step 5: FREEFORM on the outer field, both modules.
%
%   Stage B (TO) made the strip EDGES the limit on a telecentric, correctly-
%   scaled section -- 1.5k 1.3 px centre / 15.8 px at +-2.35 deg, 3k 58 um
%   centre / 1.2 mm at +-4.69 deg -- because rotationally symmetric aspheres
%   do not reach the field-dependent (coma/astig) blur off the section's own
%   axis (CCMac's step-4 finding, reproduced on the telecentric section).
%   This step takes item (2) of stage B's next-DOF list: NON-SYMMETRIC
%   (freeform / Zernike) terms for the outer field.
%
%   ROUTE B (Dave 2026-10-04): a SELF-CONSISTENT freeform solve from the
%   stage-B CONIC geometry (same parent / bias / decenter / conics, read from
%   the tA .mat), over a Zernike departure that spans the CORRECT symmetric
%   modes {5 defocus, 13 primary spherical, 25 secondary spherical} -- the
%   aspheres' job -- PLUS the non-symmetric {4,6 astig, 7,10 trefoil, 8,9 coma}
%   that the aspheres cannot reach, on M1-M3 over the WHOLE strip incl. edges.
%   No absolute asphere->Zernike conversion (that is the deferred emitter fold,
%   NOTE_asph_zernike_fold.md + PLAN_DESIGN_LAYER): the optimizer finds the
%   coefficients self-consistently, so the ZernCoef convention does not bite.
%   (My first step-5 run used "ANSI 4-11", which OMITTED the spherical modes
%   13/25 -- the reason it could not reproduce the aspheres' centre.)  Per-field
%   IMAGE-POSITION rows (`beam_pos_fov` = 330 mm*tan(theta)) ride in the SAME
%   solve so the 330 mm plate scale holds (the optimize_freeform hook CCMac
%   added for this step).
%
%   One BOUNDED rung per module (ANSI 4-11, lMon = per-mirror footprint
%   radius so OptZern stays conditioned, beam_wt stated).  Per module: rms
%   spot per field before/after (best focus, from the trace -- the 3k edges
%   are scored here because CALIB still drops +-4.69 deg, CC's open item),
%   traced plate scale, clearance, length / M2,M3 dia / mirror mass.  Emits
%   dyson5_tma_step5_<module>.in for TO's t5e end-to-end score.
%
%   OUT = DYSON5_TMA_STEP5(OVER) overrides .modules, .modes, .beam_wt,
%   .max_iters, .model, .quiet, etc.  Records dyson5_tma_step5.{txt,mat} +
%   _layout_<module>.png.  NEW FILE (round 4).  Helpers verbatim from step 4.
    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    opt = struct('EFL_m', 0.330, 'Fsys', 1.8, 'lam', 633e-9, 'model', 256, ...
                 'pixel_m', 18e-6, 'nfield', 7, 'areal_density', 25, ...
                 'modes', [4 5 6 7 8 9 10 13 25], 'ff_elts', [1 2 3], ...
                 'beam_wt', 1e-1, 'max_iters', 200, 'quiet', false, ...
                 'tag', 'dyson5_tma_step5', ...
                 'modules', struct( ...
                     'name',    {'1k5', '3k'}, ...
                     'mat',     {'dyson5_tA_B1k5.mat', 'dyson5_tA_B3k4.mat'}, ...
                     'rung',    {4, 3}, ...        % 1.5k = B1 (rung 4); 3k = B0.1 (rung 3, the e2e row)
                     'npix_xt', {1500, 3000}, ...
                     'dyson',   {'size:D:130', 'size:F:240'}));
    fn = fieldnames(over);
    for k = 1:numel(fn), if ~isfield(opt, fn{k}), error('unknown option %s', fn{k}); end, opt.(fn{k}) = over.(fn{k}); end
    D = opt.EFL_m/opt.Fsys;

    addpath('/Users/dcr/dev/MACOS_resources/mmacos/src');  macos.init(opt.model);
    fid = fopen(fullfile(here,[opt.tag '.txt']),'w');  pr = @(varargin) dp_(fid, opt.quiet, varargin{:});
    pr('dyson5 round 4 step 5 (CCMac) -- freeform on the outer field (%s)\n', datestr(now,'yyyy-mm-dd HH:MM'));
    pr('one bounded Zernike rung per module on the stage-B CONIC section: modes ANSI %s on M%s, lMon = footprint radius,\n', ...
       mat2str(opt.modes), mat2str(opt.ff_elts));
    pr('conic-base + Zernike departure over the FULL strip incl. edges; beam_pos_fov = 330 mm*tan(theta) at beam_wt %g holds the plate scale.\n', opt.beam_wt);
    pr('spot = best-focus rms radius (um) per field FROM THE TRACE (so the 3k edges are scored even where CALIB drops them).\n\n');

    rows = struct('module',{},'spot0_um',{},'spot_um',{},'spot_max0_um',{},'spot_max_um',{}, ...
                  'spot_ctr_um',{},'plate0_mm',{},'plate_mm',{},'wfe_before',{},'wfe_after',{}, ...
                  'clear_ok',{},'clear_note',{},'length_mm',{},'M2_mm',{},'M3_mm',{},'mass_kg',{}, ...
                  'conics',{},'lmon_mm',{},'deck',{},'tel',{});
    for m = 1:numel(opt.modules)
        md = opt.modules(m);
        S = load_mat_(here, md.mat);
        R = S.parent.R(:).';  t = S.parent.t(:).';  b = S.work.b;  dec = S.work.d;
        K = S.ladder(md.rung).K(:).';  shd = S.strip_half_deg;
        fx = linspace(-shd, shd, opt.nfield)*pi/180;
        fields_full = [fx(fx~=0).' zeros(nnz(fx~=0),1)];
        pr('===== module %s: parent R=%s t=%s, bias %g deg, decenter %.0f mm, strip +-%.2f deg =====\n', ...
           md.name, mat2str(R,5), mat2str(t,5), b, dec*1e3, shd);
        pr('stage-B conics K = [%.4f %.4f %.4f] (tA rung %d)\n', K(1),K(2),K(3), md.rung);
        try
            tel = build_tma_conics_(R, t, D, opt.lam, opt.model, K);
            tel.set_field_bias(b*60);
            tel.set_offaxis('none', 'dist', dec);
            tel.build();  nE = numel(tel.spec.elt);

            % baseline (conics only, before the freeform rung)
            [spot0, ee0] = spot_per_field_(tel, nE, fx, opt.pixel_m);
            plate0 = efl_of_built_(tel, nE, opt.lam)*1e3;

            % the bounded freeform rung: lMon = per-mirror footprint radius,
            % position rows = 330 mm*tan(theta) at beam_wt (CCMac's hook)
            lmon = footrad3_(tel, nE, fx, opt.ff_elts);
            P    = plate_targets_(tel, nE, fields_full, opt.EFL_m);
            res  = tel.optimize_freeform(opt.ff_elts, 'modes', opt.modes, ...
                       'fields', fields_full, 'lmon', lmon, ...
                       'beam_pos_fov', P, 'beam_wt', opt.beam_wt, 'max_iters', opt.max_iters);
            nE = numel(tel.spec.elt);

            [spot, ee] = spot_per_field_(tel, nE, fx, opt.pixel_m);
            g = geom_(tel, nE, fx, opt.areal_density, D);
            [cok, cnote] = clearance_(tel);
            plate = efl_of_built_(tel, nE, opt.lam)*1e3;
            deck = fullfile(here, sprintf('%s_%s.in', opt.tag, md.name));
            tel.build(deck);

            row = struct('module',md.name,'spot0_um',spot0,'spot_um',spot, ...
                'spot_max0_um',max(spot0,[],'omitnan'),'spot_max_um',max(spot,[],'omitnan'), ...
                'spot_ctr_um',spot(ceil(numel(spot)/2)),'plate0_mm',plate0,'plate_mm',plate, ...
                'wfe_before',res.wfe_before,'wfe_after',res.wfe_after, ...
                'clear_ok',cok,'clear_note',cnote,'length_mm',g.len_mm,'M2_mm',g.M2_mm, ...
                'M3_mm',g.M3_mm,'mass_kg',g.mass_kg,'conics',K,'lmon_mm',lmon*1e3,'deck',deck,'tel',tel);
            rows(end+1) = row;   %#ok<AGROW>

            pr('  before (conics):  %-55s  worst %.1f px, ctr %.2f px, plate %.1f mm\n', ...
               spotstr_(spot0), row.spot_max0_um*1e-6/opt.pixel_m, spot0(ceil(numel(spot0)/2))*1e-6/opt.pixel_m, plate0);
            pr('  after (freeform): %-55s  worst %.1f px, ctr %.2f px, plate %.1f mm\n', ...
               spotstr_(spot), row.spot_max_um*1e-6/opt.pixel_m, row.spot_ctr_um*1e-6/opt.pixel_m, plate);
            pr('  WFE per CALIB field  before %s  after %s nm\n', ...
               mat2str(round(res.wfe_before*1e9)), mat2str(round(res.wfe_after*1e9)));
            pr('  lMon = [%.1f %.1f %.1f] mm; clearance %s (%s); len %.1f mm, M2 %.1f, M3 %.1f, mass %.2f kg\n', ...
               lmon*1e3, tern_(cok,'PASS','FAIL'), cnote, g.len_mm, g.M2_mm, g.M3_mm, g.mass_kg);
            pr('  deck: %s\n', deck);

            png = fullfile(here, sprintf('%s_layout_%s.png', opt.tag, md.name));
            try, f1 = tel.view_orthoviews({'YZ','XZ'},'nrays',11); saveas(f1,png); if ishghandle(f1), close(f1); end
                 pr('  layout: %s\n', png); catch e, pr('  layout failed: %s\n', regexprep(e.message,'\s+',' ')); end
        catch e
            pr('  FAILED: %s\n', regexprep(e.message,'\s+',' '));
        end
        pr('\n');
    end
    assert(~isempty(rows), 'dyson5_tma_step5: every module failed');

    OUT = struct('rows', rows, 'opt', opt, 'D_m', D);
    for i=1:numel(rows), rows(i).tel = []; end
    OUT.rows = rows;
    fclose(fid);  save(fullfile(here,[opt.tag '.mat']), 'OUT');
    if ~opt.quiet, fprintf('dyson5_tma_step5: wrote %s.{txt,mat} + decks + layouts\n', opt.tag); end
end

% ======================================================================
function S = load_mat_(here, name)
    w = load(fullfile(here, name));  S = w.S;
end

function t = build_tma_conics_(R, t_sp, D, LAM, MODEL, K)
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',D,'wavelength_m',LAM,'model_size',MODEL);
    t.add_mirror('M1','radius_m',R(1),'spacing_after_m',t_sp(1),'conic',K(1));
    t.add_mirror('M2','radius_m',R(2),'spacing_after_m',t_sp(2),'convex',true,'conic',K(2));
    t.add_mirror('M3','radius_m',R(3),'spacing_after','derive','conic',K(3));
    t.add_focal_plane('FP');  t.build();
end

function r = footrad3_(tel, nE, fx, elts)
    B = tel.ray_bundle('fields', [[0 0]; [fx(fx~=0).' zeros(sum(fx~=0),1)]]);
    r = zeros(1, numel(elts));
    for j = 1:numel(elts), r(j) = footdia_(B, elts(j))/2; end
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

function [ok, note] = clearance_(tel)
    ok = false;  note = 'check_clipping unavailable';
    try
        rep = tel.check_clipping('quiet', true);
        ok = all([rep.ok]);  [cmin, imin] = min([rep.clearance]);
        note = sprintf('min clearance %+.1f mm at %s; %d conflicts', cmin*1e3, rep(imin).name, sum([rep.obstructs]));
    catch e, note = regexprep(e.message,'\s+',' '); end
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
