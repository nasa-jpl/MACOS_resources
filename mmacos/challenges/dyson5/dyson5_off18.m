function S = dyson5_off18(P, tag)
%DYSON5_OFF18  The F/1.8 Offner at the 54 mm slit (BRIEF_to_dyson5 addendum 46): one table for the deck.
%   S = dyson5_off18(P, tag) -- runner stage 'o18' (dyson5_run).  The all-
%   reflective sibling (concave sphere used in two zones, convex grating at
%   the stop) at the Dyson's own speed and slit: 3000 x 500 px at 18 um,
%   380-2500 nm, 2-px slit, F/P.off18_Fno, chord-ruled grooves ('planes',
%   as the engine), over the concave radii P.off18_R_m.  Per R:
%     ring   the slit's ring radius: the smallest that clears the grating
%            body by P.off18_clear_mm at the seed (bisection on the chain's
%            clearance gate, grating mount included).  Addendum 6's 0.22 R is
%            F/2.8's rule; at F/1.8 the grating is R/(2F) = 0.28 R across and
%            both beams cross it there (-33 mm at R 0.5, the gate's crossing
%            test of 2026-10-06), so the ring is re-derived, not assumed.
%     seed   the concentric seed (convex grating R/2, one concave sphere),
%            emitted with apertures and ENGINE-scored (spectrometer_score,
%            P.score_nx x P.score_nlam), clearance, sizes.
%     corr   offner_solve's classical corrections (convex radius factor,
%            second-zone radius factor and centre dy/dz) on the chain under
%            the ladder's residuals at that FIXED ring, then emitted and
%            ENGINE-scored.  Its second zone takes its own radius while the
%            two clear apertures still overlap -- reported, not buildable.
%     free   the same solve with the RING a variable bounded below by the
%            seed's clearing ring (CC, 2026-10-06: the ring is the Offner's
%            one layout freedom), warm from corr, the grating wall at
%            P.off18_clear_mm and a wall on the zone gap (the two clear
%            apertures may not share area once their figures differ).
%   Sizes: clear aperture = footprint + P.ap_margin_m (the declared ApVec);
%   blank = clear aperture + P.mount_margin_m all round; the two concave
%   zones are ONE blank when their figures coincide (seed) and their union
%   is sized; the zone gap (centre distance minus the two clear radii;
%   negative = the zones overlap) is reported because a corrected second
%   zone with its own radius cannot share area with the first.  Mass = the
%   blanks at 10 mm Zerodur-class (2530 kg/m^3) -- the brief's stated
%   assumption, a lightweighted flight mirror differs; the Dyson rows carry
%   their edged glass (dyson5_size.mat) plus their grating at the same rule.
%   Length = axial extent of the bodies and the slit/FPA plane.
%   Writes <tag>_off18.{txt,mat}, <tag>_off18_R<mm>_{seed,corr}.in and their
%   maps/layout/renders.
    Rs   = field_(P, 'off18_R_m', [0.5 0.75 1.0 1.25]);
    Fno  = field_(P, 'off18_Fno', 1.8);
    cl0  = field_(P, 'off18_clear_mm', 5);
    stp  = field_(P, 'off18_steps', {'seed', 'corr', 'free'});
    nit  = field_(P, 'off18_max_iter', 60);
    rho  = 2530;  tb = 10e-3;                              % the mass rule: 10 mm Zerodur-class blanks
    stem = [tag '_off18'];
    fid = fopen([stem '.txt'], 'w');
    pr = @(varargin) dp_(fid, varargin{:});
    pr('dyson5 off18 -- the F/%.1f Offner at the %.0f mm slit (addendum 46; %s)\n', Fno, P.npix(1)*P.pixel_m*1e3, datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: the s2 scorer (engine rays, %d slit positions x %d wavelengths, %d-pt grid, model %d); smile/keystone p-v of the\n', P.score_nx, P.score_nlam, P.ngridpts, P.model);
    pr('  centroids in %.0f um pixels; SRF = rect(%d px) (x) LSF (x) pixel (x) Airy FWHM; CRF = LSF (x) pixel (x) Airy; EE = geometric\n', P.pixel_m*1e6, P.slit_px);
    pr('  fraction in one pixel about the centroid (min over the grid).  The FWHM window is +-8 px: a value >= 15 px is the WINDOW\n');
    pr('  (written ">15"), the rms blur (max over the grid, px) is quoted beside it.  Equal field weights.  Grooves chord-ruled.\n');
    pr('  Ring = slit offset from the axis, the smallest clearing the grating body by %+.0f mm at the seed (mount %.0f mm).\n', cl0, P.mount_margin_m*1e3);
    pr('  Sizes: clear = footprint + %.0f mm; blank = clear + %.0f mm; mass = blanks at %.0f mm x %.0f kg/m^3 (assumption, not lightweighted).\n\n', ...
        P.ap_margin_m*1e3, P.mount_margin_m*1e3, tb*1e3, rho);
    macos.init(P.model);
    rows = struct([]);
    for R = Rs
        Pk = base_(P, Fno, R, 0.22*R);
        % -- the ring: bisection on the seed's clearance -------------------
        cf = @(fr) clr_(setfield(Pk, 'y_slit', fr*R), P); %#ok<SFLD>
        a = 0.22;  b = 0.40;  ca = cf(a);  cb = cf(b);
        assert(cb > cl0, 'dyson5 off18: R %.2f -- even 0.40 R does not clear (%+.1f mm)', R, cb);
        if ca >= cl0, b = a; else
            for it = 1:12
                c = 0.5*(a + b);  cc = cf(c);
                if cc >= cl0, b = c; else, a = c; end
                if b - a < 2e-3, break; end
            end
        end
        fr = b;  Pk.y_slit = fr*R;
        pr('R %.2f m: ring %.3f R = %.1f mm (0.22 R reads %+.1f mm, blocked: the beams cross the grating)\n', R, fr, fr*R*1e3, ca);
        Pc = struct();
        for s = stp
            lbl = s{1};  Ps = Pk;  info = '';
            if any(strcmp(lbl, {'corr', 'free'}))
                Pq = P;  Pq.Fno_offner = Fno;  Pq.offner_R_m = R;  Pq.y_slit_offner_m = Pk.y_slit;
                if strcmp(lbl, 'corr')
                    O = offner_solve(Pq, sprintf('%s_R%03.0f', stem, R*100), 'max_iter', nit, 'quiet', true);
                    Pc = O.P;
                else
                    O = offner_solve(Pq, sprintf('%s_R%03.0f_free', stem, R*100), 'max_iter', nit, 'quiet', true, ...
                                     'ring_lb', fr*R, 'warm', Pc, 'wall_mm', cl0, 'zone_wall', true);
                end
                Ps = O.P;
                info = sprintf('ring %.3f R, convex x %.5f, zone-2 R x %.5f, dy %+.2f dz %+.2f mm; chain merit %.4g -> %.4g', ...
                    Ps.y_slit/R, Ps.offner_Rg_factor, Ps.offner_M3_factor, Ps.offner_M3_dy*1e3, Ps.offner_M3_dz*1e3, O.merit);
            end
            r = score_(Ps, P, sprintf('%s_R%03.0f_%s', stem, R*100, lbl), rho, tb);
            r.R = R;  r.ring = Ps.y_slit/R;  r.step = lbl;  r.info = info;  r.P = Ps;
            rows = [rows, r];  %#ok<AGROW>
            pr('  %-4s %s\n', lbl, info);
            pr('       engine: smile %.4f  keystone %.4f  CRF %s  SRF %s px  EE %.3f  rms blur %.2f/%.2f px (u/v)  rays/pt %d-%d, 0 vignetted\n', ...
                r.smile, r.keystone, fw_(r.crf), fw_(r.srf), r.ee, r.su, r.sv, r.nr(1), r.nr(2));
            pr('       chain (identity): smile %.4f  keystone %.4f  CRF %s  SRF %s  EE %.3f\n', r.ch.smile_max, r.ch.keystone_max, fw_(r.ch.crf_max), fw_(r.ch.srf_max), r.ch.ee_min);
            pr('       grating clear %.0f mm (stop: R/(2F) = %.0f mm); concave blank %.0f mm (zone gap %+.0f mm, zones %s); length %.0f mm; mass %.1f kg\n', ...
                r.gratD, R/(2*Fno)*1e3, r.concD, r.zgap, r.zone, r.len, r.kg);
            pr('       clearance %+.2f mm (%s vs %s) -- %s\n', r.clr, r.clr_leg, r.clr_body, tern_(r.clr >= 0, 'PASS', 'FAIL'));
        end
        save([stem '.mat'], 'rows', 'P');
    end
    % -- the two Dyson references, from the record (dyson5_size.mat) ---------
    D = load(fullfile(fileparts(mfilename('fullpath')), 'dyson5_size.mat'));  T = D.OUT.table;
    refs = {'F', 240, 'Dyson, CaF2 block, no meniscus'; 'A', 220, 'Dyson, silica block + 4 mm meniscus (R4)'};
    pr('\nTABLE (deck slide "Alternatives to the Dyson for the 3k slit"): 3000 px, 54 mm slit, 380-2500 nm, F/%.1f\n', Fno);
    pr('%-42s %6s %4s %5s %7s %7s %5s %5s %5s %6s %7s %6s  %s\n', 'form', 'R/r mm', 'F#', 'slit', 'smile', 'keyst', 'CRF', 'SRF', 'EE', 'len mm', 'max mm', 'kg', 'clearance worst pair');
    for k = 1:size(refs, 1)
        i = find(strcmp(T.family, refs{k,1}) & T.r_mm == refs{k,2} & strcmp(T.variant, 'solve'), 1);
        kg = T.edged_kg(i) + rho*tb*pi*(T.gratD_mm(i)/2e3 + P.mount_margin_m)^2;
        pr('%-42s %6.0f %4.1f %5.0f %7.4f %7.4f %5.2f %5.2f %5.2f %6.0f %7.0f %6.1f  %+.2f mm %s\n', refs{k,3}, refs{k,2}, P.Fno, 54, T.smile(i), T.keystone(i), ...
            T.CRF(i), T.SRF(i), T.EE(i), T.length_mm(i), max(T.gratD_mm(i), T.blockD_mm(i)), kg, T.clear_min_mm(i), T.clear_pair{i});
    end
    for r = rows
        pr('%-42s %6.0f %4.1f %5.0f %7.4f %7.4f %5s %5s %5.2f %6.0f %7.0f %6.1f  %+.2f mm %s vs %s\n', sprintf('Offner %s, ring %.2f R', r.step, r.ring), r.R*1e3, Fno, 54, ...
            r.smile, r.keystone, fw_(r.crf), fw_(r.srf), r.ee, r.len, max(r.gratD, r.concD), r.kg, r.clr, r.clr_leg, r.clr_body);
    end
    pr('  Dyson mass = the edged block (record) + its grating at the Offner''s blank rule; Offner mass = grating + concave blank(s).\n');
    fclose(fid);
    S.rows = rows;
    save([stem '.mat'], 'rows', 'P');
    fprintf('dyson5 off18: wrote %s.{txt,mat}\n', stem);
end

% =====================================================================
function Pk = base_(P, Fno, R, ys)
    Pk = struct('Fno', Fno, 'pixel_m', P.pixel_m, 'npix', P.npix, 'band_m', P.band_m, ...
                'lambda_ref_m', P.lambda_ref_m, 'order', P.order, 'y_slit', ys, 'offner_R', R, ...
                'block_r', P.block_r_m, 'glass', P.glass, 'face_offset', P.face_offset_m, 'Rg_factor', P.Rg_factor, ...
                'grating_model', 'planes', 'slit_px', P.slit_px, ...
                'offner_Rg_factor', 1, 'offner_M3_factor', 1, 'offner_M3_dy', 0, 'offner_M3_dz', 0);
end

function c = clr_(Pk, P)
    try
        C = spectrometer_clearance(spectrometer_geom('offner', Pk), P, 'quiet', true);  c = C.min_mm;
    catch
        c = -Inf;
    end
end

function r = score_(Ps, P, file_stem, rho, tb)
%SCORE_  Emit with apertures, load, verify no ray is vignetted, engine-score, gate, size.
    G = spectrometer_geom('offner', Ps);
    [d, nm] = fileparts(file_stem);
    M = spectrometer_rx(G, [file_stem '.in'], 'ngridpts', P.ngridpts, 'name', nm, 'apertures', true, 'margin', P.ap_margin_m);
    macos.load_rx([file_stem '.in']);
    assert(macos.num_elt() == M.nElt, 'dyson5 off18: %s loads %d of %d elements', nm, macos.num_elt(), M.nElt);
    macos.stop(M.iG);  macos.modify();
    tr = macos.trace(M.nElt);  ri = macos.get_ray_info(tr.nRays);
    nv = nnz(ri.ok_trace & ~ri.ok_pass);
    assert(nv == 0, 'dyson5 off18: %s -- %d rays vignetted by the declared apertures', nm, nv);
    Pe = P;  Pe.Fno = G.P.Fno;
    E = spectrometer_score(G, M, Pe, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
    Ch = spectrometer_score_chain(G, Ps, 'nx', P.score_nx, 'nlam', P.score_nlam);
    C = spectrometer_clearance(G, P, 'quiet', true);
    spectrometer_maps_fig(E, [file_stem '_maps.png'], 'title', [strrep(nm, '_', ' ') ', engine'], 'pixel_um', P.pixel_m*1e6);
    spectrometer_layout_fig(G, [file_stem '_layout.png'], 'title', strrep(nm, '_', ' '));
    % sizes from the clearance gate's own footprints (the declared apertures)
    F = C.footprints;  am = P.ap_margin_m;  mm = P.mount_margin_m;
    iG = G.iG;  gc = 2*(F(iG).radius + am);
    cen = @(k) G.surf(k).vpt(:) + F(k).xap*F(k).xc + F(k).yap*F(k).yc;
    c1 = cen(1);  c3 = cen(3);  r1 = F(1).radius + am;  r3 = F(3).radius + am;
    zgap = norm(c1 - c3) - r1 - r3;
    same = abs(Ps.offner_M3_factor - 1) < 1e-12 && Ps.offner_M3_dy == 0 && Ps.offner_M3_dz == 0;
    if same
        concD = norm(c1 - c3) + r1 + r3;  zone = 'one sphere, one blank';
        kgC = rho*tb*pi*(concD/2 + mm)^2;
    else
        concD = max(2*r1, 2*r3);
        if zgap < -0.5e-3, zone = 'OVERLAP: two figures on one area';
        elseif zgap < 0.5e-3, zone = 'touching (the zone wall''s hinge)';
        else, zone = 'separate'; end
        kgC = rho*tb*pi*((r1 + mm)^2 + (r3 + mm)^2);
    end
    Z = [];  for b = C.bodies, Z = [Z, b{1}.pts(3, :)]; end   %#ok<AGROW>
    Z = [Z, G.slit(3), G.fpa.center(3)];
    r = struct('file', [file_stem '.in'], 'smile', E.smile_max, 'keystone', E.keystone_max, 'crf', E.crf_max, 'srf', E.srf_max, ...
               'ee', E.ee_min, 'su', max(E.SU(:)), 'sv', max(E.SV(:)), 'nr', [min(E.nrays(:)) max(E.nrays(:))], ...
               'ch', Ch, 'E', E, 'gratD', gc*1e3, 'concD', concD*1e3, 'zgap', zgap*1e3, 'zone', zone, ...
               'len', (max(Z) - min(Z))*1e3, 'kg', kgC + rho*tb*pi*(gc/2 + mm)^2, ...
               'clr', C.min_mm, 'clr_leg', C.table.leg{1}, 'clr_body', C.table.body{1}, 'lpmm', G.grating.lines_per_mm);
    r.E = rmfield(E, {'raysU', 'raysV'});
end

function s = fw_(x), if x >= 15, s = '>15'; else, s = sprintf('%.2f', x); end, end
function v = field_(P, f, d), if isfield(P, f) && ~isempty(P.(f)), v = P.(f); else, v = d; end, end
function t = tern_(c, a, b), if c, t = a; else, t = b; end, end
function dp_(fid, varargin), fprintf(fid, varargin{:}); fprintf(varargin{:}); end
