function H = his_designs(map_n, opts)
%HIS_DESIGNS  Build Rodgers' OWN stage-3 and stage-4 solutions verbatim from
%   his parameters and strict-score them.  Pure evaluation -- exactly
%   the class of the stage-2 gate, with no optimizer anywhere in the loop.
%
%   H = HIS_DESIGNS()             map_n = 9, the committed EPD-4060 run
%                                 (slide parameters, 9x9 full box).
%   H = HIS_DESIGNS(N,'variant','seq')
%                                 the same evaluation at the CODE V .seq
%                                 TRUTH: EPD 5000, the M1 hole, his 15-point
%                                 half-box field set, full-precision conics
%                                 and rigid body.  Artifacts are suffixed
%                                 _seq; the committed _epd4060 set is not
%                                 touched.
%
%   Name-value:
%     'variant'  'epd4060' (default) | 'seq'
%     'Frel'     explicit (K,2) rad box-relative field set; default is the
%                variant's own (9x9 grid for epd4060, the .seq 15 for seq)
%     'base'     explicit base deck; default is the committed stage-2 deck
%                for epd4060, and a freshly BUILT one for seq
%
%   THE BAND.  The max/avg ratios this prints for rodgersS3 / rodgersS4,
%   together with strict_stage_table's S2, ARE the residual band.  Nothing
%   in here is tuned to it.
%
%   Construction, per stage:
%     1. load the committed rodgers1_epd4060_stage2.in -- his verbatim
%        layout, the +0.5 deg bias and EPD 4060 -- with its stale
%        realize_apertures clip apertures stripped (ApType=None);
%     2. set the three conics to HIS solved values (P.K_s3 / P.K_s4);
%     3. stage 4 only: apply HIS M2/M3 rigid body through the SAME engine
%        path CALIB uses -- cmd='PERTURB' / CPERTURB_PROG, reached from
%        MATLAB as macos.perturb (smacos_compute.inc:300-322).  The
%        applied values are then READ BACK and asserted against his
%        numbers in the SAME convention run_epd4060/rigid_of reports ours
%        (Ydec = Vpt(2), alpha = atan2d(psi(2), -psi(3))), so "his values"
%        means the same thing on both sides of the comparison;
%     4. fit the FPA by HIS procedure -- the align_focal_plane algorithm
%        (per-field best focus by least-squares closest point to the
%        arriving ray bundle, plane fit through the foci, 5x5 over +/-6')
%        -- and then FREEZE it;
%     5. strict-score the 9x9 box against that frozen FPA.
%
%   The fitted FPA is handed to STRICT_WFE_DECK as an explicit 'detector'
%   override.  That is exact: rays are straight after the last surface, so
%   the sphere construction is independent of where the deck's own
%   FocalPlane happens to sit.  The deck IS also updated to the fitted FPA
%   and saved as the artifact, and the two routes are cross-checked
%   against each other (they must agree to round-off).
%
%   Writes rodgers1_epd4060_rodgersS{3,4}.in, ..._his_designs.mat, and
%   rodgers1_epd4060_rodgersS{3,4}_strict.png.

    arguments
        map_n (1,1) double {mustBeInteger,mustBePositive} = 9
        opts.variant (1,:) char {mustBeMember(opts.variant,{'epd4060','seq'})} = 'epd4060'
        opts.Frel (:,2) double = zeros(0,2)
        opts.base (1,:) char = ''
        opts.hole (1,1) double = NaN    % override the M1 hole radius (m); 0 = none
        opts.suffix (1,:) char = ''     % extra filename suffix, for attribution runs
        opts.reference (1,:) char = ''  % '' = the variant's default
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    isseq = strcmpi(opts.variant,'seq');
    if isseq, P = rodgers_common('seq'); else, P = rodgers_common(); end
    lam_nm = P.lambda_m*1e9;
    if ~isnan(opts.hole), P.M1_hole_m = opts.hole; end
    tag    = [opts.variant opts.suffix];         % filename stem: _epd4060 | _seq[...]
    % REFERENCE (Dave 2026-07-31): the .seq-truth study defaults to the
    % CENTROID, per the ruling.  The epd4060 variant keeps 'strict-chief' so
    % the committed baselines reproduce bit-for-bit.  Either can be forced.
    ref = opts.reference;
    if isempty(ref)
        if isseq, ref = 'strict-centroid'; else, ref = 'strict-chief'; end
    end
    fprintf('  WFE reference: %s\n', ref);

    % ---- the base deck: his verbatim layout, the +0.5 deg bias, the EPD.
    % epd4060 = the committed deck (baseline, byte-for-byte).
    % seq     = BUILT here at the .seq truth, so the M1 hole and EPD 5000 are
    %           in the layout the scoring path actually traces.
    if ~isempty(opts.base)
        base = opts.base;
    elseif isseq
        base = build_base_deck_(P, here, tag);   % honours P.M1_hole_m
    else
        base = fullfile(here,'rodgers1_epd4060_stage2.in');
    end

    if ~isempty(opts.Frel)
        Frel = opts.Frel;
    elseif isseq
        Frel = P.seq.Frel;                       % his 15-point half box
    else
        Frel = macos.design.field_grid(P.fov_half_deg*60, map_n, 'units','arcmin');
    end
    H = struct('lambda_nm',lam_nm,'map_n',map_n,'stage',cell(1,3));

    cfg = { struct('st',3, 'K',P.K_s3, 'rb',[], 'tag','rodgersS3', ...
                   'gt',P.gt.s3_box, 'ref',[]), ...
            struct('st',4, 'K',P.K_s4, 'tag','rodgersS4', ...
                   'rb',[2 P.Ydec_M2_mm -P.tilt_M2_deg; ...
                         3 P.Ydec_M3_mm -P.tilt_M3_deg], ...
                   'gt',P.gt.s4_box, 'ref',[]) };
    % Stage 2 -- his verbatim conics, only the detector re-fitted -- is the
    % THIRD rung of the band.  At epd4060 it came from STRICT_STAGE_TABLE on
    % the committed deck (429.627 / 246.8 nm); running it here too would
    % rewrite that artifact, so it is added for the 'seq' variant only, where
    % it goes through the identical recipe as S3/S4 and the whole band is
    % produced by ONE path.
    if isseq
        cfg = [ { struct('st',2, 'K',P.K_nom, 'rb',[], 'tag','rodgersS2', ...
                         'gt',P.gt.s2_box, 'ref',[]) }, cfg ];
    end
    % ^ ADE SIGN: decoded by convention_decode.m, not chosen.  His ADE has the
    % OPPOSITE positive sense to rigid_of's alpha = atan2d(psi_y,-psi_z); his
    % YDE matches ours.  Of the 16 sign combinations x 2 application orders,
    % this one reads 1.911 um spot / 8.967 nm at the box centre where the
    % runner-up reads 52.7 um / 293 nm and everything else is mm-scale -- a
    % 30x separation, so it is a decode, not a fit.

    % ---- injection round-trip: OUR OWN stage-4 solve, read out of the
    % committed results and pushed back through the SAME perturb path.  If
    % this reproduces the committed stage-4 strict number (118.591 / 84.806
    % nm, PACKET Addendum 3 §D.2) then the injection machinery is sound and
    % any failure above is about the VALUES, not about the path.
    % (epd4060 only: the reference numbers 118.591/84.806 belong to that
    %  configuration.  At the .seq truth there is no committed solve to
    %  round-trip against yet, so the check is skipped rather than compared
    %  to a number from a different aperture.)
    if ~isseq
        Rc = load(fullfile(here,'rodgers1_epd4060_results.mat'));
        cfg{end+1} = struct('st',44, 'K',Rc.R.K_s4, 'tag','oursS4roundtrip', ...
                            'rb',[2 Rc.R.rigid(1,1) Rc.R.rigid(1,2); ...
                                  3 Rc.R.rigid(2,1) Rc.R.rigid(2,2)], ...
                            'gt',P.gt.s4_box, 'ref',[118.591 84.806]);
    end

    for c = 1:numel(cfg)
        C = cfg{c};
        if C.st == 44
            banner('ROUND-TRIP CHECK -- OUR committed stage-4 solve, re-injected');
        else
            banner('RODGERS'' OWN STAGE %d -- verbatim from his slides', C.st);
        end

        % ---- 1/2/3: his optics, in the engine -------------------------
        txt = regexprep(fileread(base), '(ApType=\s*)\S+', '$1None');
        work = [tempname '.in'];
        fid = fopen(work,'w');  fprintf(fid,'%s',txt);  fclose(fid);
        macos.init(P.model_size);  macos.load_rx(work);
        nE = macos.num_elt();

        for i = 1:3, macos.set_elt_kc(i, C.K(i)); end
        fprintf('  conics set: %.12f  %.12f  %.12f\n', C.K);

        if ~isempty(C.rb)
            for r = 1:size(C.rb,1)
                ie = C.rb(r,1);  dy = C.rb(r,2)*1e-3;  al = C.rb(r,3)*pi/180;
                % SIGN CONVENTION, determined by the readback assert below and
                % not by taste: macos.perturb's LOCAL +Rx reports back through
                % rigid_of's alpha = atan2d(psi_y, -psi_z) as NEGATIVE, so his
                % +alpha is applied as a -Rx rotation.  The translation needs
                % no flip (rotation is about RptElt = VptElt, so Vpt(2) = dy
                % exactly).
                macos.perturb(ie, 'rotation',[-al;0;0], 'translation',[0;dy;0]);
            end
        end
        macos.modify();

        % ---- readback in the reporting convention, ASSERTED -----------
        if ~isempty(C.rb)
            fprintf('  rigid-body readback (Ydec = Vpt(2), alpha = atan2d(psi_y,-psi_z));\n');
            fprintf('  "his" = his slide value CONVERTED to this frame (ADE sign flipped):\n');
            for r = 1:size(C.rb,1)
                ie = C.rb(r,1);
                v = macos.get_elt_vpt(ie);  p = macos.get_elt_psi(ie);
                yd = v(2)*1e3;  al = atan2d(p(2), -p(3));
                fprintf('    M%d  Ydec %10.6f mm (his %10.6f)   alpha %9.6f deg (his %9.6f)\n', ...
                        ie, yd, C.rb(r,2), al, C.rb(r,3));
                assert(abs(yd - C.rb(r,2)) < 1e-6 && abs(al - C.rb(r,3)) < 1e-6, ...
                    'his_designs:rigidBody', ...
                    ['M%d readback does not match his stated value -- the ' ...
                     'perturb convention differs from rigid_of''s, so "his ' ...
                     'design" would not mean what the comparison claims.'], ie);
            end
        end

        % ---- 4: FPA by his procedure, then frozen ---------------------
        macos.save_rx(work);
        wtxt = regexprep(fileread(work), '(ApType=\s*)\S+', '$1None');
        fpa  = align_fpa_deck_(wtxt, nE);
        fprintf('  FPA fit (his procedure): tilt %.4f deg vs arriving chief\n', fpa.tilt_deg);
        fprintf('    Vpt [%.6f %.6f %.6f]   psi [%.9f %.9f %.9f]\n', fpa.Vpt, fpa.psi);

        % write the artifact deck WITH the fitted FPA installed.
        % RELOAD `work` first: align_fpa_deck_ drove the engine through the
        % 5x5 probe grid and left it holding the LAST probe field's chief
        % ray.  Saving from that state bakes a box-corner field into the
        % deck, and every later scan is then centred 6' off.  (Caught by the
        % field map's x axis reading 0..12' instead of -6..+6'.)
        macos.load_rx(work);
        macos.set_elt_vpt(nE, fpa.Vpt(:));  macos.set_elt_psi(nE, fpa.psi(:));
        macos.set_elt_rpt(nE, fpa.Vpt(:));  macos.modify();
        deck = fullfile(here, sprintf('rodgers1_%s_%s.in', tag, C.tag));
        macos.save_rx(deck);
        % the saved deck must still point at the nominal field
        dchk = regexp(fileread(deck),'ChfRayDir=\s*([^\n]*)','tokens','once');
        bchk = regexp(wtxt,          'ChfRayDir=\s*([^\n]*)','tokens','once');
        assert(max(abs(sscanf(dchk{1},'%f',3) - sscanf(bchk{1},'%f',3))) < 1e-14, ...
            'his_designs:chiefRay', ...
            'artifact deck chief ray drifted from nominal -- the scanned box would be offset.');

        % ---- 5: strict score ------------------------------------------
        s  = strict_wfe_deck(deck, Frel, 'reference', ref, 'detector', ...
                             struct('Vpt',fpa.Vpt,'psi',fpa.psi));
        s2 = strict_wfe_deck(deck, Frel, 'reference', ref);   % deck's own FP
        w  = s.wfe_m(isfinite(s.wfe_m))*1e9;
        w2 = s2.wfe_m(isfinite(s2.wfe_m))*1e9;
        fprintf('  detector-override vs deck-FP cross-check: %.3e relative on max\n', ...
                abs(max(w)-max(w2))/max(w));

        H(c).stage = C.st;  H(c).scan = s;  H(c).fpa = fpa;  H(c).K = C.K;
        H(c).rb = C.rb;     H(c).gt = C.gt; H(c).deck = deck;
        H(c).min = min(w);  H(c).max = max(w);  H(c).avg = mean(w);
        bf = bestfoc_(deck, Frel, fpa);
        H(c).bestfoc = bf;
        fprintf('  best-focus rung (the floor ANY detector surface can reach):\n');
        fprintf('              min %9.3f  max %9.3f  avg %9.3f nm   (focus shift %.3f..%.3f mm)\n', ...
                min(bf.wfe_nm), max(bf.wfe_nm), mean(bf.wfe_nm), min(bf.dfoc_mm), max(bf.dfoc_mm));
        H(c).nfin = numel(w); H(c).ntot = numel(s.wfe_m);
        fprintf('  rays/field  : %d..%d   fields %d/%d finite\n', ...
                min(s.nrays), max(s.nrays), H(c).nfin, H(c).ntot);
        fprintf('  STRICT (nm) : min %9.3f  max %9.3f  avg %9.3f\n', ...
                H(c).min, H(c).max, H(c).avg);
        fprintf('  Rodgers (nm): min %9.3f  max %9.3f  avg %9.3f\n', ...
                C.gt(1)*lam_nm, C.gt(2)*lam_nm, C.gt(3)*lam_nm);
        fprintf('  ratio       : max %.3f x   avg %.3f x\n', ...
                H(c).max/(C.gt(2)*lam_nm), H(c).avg/(C.gt(3)*lam_nm));
        H(c).tag = C.tag;  H(c).ref = C.ref;
        if ~isempty(C.ref)
            fprintf('  ROUND-TRIP vs committed §D.2 (%.3f / %.3f nm): %.3e / %.3e relative\n', ...
                    C.ref(1), C.ref(2), abs(H(c).max-C.ref(1))/C.ref(1), ...
                    abs(H(c).avg-C.ref(2))/C.ref(2));
        end
        delete(work);
    end

    % ---- maps (same rendering as the committed set) --------------------
    trend = renderer_(P);
    for c = 1:numel(H)
        s = H(c).scan;
        png = fullfile(here, sprintf('rodgers1_%s_%s_strict.png', tag, H(c).tag));
        scan = struct('fields', s.fields*180/pi*60 + s.bias_deg*60, ...
                      'wfe', s.wfe(:), 'metric','strict');
        fig = trend.view_field_map(scan,'kind','contour','save',png,'visible',false);
        close(fig);  H(c).png = png;
    end

    banner('HIS OWN SOLVES, STRICT-SCORED  (%s: EPD %g, nm @ %g nm, %d fields)', tag, P.EPD_mm, lam_nm, size(Frel,1));
    fprintf('  %-16s | %-19s | %-15s | %8s | %8s\n','design','strict max/avg','Rodgers max/avg','max x','avg x');
    for c = 1:numel(H)
        g = H(c).gt;
        fprintf('  %-16s | %9.1f/%-9.1f | %7.1f/%-7.1f | %8.2f | %8.2f\n', ...
            H(c).tag, H(c).max, H(c).avg, g(2)*lam_nm, g(3)*lam_nm, ...
            H(c).max/(g(2)*lam_nm), H(c).avg/(g(3)*lam_nm));
    end
    save(fullfile(here,sprintf('rodgers1_%s_his_designs.mat',tag)),'H');
    fprintf('\nsaved rodgers1_%s_his_designs.mat + decks + maps\n', tag);
end

% =====================================================================
function bf = bestfoc_(deck, Frel, fpa)
%BESTFOC_  Per-field, slide the sphere centre along the chief to that
%   field's own best focus.  The residual is the floor ANY detector
%   surface -- plane or curved -- can reach, so it separates "the FPA fit
%   is wrong" from "the design is what it is".
    txt = regexprep(fileread(deck), '(ApType=\s*)\S+', '$1None');
    [cdir0, cpos0, apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    Nd = fpa.psi(:)/norm(fpa.psi);  Vd = fpa.Vpt(:);
    tmp = [tempname '.in'];
    K = size(Frel,1);
    bf = struct('wfe_nm',nan(K,1),'dfoc_mm',nan(K,1));
    for k = 1:K
        emit_field_(txt, tmp, apst, stand, bx0+Frel(k,1), by0+Frel(k,2));
        macos.load_rx(tmp);  nE = macos.num_elt();
        s = macos.trace(nE);  ri = macos.get_ray_info(s.nRays);
        ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
        c0 = p1 + d1*(dot(Nd, Vd - p1)/dot(Nd, d1));
        emit_field_(txt, tmp, apst, stand, bx0+Frel(k,1)+1e-5, by0+Frel(k,2));
        macos.load_rx(tmp);  macos.trace(nE);
        rp = macos.get_ray_info(s.nRays);
        X = fex_cross_(p1, d1, rp.pos(:,1), rp.dir(:,1));
        P = ri.pos(:,ok);  D = ri.dir(:,ok);  L = ri.opl(ok);
        f = @(u) std(strict_sphere_opl(P, D, L, c0 + d1*u, norm(X - (c0 + d1*u))));
        u = fminbnd(f, -0.5, 0.5, optimset('TolX',1e-9));
        bf.wfe_nm(k) = f(u)*1e9;  bf.dfoc_mm(k) = u*1e3;
    end
    delete(tmp);
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function fpa = align_fpa_deck_(txt, nE)
%ALIGN_FPA_DECK_  align_focal_plane's algorithm (Telescope.m:390-431) run
%   on a deck: per-field best focus = least-squares closest point to the
%   arriving ray bundle; plane fit through the foci; FP psi oriented along
%   the arriving chief.  grid = 5, span = 6 arcmin -- the committed recipe.
    F = macos.design.field_grid(6, 5, 'units','arcmin');
    F = F(any(abs(F) > 1e-12, 2), :);
    F = [0 0; F];
    nF = size(F,1);  foci = zeros(3,nF);  dch = zeros(3,1);
    [cdir0, cpos0, apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    tmp = [tempname '.in'];
    for f = 1:nF
        emit_field_(txt, tmp, apst, stand, bx0+F(f,1), by0+F(f,2));
        macos.load_rx(tmp);
        s = macos.trace(nE-1);  a = macos.get_ray_info(s.nRays);
        macos.trace(nE);        b = macos.get_ray_info(s.nRays);
        ok = a.ok_trace(:) & a.ok_pass(:) & b.ok_trace(:) & b.ok_pass(:);
        Pb = b.pos(:,ok);
        D  = b.pos(:,ok) - a.pos(:,ok);  D = D ./ vecnorm(D);
        A  = nnz(ok)*eye(3) - D*D.';
        bb = sum(Pb,2) - D*sum(D.*Pb,1).';
        foci(:,f) = A \ bb;
        if f == 1, dch = mean(D,2);  dch = dch/norm(dch); end
    end
    delete(tmp);
    C = mean(foci,2);
    [U,~,~] = svd(foci - C, 'econ');
    nrmv = U(:,3);  nrmv = nrmv * sign(dot(nrmv, dch));
    Vnew = foci(:,1) - nrmv*dot(nrmv, foci(:,1)-C);
    fpa = struct('Vpt',Vnew.', 'psi',nrmv.', ...
                 'tilt_deg', acosd(min(1,abs(dot(nrmv,dch)))), ...
                 'sag_m', (nrmv.'*(foci - C)), 'foci', foci);
end

function emit_field_(txt, tmp, apst, stand, bx, by)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3(cpos)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function [cdir, cpos, apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');
    cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function base = build_base_deck_(P, here, tag)
%BUILD_BASE_DECK_  His verbatim layout at THIS variant's configuration, with
%   the FPA fitted by his procedure -- the stage-2 deck, built rather than
%   read.  Used by the 'seq' variant, where the committed EPD-4060 deck is
%   the wrong aperture and carries no M1 hole.
%
%   The M1 central perforation (CIR HOL, semi-diameter 513.9422766474837 mm)
%   is emitted as a REAL circular obscuration via set_hole, so the traced
%   pupil is the annulus CODE V traces, not a filled disc.
    t = macos.design.Telescope('family','TMA', ...
            'aperture_diameter_mm', P.EPD_mm, ...
            'wavelength_m', P.lambda_m, 'model_size', P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1), ...
                 'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2), ...
                 'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3), ...
                 'spacing_after','derive');
    if isfield(P,'M1_hole_m') && P.M1_hole_m > 0
        t.set_hole('M1', P.M1_hole_m);
    end
    t.set_field_bias(P.offset_deg*60);
    t.build();
    t.align_focal_plane('grid',5,'span_arcmin',6);
    base = fullfile(here, sprintf('rodgers1_%s_stage2.in', tag));
    t.save(base);
    fprintf('  base deck built at the %s configuration -> %s\n', tag, base);
    fprintf('    EPD %.1f mm, M1 hole semi %.4f mm, bias +%.3f deg\n', ...
            P.EPD_mm, 1e3*getfielddef_(P,'M1_hole_m',0), P.offset_deg);
end

function v = getfielddef_(s, f, d)
    if isfield(s,f), v = s.(f); else, v = d; end
end

function t = renderer_(P)
    t = macos.design.Telescope('family','TMA','aperture_diameter_mm',P.EPD_mm, ...
            'wavelength_m',P.lambda_m,'model_size',P.model_size);
    t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1),'spacing_after_mm',abs(P.s12_mm));
    t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2),'spacing_after_mm',abs(P.s23_mm));
    t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3),'spacing_after','derive');
    t.build();
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
