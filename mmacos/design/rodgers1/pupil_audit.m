function A = pupil_audit(opts)
%PUPIL_AUDIT  Is MACOS tracing the pupil the prescription declares?  No.
%
%   A = PUPIL_AUDIT()  measures the traced entrance pupil FIVE independent
%   ways on the .seq-truth decks, then re-scores Rodgers' three designs with
%   the rays MASKED to the pupil the prescription actually declares.
%
%   THE FINDING (Dave, 2026-08-01 -- "someone changed the code").
%   SOURCSUB.F:220, in COLSOURCE (the collimated-source ray-grid layout):
%
%       !ToDo: defining aperture for the sake of plotting is wrong
%       !      => what is the correct physics
%       !A2=(Aperture/2d0)**2  ! original
%       A2= (Aperture/2d0 + Aperture/DBLE(npts))**2 ! for plotting a 'bottom' ray
%
%   The circular-grid ACCEPTANCE RADIUS is Aperture/2 + Aperture/npts, not
%   Aperture/2.  With npts = nGridpts-1 (iosub.inc:240, smacosio.F:277/304,
%   macosio.F:369/481/1322/1962) the traced pupil radius exceeds the declared
%   one by
%
%       1 + 2/(nGridpts - 1)
%
%   = +5.0% here (nGridpts=41 -> npts=40), +0.78% at nGridpts=257.  The
%   correct form sits COMMENTED OUT one line above, labelled "! original",
%   and PTSOURCE (sourcsub.F:731) uses A2=(Aperture*0.50001)**2 -- correct.
%   So a COLLIMATED and a POINT source trace DIFFERENT pupils for the same
%   Aperture=.
%
%   WHY IT HID FOR SO LONG.  The traced pupil is a disc of radius
%   Aperture/2 + Aperture/npts CLIPPED by the lattice square |x|,|y| <=
%   Aperture/2 -- a square with rounded corners.  Measured ALONG EITHER AXIS
%   it reads EXACTLY the declared aperture; only the DIAGONAL is oversize.
%   A span check, or a plot, reports 5000.0000 mm and looks right.
%
%   Sections (choose with 'sections'):
%     0  the five-way pupil measurement -- source formula, Elt-1 SPOT chord,
%        ray positions at M1, ray counts, radial histogram
%     1  the MASKING HARNESS: the reference ladder computed twice per field,
%        once on the engine's pupil and once on the TRUE declared pupil
%
%   Name-value:
%     'decks'    cellstr of decks to score (default: the three committed
%                .seq-truth decks rodgers1_seq_rodgersS{2,3,4}.in)
%     'sections' subset of [0 1] (default both)
%     'Rtrue_mm' the true pupil semi-diameter to mask to (default EPD/2)
%     'save'     write rodgers1_pupil_audit.mat (default true)
%
%   HEADLINE.  Masking to the declared 5000 mm pupil and granting the one
%   reference convention CODE V's field-map RMS uses (per-field best focus +
%   least-squares tip/tilt) reproduces Rodgers' reported MAX on all three of
%   his designs to 0.0% / 0.8% / 0.1%.  See PACKET.md Addendum 10.
%
%   NOTE this does NOT change the engine.  Masking is done in MATLAB, on the
%   rays the engine returns, so the comparison isolates the pupil and nothing
%   else -- same optics, same detector, same fields, same rays otherwise.

    arguments
        opts.decks    (1,:) cell   = {}
        opts.sections (1,:) double = [0 1]
        opts.Rtrue_mm (1,1) double = NaN
        opts.save     (1,1) logical = true
    end
    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);

    P = rodgers_common('seq');
    Frel  = P.seq.Frel;
    Rtrue = opts.Rtrue_mm;  if isnan(Rtrue), Rtrue = P.EPD_mm/2; end
    decks = opts.decks;
    if isempty(decks)
        decks = { fullfile(here,'rodgers1_seq_rodgersS2.in')
                  fullfile(here,'rodgers1_seq_rodgersS3.in')
                  fullfile(here,'rodgers1_seq_rodgersS4.in') };
    end
    gts = { P.gt.s2_box, P.gt.s3_box, P.gt.s4_box };
    tags= { 'rodgersS2','rodgersS3','rodgersS4' };

    A = struct('EPD_mm',P.EPD_mm,'Rtrue_mm',Rtrue,'nGridpts',NaN, ...
               'npts',NaN,'lattice_mm',NaN,'Racc_mm',NaN, ...
               'measure',struct(),'stage',struct([]));
    macos.init(P.model_size);

    % =================================================================
    % SECTION 0 -- the pupil, measured five ways
    % =================================================================
    if any(opts.sections == 0)
        txt = strip_ap_(fileread(decks{1}));
        [cdir0,cpos0,apst] = deck_src_(txt);
        stand = dot(apst - cpos0, cdir0);
        nG = sscanf(char(regexp(txt,'nGridpts=\s*(\S+)','tokens','once')),'%d');
        Ap = sscanf(strrep(char(regexp(txt,'Aperture=\s*(\S+)','tokens','once')),'D','E'),'%f');
        np = nG - 1;                                % iosub.inc:240
        Racc = Ap/2 + Ap/np;                        % sourcsub.F:220
        A.nGridpts = nG;  A.npts = np;
        A.lattice_mm = Ap/(np-1)*1e3;  A.Racc_mm = Racc*1e3;

        fprintf('\n#################### THE PUPIL, MEASURED FIVE WAYS ####################\n');
        fprintf('  deck            %s\n', decks{1});
        fprintf('  DECLARED        Aperture = %.4f mm   (semi %.4f mm)\n', Ap*1e3, Ap*1e3/2);
        fprintf('  nGridpts %d  ->  npts = nGridpts-1 = %d  ->  lattice pitch Aperture/(npts-1) = %.6f mm\n', ...
                nG, np, A.lattice_mm);
        fprintf('\n  (1) SOURCE FORMULA  sourcsub.F:220  A2 = (Aperture/2 + Aperture/npts)^2\n');
        fprintf('      acceptance radius   %10.4f mm   -> diameter %10.4f mm  = %.5f x declared\n', ...
                A.Racc_mm, 2*A.Racc_mm, 2*A.Racc_mm/(Ap*1e3));

        tmp = [tempname '.in'];
        cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>
        kc  = 2;                                     % box centre
        [ri,nE] = trace_field_(txt,tmp,apst,stand, asin(cdir0(1))+Frel(kc,1), ...
                                                   asin(cdir0(2))+Frel(kc,2));

        % --- (2) the SPOT at Elt 1: the distance across the pupil ---------
        sp = macos.spot(1,'ref','telt','at','elt','reset_trace',false);
        x = sp.pts(:,1)*1e3;  y = sp.pts(:,2)*1e3;
        Dmax = 0;
        for a = 1:numel(x), Dmax = max(Dmax, max(hypot(x-x(a), y-y(a)))); end
        fprintf('\n  (2) SPOT AT ELT 1 (= the stop = the entrance pupil), %d rays\n', sp.nSpot);
        fprintf('      x span (ALONG THE AXIS)            %10.4f mm  = %.5f x declared  <-- looks right\n', ...
                max(x)-min(x), (max(x)-min(x))/(Ap*1e3));
        fprintf('      y span                             %10.4f mm  = %.5f x declared\n', ...
                max(y)-min(y), (max(y)-min(y))/(Ap*1e3));
        fprintf('      GREATEST POINT-TO-POINT DISTANCE   %10.4f mm  = %.5f x declared  <-- the diagonal\n', ...
                Dmax, Dmax/(Ap*1e3));
        A.measure.spot_xspan_mm = max(x)-min(x);
        A.measure.spot_yspan_mm = max(y)-min(y);
        A.measure.spot_chord_mm = Dmax;
        A.measure.spot_nSpot    = sp.nSpot;

        % --- (3) ray positions at M1 (a different engine path) -----------
        macos.trace(1);  rm = macos.get_ray_info(numel(ri.opl));
        pm  = rm.pos - rm.pos(:,1);
        rho = sqrt(sum(pm(1:2,:).^2,1)).'*1e3;
        okA = rm.ok_trace(:);
        okP = rm.ok_trace(:) & rm.ok_pass(:);
        fprintf('\n  (3) RAY POSITIONS AT M1 (macos.trace(1) + get_ray_info)\n');
        fprintf('      outermost ray radius               %10.4f mm  = %.5f x declared semi\n', ...
                max(rho(okP)), max(rho(okP))/(Ap*1e3/2));
        fprintf('      innermost passing ray radius       %10.4f mm  (declared hole semi %.4f mm)\n', ...
                min(rho(okP)), P.M1_hole_m*1e3);
        A.measure.r_max_mm = max(rho(okP));
        A.measure.r_min_mm = min(rho(okP));

        % --- (4) ray COUNTS -> the implied grid radius --------------------
        nOut = nnz(okP & rho > Rtrue);
        nIn  = nnz(okP & rho <= Rtrue);
        f    = nOut/(nOut+nIn);
        Rimp = sqrt((Rtrue^2 - f*(P.M1_hole_m*1e3)^2)/(1-f));
        fprintf('\n  (4) RAY COUNTS\n');
        fprintf('      launched %d   passing %d   obscured %d\n', ...
                numel(rho), nnz(okP), nnz(okA)-nnz(okP));
        fprintf('      OUTSIDE the declared %.1f mm pupil   %d of %d  = %.2f %%\n', ...
                2*Rtrue, nOut, nOut+nIn, 100*f);
        fprintf('      area-implied grid radius           %10.4f mm  = %.5f x declared semi\n', ...
                Rimp, Rimp/(Ap*1e3/2));
        A.measure.n_launched = numel(rho);
        A.measure.n_passing  = nnz(okP);
        A.measure.n_outside  = nOut;
        A.measure.frac_outside = f;
        A.measure.R_implied_mm = Rimp;

        % --- (5) the radial histogram ------------------------------------
        edges = 0:125:2750;
        h = histcounts(rho(okP), edges);
        fprintf('\n  (5) RADIAL HISTOGRAM of the passing rays, 125 mm bins 0..2750 mm\n      ');
        fprintf('%d ', h);
        fprintf('\n      the last populated bin lies BEYOND the declared pupil edge (%.1f mm).\n', Rtrue);
        A.measure.hist_edges_mm = edges;
        A.measure.hist = h;

        fprintf(['\n  SHAPE.  Disc of radius %.1f mm CLIPPED by the lattice square\n' ...
                 '  |x|,|y| <= %.1f mm: exact on the axes, %.2f %% oversize on the diagonals.\n'], ...
                A.Racc_mm, Ap*1e3/2, 100*(Dmax/(Ap*1e3)-1));
    end

    % =================================================================
    % SECTION 1 -- the masking harness
    % =================================================================
    if any(opts.sections == 1)
        fprintf('\n#################### MASKING HARNESS: ENGINE PUPIL vs TRUE PUPIL ####################\n');
        fprintf('  mask = keep rays whose radius at M1 (about the chief intercept) is <= %.4f mm\n', Rtrue);
        rung = {'strict-chief','strict-centroid','bestfoc','bestfoc+LStilt'};
        for c = 1:numel(decks)
            L = ladder_both_(decks{c}, Frel, Rtrue*1e-3);
            g = gts{c}*1e3;
            fprintf('\n  === %s   (CODE V reported: max %.2f  avg %.2f nm) ===\n', tags{c}, g(2), g(3));
            fprintf('  %-16s | %-27s | %-27s\n','rung','ENGINE pupil','TRUE declared pupil');
            fprintf('  %-16s | %8s %8s %8s | %8s %8s %8s\n','','max','avg','max x','max','avg','max x');
            for r = 1:4
                a = L(:,r,1)*1e9;  b = L(:,r,2)*1e9;
                fprintf('  %-16s | %8.2f %8.2f %8.3f | %8.2f %8.2f %8.3f\n', rung{r}, ...
                        max(a),mean(a),max(a)/g(2), max(b),mean(b),max(b)/g(2));
            end
            A.stage(c).tag = tags{c};  A.stage(c).deck = decks{c};
            A.stage(c).gt_nm = g;      A.stage(c).rungs = rung;
            A.stage(c).L = L;          A.stage(c).fields = Frel;
        end
        fprintf(['\n  READING.  The last row, right-hand block, is the comparison:\n' ...
                 '  the declared pupil + CODE V''s own reference convention (per-field best\n' ...
                 '  focus + least-squares tip/tilt).  Nothing is tuned to it.\n']);
    end

    if opts.save
        out = fullfile(here,'rodgers1_pupil_audit.mat');
        save(out,'A');
        fprintf('\n  saved %s\n', out);
    end
end

% =====================================================================
function L = ladder_both_(deck, Frel, Rtrue_m)
%LADDER_BOTH_  Per field, the 4-rung reference ladder computed TWICE: once on
%   every ray the engine returns, once on the rays inside the declared pupil.
%   Returns L(field, rung, pupil) in metres; pupil 1 = engine, 2 = true.
    txt = strip_ap_(fileread(deck));
    [cdir0,cpos0,apst] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));  by0 = asin(cdir0(2));
    [Vd, Nd] = deck_fp_(txt);  Nd = Nd/norm(Nd);
    K = size(Frel,1);  L = nan(K,4,2);
    tmp = [tempname '.in'];  cu = onCleanup(@() del_(tmp)); %#ok<NASGU>
    for k = 1:K
        [ri,nE] = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1), by0+Frel(k,2));
        p1 = ri.pos(:,1);  d1 = ri.dir(:,1);
        % pupil radius at M1 -- the SAME trace, stepped back to element 1
        macos.trace(1);   rm = macos.get_ray_info(numel(ri.opl));
        pm  = rm.pos - rm.pos(:,1);
        rho = sqrt(sum(pm(1:2,:).^2,1)).';
        macos.trace(nE);  ri = macos.get_ray_info(numel(ri.opl));
        % exit pupil, FEX's two-probe chief-ray crossing
        rp = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1)+1e-5, by0+Frel(k,2));
        X  = fex_cross_(p1,d1,rp.pos(:,1),rp.dir(:,1));
        ri = trace_field_(txt,tmp,apst,stand, bx0+Frel(k,1), by0+Frel(k,2));
        base = ri.ok_trace(:) & ri.ok_pass(:);  base(1) = false;   % engine OPD skips the chief
        for pu = 1:2
            ok = base;  if pu == 2, ok = ok & (rho <= Rtrue_m); end
            if nnz(ok) < 10, continue; end
            L(k,:,pu) = rungs_(ri.pos(:,ok), ri.dir(:,ok), ri.opl(ok), p1,d1,Vd,Nd,X);
        end
    end
end

function v = rungs_(Pp,Dd,Ll,p1,d1,Vd,Nd,X)
%RUNGS_  The four reference-freedom rungs, on whatever ray subset is handed in.
    f = strict_refs(Pp,Dd,Ll,p1,d1,Vd,Nd,X);
    v = nan(1,4);
    v(1) = f.wfe_chief;                       % Dave's original ruling
    v(2) = f.wfe_centroid;                    % the 2026-07-31 ruling
    e3 = d1(:)/norm(d1(:));
    e1 = [1;0;0] - e3*dot([1;0;0],e3);
    if norm(e1) < 1e-8, e1 = [0;1;0] - e3*dot([0;1;0],e3); end
    e1 = e1/norm(e1);  e2 = cross(e3,e1);
    tp = (e3.'*(X(:)-Pp))./(e3.'*Dd);  Q = Pp + Dd.*tp;
    px = (e1.'*(Q-X(:))).';  py = (e2.'*(Q-X(:))).';
    c0 = f.c_centroid;  R0 = f.R_centroid;
    ff = @(u) std(strict_sphere_opl(Pp,Dd,Ll, c0+e3*u, R0+u));
    u  = fminbnd(ff, -0.05, 0.05);            % per-field best focus
    v(3) = ff(u);
    W  = strict_sphere_opl(Pp,Dd,Ll, c0+e3*u, R0+u);
    Am = [ones(numel(px),1), px(:), py(:)];
    v(4) = std(W(:) - Am*(Am\W(:)));          % + least-squares tip/tilt
end

% ---- deck helpers (mirrors of STRICT_WFE_DECK's, kept local so this
%      diagnostic is self-contained) ---------------------------------------
function s = strip_ap_(txt),  s = regexprep(txt,'(ApType=\s*)\S+','$1None');  end

function [cdir,cpos,apst] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');  cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
end

function [V,N] = deck_fp_(txt)
    Vs = grabAll3_(txt,'VptElt');  Ps = grabAll3_(txt,'psiElt');
    V = Vs(:,end);  N = Ps(:,end);
end

function v = grab3_(txt,key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function M = grabAll3_(txt,key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens');
    M = zeros(3,numel(t));
    for i = 1:numel(t), M(:,i) = sscanf(strrep(t{i}{1},'D','E'),'%f',3); end
end

function [ri,nE] = trace_field_(txt,tmp,apst,stand,bx,by)
    cdir = [sin(bx); sin(by); sqrt(max(0,1-sin(bx)^2-sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt,'(ChfRayDir=\s*)[^\n]*',['$1' v3_(cdir)]);
    s = regexprep(s,  '(ChfRayPos=\s*)[^\n]*',['$1' v3_(cpos)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
    macos.load_rx(tmp);
    nE = macos.num_elt();
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1-p2;  b = dot(d1,d2);  den = 1-b^2;
    if abs(den) < 1e-14, X = p1; return; end
    X = 0.5*((p1 + d1*((b*dot(d2,w0)-dot(d1,w0))/den)) + ...
             (p2 + d2*((dot(d2,w0)-b*dot(d1,w0))/den)));
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
