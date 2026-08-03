function out = afocal_score_psf(deck, Frel, opts)
%AFOCAL_SCORE_PSF  Score an AFOCAL deck with a focus, without touching it.
%
%   An afocal system has no image, so every image-plane tool -- PSF, Strehl
%   from a spot, DESIGN_REPORT, the focal STRICT_* ladder -- has nothing to
%   work on.  This harness appends an IDEAL LENS behind the interface pupil
%   in a SEPARATE scoring deck and scores there.  The delivered afocal `.in`
%   is never modified: it is the deliverable, and a deliverable with a
%   scoring lens welded to its back end is not the design.
%
%   out = AFOCAL_SCORE_PSF(DECK, FREL) scores DECK over the box-relative
%   field offsets FREL (K x 2, rad).
%
%   THE LENS IS RE-POINTED PER FIELD, and that is the whole design of this
%   harness.  A singlet fixed on the box-centre chief would meet the 0.25 deg
%   edge field at 30 x 0.25 = 7.5 degrees off ITS OWN axis and contribute
%   several waves of its own coma -- the score would then be of the lens, not
%   of the telescope.  Re-pointed, the lens works at its infinite conjugate
%   on every field, where the K = -n^2 conic is stigmatic to ~1e-14, so what
%   is measured is the telescope's exit wavefront and nothing else.
%   TAFOCALKERNEL gates that: the focal ladder through the lens returns the
%   AFOCAL ladder's own rungs.
%
%   Name-value:
%     'ref_elt'    interface-pupil element the lens is placed behind
%                  (default: the last element)
%     'f'          lens focal length, m.  Default gives f/20 on the measured
%                  exit beam -- slow enough that the singlet's own residual
%                  is far below anything being measured.
%     'D'          lens clear diameter, m (default 1.3x the exit beam)
%     'n'          glass index (1.5)
%     'standoff'   lens front vertex behind the interface pupil, m
%                  (default 2x the exit beam diameter)
%     'strip_ap'   rewrite every ApType= to None (true)
%     'model_size' (256)   'init' call macos.init first (true)
%     'save'       path STEM; when given, each field's scoring deck is kept
%                  as <stem>_f<k>.in for DESIGN_REPORT / view_rx.  Default
%                  '' = temporary files, deleted.
%     'quiet'      (true)
%
%   Returns OUT with .rungs (K x 4, metres -- the focal STRICT_RUNGS ladder
%   through the lens), .strehl (K x 4), .lens (the ideal_lens spec), .fp
%   (K x 3 fitted focal-plane vertices), .decks (paths, when saved),
%   .exit_beam_m, .fno.
%
%   See also AFOCAL_LADDER_DECK, STRICT_LADDER_DECK, MACOS.DESIGN.IDEAL_LENS.

    arguments
        deck (1,:) char
        Frel (:,2) double
        opts.ref_elt    (1,1) double  = 0
        opts.f          (1,1) double  = 0
        opts.D          (1,1) double  = 0
        opts.n          (1,1) double  = 1.5
        opts.standoff   (1,1) double  = 0
        opts.strip_ap   (1,1) logical = true
        opts.model_size (1,1) double  = 256
        opts.init       (1,1) logical = true
        opts.save       (1,:) char    = ''
        opts.quiet      (1,1) logical = true
    end

    txt = fileread(deck);
    if opts.strip_ap
        txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
    end
    [cdir0, cpos0, apst, lam] = deck_src_(txt);
    stand = dot(apst - cpos0, cdir0);
    bx0 = asin(cdir0(1));   by0 = asin(cdir0(2));
    nElt0 = str2double(regexp(txt,'nElt=\s*(\d+)','tokens','once'));
    ie = opts.ref_elt;   if ie <= 0, ie = nElt0; end

    if opts.init, macos.init(opts.model_size); end

    tmp = [tempname '.in'];
    cu  = onCleanup(@() del_(tmp)); %#ok<NASGU>

    % ---- size the lens from the box-centre exit beam ----------------------
    [~, kc] = min(sum(Frel.^2, 2));
    r0 = exit_beam_(txt, tmp, apst, stand, bx0+Frel(kc,1), by0+Frel(kc,2), ie);
    Dl = opts.D;   if Dl <= 0, Dl = 2.6*r0; end
    fl = opts.f;   if fl <= 0, fl = 20*(2*r0); end
    so = opts.standoff;  if so <= 0, so = 4*r0; end
    L = macos.design.ideal_lens(fl, Dl, 'n', opts.n, 'type','conic', ...
                                'mode','focus', 'name','scorelens');

    K = size(Frel,1);
    out = struct('deck',deck, 'fields',Frel, 'lambda',lam, 'ref_elt',ie, ...
                 'lens',L, 'exit_beam_m',2*r0, 'fno',fl/(2*r0), ...
                 'rungs',nan(K,4), 'strehl',nan(K,4), 'fp',nan(K,3), ...
                 'decks',{cell(1,K)}, 'standoff',so);

    for k = 1:K
        bx = bx0 + Frel(k,1);   by = by0 + Frel(k,2);
        [p1, d1] = exit_chief_(txt, tmp, apst, stand, bx, by, ie);
        vf = p1 + d1*so;                                % lens front vertex
        sd = build_score_deck_(txt, L, nElt0, vf, d1, Dl, bx, by, apst, stand);
        if isempty(opts.save)
            path = [tempname '.in'];
        else
            path = sprintf('%s_f%d.in', opts.save, k);
        end
        fid = fopen(path,'w');  fprintf(fid,'%s',sd);  fclose(fid);
        % focal plane: seed at the powered vertex + f, then FIT it
        zfp = fit_focus_(path, vf + d1*(L.thickness + fl), d1);
        sd  = set_last_vpt_(fileread(path), zfp, d1);
        fid = fopen(path,'w');  fprintf(fid,'%s',sd);  fclose(fid);
        out.fp(k,:) = zfp.';
        [Lr, info] = strict_ladder_deck(path, [0 0], 'lambda', lam, ...
                                        'strip_ap', false);
        out.rungs(k,:) = Lr(1,:);
        out.strehl(k,:) = info.strehl(1,:);
        if isempty(opts.save)
            delete(path);
        else
            out.decks{k} = path;
        end
        if ~opts.quiet
            fprintf('   [%+7.3f %+7.3f]'' : %10.4g nm (rung 4)  Strehl %6.4f\n', ...
                Frel(k,1)*180/pi*60, Frel(k,2)*180/pi*60, Lr(1,4)*1e9, info.strehl(1,4));
        end
    end
end

% =====================================================================
function r = exit_beam_(txt, tmp, apst, stand, bx, by, ie)
%EXIT_BEAM_  Radius of the exit beam about its chief, at the interface.
    ri = trace_(txt, tmp, apst, stand, bx, by, ie);
    ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
    d1 = ri.dir(:,1)/norm(ri.dir(:,1));
    dP = ri.pos(:,ok) - ri.pos(:,1);
    r = max(vecnorm(dP - d1*(d1.'*dP)));
end

function [p1, d1] = exit_chief_(txt, tmp, apst, stand, bx, by, ie)
    ri = trace_(txt, tmp, apst, stand, bx, by, ie);
    p1 = ri.pos(:,1);   d1 = ri.dir(:,1)/norm(ri.dir(:,1));
end

function ri = trace_(txt, tmp, apst, stand, bx, by, ie)
    fid = fopen(tmp,'w');  fprintf(fid,'%s',set_field_(txt, apst, stand, bx, by));  fclose(fid);
    macos.load_rx(tmp);
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
end

function s = set_field_(txt, apst, stand, bx, by)
    cdir = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
    cpos = apst - stand*cdir;
    s = regexprep(txt, '(ChfRayDir=\s*)[^\n]*', ['$1' v3_(cdir)]);
    s = regexprep(s,   '(ChfRayPos=\s*)[^\n]*', ['$1' v3_(cpos)]);
    s = regexprep(s,   '(yGrid=\s*)[^\n]*', ['$1' v3_([0; cos(by); -sin(by)])]);
end

function s = build_score_deck_(txt, L, nElt0, vf, d1, Dl, bx, by, apst, stand)
%BUILD_SCORE_DECK_  DECK + the two lens surfaces + a focal plane.
%   Inserted ahead of the nOutCord block, with nElt bumped by 3.  Nothing
%   upstream is touched, so the scored optics ARE the delivered ones.
    s = set_field_(txt, apst, stand, bx, by);
    s = regexprep(s, '(nElt=\s*)\d+', ['$1' num2str(nElt0+3)], 'once');
    lens = macos.design.ideal_lens_emit(L, nElt0+1, vf(:).', d1(:).');
    fp   = fp_block_(nElt0+3, vf + d1*(L.thickness + L.f), d1);
    blk  = [newline lens newline fp newline];
    i = regexp(s, '\n[^\n]*nOutCord=', 'once');
    if isempty(i)
        error('macos:design:afocal_score_psf:emit', ...
              'deck has no nOutCord block to insert ahead of');
    end
    s = [s(1:i-1) blk s(i:end)];
end

function b = fp_block_(ie, vpt, psi)
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    L = {};
    L{end+1} = sprintf('             iElt=  %d', ie);
    L{end+1} = '          EltName=  ScoreFP';
    L{end+1} = '          Element=  FocalPlane';
    L{end+1} = '          Surface=  Flat';
    L{end+1} = '            KrElt=-1.0E+22';
    L{end+1} = '            KcElt=0.0E+00';
    L{end+1} = ['           psiElt=  ' v3(psi(:).')];
    L{end+1} = ['           VptElt=  ' v3(vpt(:).')];
    L{end+1} = ['           RptElt=  ' v3(vpt(:).')];
    L{end+1} = '           IndRef=1.0E+00';
    L{end+1} = '           Extinc=0.0E+00';
    L{end+1} = '            nCoat=  0';
    L{end+1} = '             nObs=  0';
    L{end+1} = '           ApType=  None';
    L{end+1} = '         PropType=  Geometric';
    L{end+1} = '             zElt=1.0E+20';
    L{end+1} = '          nECoord=  -6';
    b = strjoin(L, newline);
end

function V = fit_focus_(path, seed, d1)
%FIT_FOCUS_  Least-squares closest point of the arriving ray bundle.
%   Same construction ALIGN_FOCAL_PLANE uses; a seeded focal plane is not
%   good enough for a metric whose reference sphere is centred on it.
    macos.load_rx(path);
    nE = macos.num_elt();
    tr = macos.trace(nE-1);   a = macos.get_ray_info(tr.nRays);
    tr = macos.trace(nE);     b = macos.get_ray_info(tr.nRays);
    ok = a.ok_trace(:) & a.ok_pass(:) & b.ok_trace(:) & b.ok_pass(:);
    if nnz(ok) < 10, V = seed(:); return; end
    P = b.pos(:,ok);
    D = b.pos(:,ok) - a.pos(:,ok);   D = D ./ vecnorm(D);
    A = nnz(ok)*eye(3) - D*D.';
    bb = sum(P,2) - D*sum(D.*P,1).';
    V = A \ bb;
    % a fitted focus a long way from the paraxial seed means the solve went
    % wrong, not that the lens did; keep the seed rather than move the
    % detector somewhere the metric cannot mean anything.
    if ~all(isfinite(V)) || abs(dot(V - seed(:), d1(:))) > 0.05*norm(seed(:))
        V = seed(:);
    end
    V = V(:);
end

function s = set_last_vpt_(txt, V, psi)
%SET_LAST_VPT_  Move the LAST element's vertex (and its Rpt) to V.
    v3 = @(a) sprintf('%.16E  %.16E  %.16E', a(1), a(2), a(3));
    i = regexp(txt, 'iElt=\s*\d+', 'start');
    j = i(end);
    head = txt(1:j-1);   tail = txt(j:end);
    tail = regexprep(tail, '(VptElt=\s*)[^\n]*', ['$1' v3(V(:).')], 'once');
    tail = regexprep(tail, '(RptElt=\s*)[^\n]*', ['$1' v3(V(:).')], 'once');
    tail = regexprep(tail, '(psiElt=\s*)[^\n]*', ['$1' v3(psi(:).')], 'once');
    s = [head tail];
end

function s = v3_(v),  s = sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));  end

function [cdir, cpos, apst, lam] = deck_src_(txt)
    cdir = grab3_(txt,'ChfRayDir');   cpos = grab3_(txt,'ChfRayPos');
    apst = grab3_(txt,'ApStop');
    t = regexp(txt,'Wavelen=\s*([-\d.EeD+]+)','tokens','once');
    lam = str2double(strrep(t{1},'D','E'));
end

function v = grab3_(txt, key)
    t = regexp(txt,[key '=\s*([^\n]*)'],'tokens','once');
    v = sscanf(strrep(t{1},'D','E'),'%f',3);
end

function del_(p),  if exist(p,'file'), delete(p); end,  end
