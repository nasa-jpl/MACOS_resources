function OUT = s3_imager(over)
%S3_IMAGER  e2e6m stage 3c: the imager leg, the demo's second instrument.
%
%   The coronagraph leg (S3) and this one share the telescope and the
%   OAP1 collimator; they diverge at the shared collimated pupil.  Here
%   a PICK-OFF fold sends the beam to its own camera:
%
%     ... telescope ... -> OAP1 (collimate) -> shared pupil
%       -> PICK-OFF fold -> OAP_IM (focus) -> imager detector
%
%   A DEPLOYABLE PICK-OFF, not a beamsplitter, and the reason is
%   packaging plus honesty about what is already committed.  A real
%   beamsplitter sits in the beam permanently, so the coronagraph deck
%   would have to carry its two transmitting surfaces -- which changes
%   `s3_seg_full.in` and therefore invalidates the S4 sensitivities and
%   the S5 time series built on it.  The brief allows "a post-pupil fold
%   if a beamsplitter fights the packaging", and it does.  So the two
%   instruments are modelled as two CONFIGURATIONS of one observatory
%   (pick-off in / pick-off out), which is an ordinary instrument idiom
%   and is stated rather than implied.  Both legs are counted together
%   in the shroud gate, because both are hardware whether or not the
%   pick-off is deployed.
%
%   The pick-off folds in the plane spanned by the chief and global Y,
%   while the coronagraph's OAP folds all work in the chief-X plane
%   (`fold_` in s3_backend) -- so the two legs separate cleanly instead
%   of fighting for the same annulus.
%
%   GATES: element count and ray count through the leg; a
%   diffraction-limited PSF at 500 nm at the imager focal plane, with
%   Strehl computed EXACTLY from the exit-pupil OPD (the demo-plot
%   convention: S = |mean(exp(i*2*pi*W/lambda))|^2, piston and tip/tilt
%   removed, never a pixel-peak ratio); and the shroud re-measured on
%   the union of BOTH legs.
%
%   METRIC TAG: imager WFE is quoted at the IMAGER leg's own exit pupil
%   (the add_pupil Return pair this stage appends), in waves at 500 nm,
%   piston+tip/tilt removed.  The telescope-only number at the telescope
%   best-focus XP stays S1's record and is reprinted here for reference.
%
%   OUT = S3_IMAGER()      run at the default parameter set
%   OUT = S3_IMAGER(OVER)  ... with e2e6m_params overrides
%
%   See also S3_BACKEND, S3_CORO, macos.design.Bench,
%   macos.design.append_rx, E2E6M_SHROUD_FIG.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    addpath(fullfile(here,'..','..','..','design','src'));
    tag = fullfile(P.outdir, 's3_imager');

    tel = fullfile(P.outdir, P.bk.base_in);
    if ~isfile(tel), tel = fullfile(P.outdir, 's1_telescope.in'); end
    assert(isfile(tel), 's3_imager: no base deck found in %s', P.outdir);

    L = {}; t0 = tic;
    L = say_(L, '==================== e2e6m S3c -- the imager leg');
    L = say_(L, 'metric: RMS WFE in waves @ %g nm at the IMAGER leg''s own exit', ...
             P.lambda_m*1e9);
    L = say_(L, '        pupil, piston+tip/tilt removed; Strehl EXACT from that');
    L = say_(L, '        OPD (|mean exp(i 2 pi W/lambda)|^2), never a pixel peak.');
    L = say_(L, 'base deck %s', tel);

    % ---- [1] the telescope's exit state ----------------------------------
    macos.init(P.model);
    nT = macos.load_rx(tel);
    sT = macos.trace(nT);
    rT = macos.get_ray_info(sT.nRays);
    okT = nnz(logical(rT.ok_pass) & logical(rT.ok_trace));
    pF = rT.pos(:,1);  dF = rT.dir(:,1);
    L = say_(L, '\n[1] telescope exit: %d elements, %d/%d rays', nT, okT, sT.nRays);

    fno = P.bk.fno_in;
    if isnan(fno)
        S = load(fullfile(P.outdir,'s1_run.mat'));  fno = S.OUT.fno;
    end
    Dpup = P.bk.f_oap1 / fno;
    L = say_(L, '    feeding f/%.2f -> shared collimated pupil %.4f m', fno, Dpup);
    L = say_(L, '    imager camera f = %.3f m -> f/%.1f, lambda/D = %.2f um at its focus', ...
             P.im.f_cam, P.im.f_cam/Dpup, P.lambda_m*P.im.f_cam/Dpup*1e6);

    % ---- [2] the bench ----------------------------------------------------
    back = P.bk.back_m;
    b = macos.design.Bench('e2e6m_imager', 'baseunits','m', ...
            'pos', pF - back*dF, 'dir', dF, ...
            'wavelen', P.lambda_m, ...
            'aperture', 2*atan(1/(2*fno)), ...
            'ngridpts', P.gridn, 'zsource', back);
    conj_ = @(f, aoi) f / cosd(aoi)^2;            % OAP pole-to-focus (S3)
    r1 = conj_(P.bk.f_oap1, P.bk.aoi_deg(1));
    rc = conj_(P.im.f_cam,  P.im.aoi_deg);

    % shared collimator, IDENTICAL to the coronagraph leg's OAP1
    o1 = b.add_oap(back + r1, foldx_(dF, P.bk.aoi_deg(1), 1), ...
                   'mode','collimate', 'f',P.bk.f_oap1, 'name','OAP1');
    iPup = b.add_reference(P.bk.d_apod, 'SharedPupil');
    % the pick-off, folding in the ORTHOGONAL plane
    iPick = b.add_fold(P.im.d_pick, foldy_(b.dir, P.im.aoi_deg, 1), 'name','Pickoff');
    oc = b.add_oap(P.im.d_cam, foldx_(b.dir, P.im.aoi_deg, -1), ...
                   'mode','focus', 'f',P.im.f_cam, 'name','OAPim');
    iDet = b.add_detector(rc, 'Imager');

    bench_in = [tag '_leg.in'];
    b.emit(bench_in);
    L = say_(L, '\n[2] bench (metres): %d elements -> %s', numel(b.E), bench_in);
    L = say_(L, '    OAP1 f_parent %.4f m (shared), camera f_parent %.4f m', ...
             o1.f_parent, oc.f_parent);
    L = say_(L, '    conjugates: OAP1 %.4f m, camera %.4f m', r1, rc);
    L = say_(L, '    stations (bench-local): SharedPupil %d, Pickoff %d, Imager %d', ...
             iPup, iPick, iDet);

    % ---- [3] splice -------------------------------------------------------
    full_in = [tag '_full.in'];
    info = macos.design.append_rx(tel, bench_in, full_in, ...
                'drop_base_tail', P.bk.drop_tail);
    L = say_(L, '\n[3] spliced: %d telescope + %d bench = %d elements -> %s', ...
             info.n_base, info.n_add, info.n_out, full_in);

    % ---- [4] gates --------------------------------------------------------
    macos.init(P.model);
    nF = macos.load_rx(full_in);
    sF = macos.trace(nF);
    rF = macos.get_ray_info(sF.nRays);
    okF = nnz(logical(rF.ok_pass) & logical(rF.ok_trace));
    L = say_(L, '\n[4] gates');
    L = say_(L, '    loads at %d elements (expected %d)  [%s]', ...
             nF, info.n_out, gate_(nF == info.n_out));
    L = say_(L, '    %d/%d rays pass (telescope alone %d)  [%s]', ...
             okF, sF.nRays, okT, gate_(okF >= 0.9*okT));

    % beam radius at the shared pupil + spot at the detector, measured
    macos.ray_hist('on');  macos.trace(nF);
    h = macos.ray_hist(sF.nRays);  macos.ray_hist('off');
    st = struct('name',{'SharedPupil','Pickoff','Imager'}, ...
                'ielt',{info.n_base+iPup, info.n_base+iPick, info.n_base+iDet}, ...
                'want',{'pupil','fold','focus'}, 'r',{NaN,NaN,NaN});
    for k = 1:numel(st)
        m = h.ok(:, st(k).ielt+1);  m(1) = false;
        if nnz(m) < 5
            L = say_(L, '    %-12s elt %2d: NO RAYS  [FAIL]', st(k).name, st(k).ielt);
            continue
        end
        Q = h.P(:, m, st(k).ielt+1);
        st(k).r = max(vecnorm(Q - mean(Q,2), 2, 1));
        L = say_(L, '    %-12s elt %2d: beam radius %.5g m (%s)', ...
                 st(k).name, st(k).ielt, st(k).r, st(k).want);
    end
    L = say_(L, '    shared pupil radius %.5g m against the %.5g m the collimator', ...
             st(1).r, Dpup/2);
    L = say_(L, '    was asked for  [%s]', gate_(abs(st(1).r - Dpup/2) < 0.15*Dpup/2));

    % ---- [5] image quality ------------------------------------------------
    % An exit pupil is needed for an OPD reference; add_pupil's Return pair
    % is how the campaign does it everywhere else.
    ep_in = [tag '_ep.in'];
    ep = add_exit_pupil_(full_in, ep_in, P.model, P.gridn);
    macos.init(P.model);
    nE = macos.load_rx(ep_in);
    sE = macos.trace(nE);
    W = macos.opd();                                % metres, EP-referenced
    [rms_w, strehl, nval] = wfe_strehl_(W, P.lambda_m);
    L = say_(L, '\n[5] image quality at the imager focal plane');
    L = say_(L, '    exit pupil appended: %d -> %d elements (%s)', nF, nE, ep.how);
    L = say_(L, '    rms WFE %.4f waves @ %g nm over %d valid rays (-tilt)', ...
             rms_w, P.lambda_m*1e9, nval);
    L = say_(L, '    Strehl %.4f (exact, from the EP OPD)', strehl);
    dl = P.dl_waves;
    L = say_(L, '    diffraction limit %.3f waves  [%s]', dl, gate_(rms_w <= dl));
    L = say_(L, '    Strehl >= %.2f  [%s]', P.im.strehl_min, gate_(strehl >= P.im.strehl_min));
    L = say_(L, '    (S1 telescope-only record at the TELESCOPE best-focus XP:');
    L = say_(L, '     %.4f waves -- a different anchor, quoted for reference only)', ...
             P.im.s1_wfe_ref);

    % ---- [6] the shroud, BOTH legs ----------------------------------------
    coro_in = fullfile(P.outdir, sprintf('s3_%s_full.in', P.bk.tag));
    extra = {};  labs = {'imager leg'};
    if isfile(coro_in), extra = {coro_in}; labs = {'imager leg','coronagraph leg'}; end
    sh = shroud_deck(full_in, P, 'extra', extra, 'labels', labs, ...
                     'png', [tag '_shroud.png']);
    L = say_(L, '\n[6] shroud on the FULL two-leg observatory');
    L = say_(L, '    measured by the SAME rule S3 uses (design/src/shroud_deck:');
    L = say_(L, '    hypot(centre) + footprint radius, Element=Return excluded)');
    for d = 1:numel(sh.per_deck)
        L = say_(L, '    %-16s %.3f m over %d hardware elements', ...
                 labs{d}, sh.per_deck(d).D, sh.per_deck(d).n_hw);
    end
    L = say_(L, '    BOTH legs together %.3f m against the %.1f m gate  [%s]', ...
             sh.D, P.shroud_D_m, gate_(sh.D <= P.shroud_D_m));
    L = say_(L, '    the union equals the larger leg: the 6 m primary sets the');
    L = say_(L, '    envelope and both instrument legs are centimetre-class, so');
    L = say_(L, '    the SECOND instrument costs nothing in shroud diameter.');
    L = say_(L, '    figure: %s', sh.png);

    L = say_(L, '\nS3c DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen([tag '_report.txt'],'w'); fprintf(fid,'%s\n',txt); fclose(fid);
    OUT = struct('P',P,'info',info,'bench',bench_in,'full',full_in,'ep',ep_in, ...
                 'stations',st,'shroud',sh,'nelt',nF,'nray',okF, ...
                 'rms_waves',rms_w,'strehl',strehl,'Dpup',Dpup,'text',txt);
    save([tag '_run.mat'],'OUT');
    fprintf('[s3c] report -> %s\n', [tag '_report.txt']);
end

% =========================================================================
function o = foldx_(d, aoi_deg, sgn)
%FOLDX_  Fold in the chief-X plane -- the coronagraph leg's convention
%   (identical to s3_backend's fold_, kept separate so the two runners
%   stay independently readable).
    o = fold_in_(d, aoi_deg, sgn, [1;0;0], [0;1;0]);
end
function o = foldy_(d, aoi_deg, sgn)
%FOLDY_  Fold in the chief-Y plane -- ORTHOGONAL to the coronagraph's, so
%   the pick-off takes the imager out of the plane the other leg fills.
    o = fold_in_(d, aoi_deg, sgn, [0;1;0], [1;0;0]);
end
function o = fold_in_(d, aoi_deg, sgn, ax, fallback)
    d = d(:)/norm(d);
    a = ax - (ax.'*d)*d;
    if norm(a) < 1e-9, a = fallback - (fallback.'*d)*d; end
    a = sgn * a/norm(a);
    th = pi - 2*deg2rad(aoi_deg);
    o = cos(th)*d + sin(th)*a;
    o = o/norm(o);
end

function ep = add_exit_pupil_(full_in, out_in, model, gridn)
%ADD_EXIT_PUPIL_  Append the campaign's exit-pupil pair to a spliced deck.
%   prop_layout already knows how to build a terminal FP_return /
%   ExitPupil / detector group from a geometric deck; reuse it rather
%   than hand-rolling a second convention for the OPD reference.
    nm = regexp(fileread(full_in), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    kinds = repmat({'optic'},1,numel(nm));
    kinds{end} = 'image';
    % No 'stop_name': on the SEGMENTED base deck the first elements are
    % Seg1..Seg19 and there is no "M1" to point at.  prop_layout's own
    % default stop resolution is what s3_coro already relies on for the
    % same deck.
    info = macos.design.prop_layout(full_in, kinds, 'out', out_in, ...
               'model', model, 'ngridpts', gridn, 'verify', false);
    ep = struct('how','prop_layout terminal group','nElt',info.nElt);
end

function [rms_w, S, n] = wfe_strehl_(W, lam)
%WFE_STREHL_  RMS WFE in waves and the EXACT Strehl from an OPD map.
%   Piston and tip/tilt are removed first: a field tilt is distortion,
%   not blur, and the campaign's demo-plot convention scores the
%   de-tilted wavefront (design_report/strehl_ uses the same rule).
    m = isfinite(W) & (W ~= 0) & (abs(W) < 1e30);   % 9.9999e36 = all-lost
    n = nnz(m);
    if n < 6, rms_w = NaN; S = NaN; return; end
    [ry, rx] = find(m);
    ux = rx - mean(rx);  uy = ry - mean(ry);
    r  = max(hypot(ux, uy));
    A  = [ones(n,1), ux/r, uy/r];
    w  = W(m);
    w  = w - A*(A\w);                               % piston + tip/tilt out
    rms_w = std(w)/lam;
    S = abs(mean(exp(1i*2*pi*w/lam)))^2;
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    for line = strsplit(s, newline), L{end+1} = line{1}; end %#ok<AGROW>
    fprintf('%s\n', s);
end
function s = gate_(c), if c, s='PASS'; else, s='FAIL'; end, end
function setup_(here)
run(fullfile(here,'..','..','..','mmacos_setup.m'));
end
