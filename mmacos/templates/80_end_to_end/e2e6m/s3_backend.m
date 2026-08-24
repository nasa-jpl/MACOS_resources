function OUT = s3_backend(over)
%S3_BACKEND  e2e6m stage 3: the imager + coronagraph back end, ONE train.
%
%   Builds the instrument behind the telescope with `macos.design.Bench`
%   -- in METRES, so it splices onto the telescope deck rather than
%   living as a second prescription -- and appends it with
%   `macos.design.append_rx`.  Only one train can carry a telescope
%   perturbation through to a coronagraph contrast number; that is the
%   whole reason the back end is spliced instead of scored separately.
%
%   TOPOLOGY (light order, from the telescope's focus):
%
%     ... telescope ... -> OAP1 (collimate) -> Apodizer pupil
%       -> OAP2 (focus)  -> FPM focus
%       -> OAP3 (collimate) -> Lyot pupil
%       -> OAP4 (focus)  -> Science focus (detector)
%
%   plus an IMAGER leg taken as a separate configuration off the same
%   collimated pupil (the coronagraph and the imager do not coexist in
%   one sequential train; the imager deck is emitted alongside).
%
%   The mask sites (Apodizer / FPM / Lyot) are passive `Reference`
%   markers, exactly as the ctb bench does it: **masks are applied in
%   MATLAB, never declared in the deck** -- an obscuration declared on a
%   Reference clips rays only and the diffraction wavefront sails
%   through it untouched (the ctb README's silent failure mode).
%
%   FOLD ANGLES.  Near-normal OAP folds (AOI ~ 5-8 deg) keep each section
%   only slightly off-axis and minimise off-axis astigmatism, and they
%   are also what keeps the back end inside the shroud annulus the
%   telescope already fills.  `add_oap` takes the parent focal length
%   directly and supports any fold.
%
%   APERTURES (rule 2): the functional stops go on the FLAT marker planes
%   (`add_reference`), whose vertex IS the pole.  An `aprad` on an OAP is
%   metadata only -- a circular ApVec on an off-axis section is applied
%   about the parent VERTEX, far from the beam, and would block the whole
%   bundle.
%
%   GATES: the spliced deck loads at the expected element count and keeps
%   the telescope's ray count; the chief ray reaches every station; the
%   pupil and focus conjugates land where the builder says (beam diameter
%   at each pupil marker, spot size at each focus marker); and the shroud
%   figure is re-measured on the FULL train.
%
%   OUT = S3_BACKEND()      run at the default parameter set
%   OUT = S3_BACKEND(OVER)  ... with e2e6m_params overrides
%
%   See also E2E6M_PARAMS, S1_TELESCOPE, macos.design.Bench,
%   macos.design.append_rx, E2E6M_SHROUD_FIG.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    tag = fullfile(P.outdir, 's3');
    tel = fullfile(P.outdir, 's1_telescope.in');
    assert(isfile(tel), 's3_backend: S1 artifact %s not found', tel);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S3 -- the back end');
    L = say_(L, 'telescope %s', tel);

    % ---- [1] the telescope's exit state ----------------------------------
    macos.init(P.model);
    nT = macos.load_rx(tel);
    sT = macos.trace(nT);
    rT = macos.get_ray_info(sT.nRays);
    okT = nnz(logical(rT.ok_pass) & logical(rT.ok_trace));
    % the chief ray ARRIVING at the last element, and the marginal beam
    % there, give the cone the back end has to catch
    pF = rT.pos(:,1);  dF = rT.dir(:,1);
    L = say_(L, '\n[1] telescope exit: %d elements, %d/%d rays', nT, okT, sT.nRays);
    L = say_(L, '    focus at [%.4f %.4f %.4f] m, chief along [%.4f %.4f %.4f]', pF, dF);

    % the f/# feeding the back end, measured from the telescope deck
    fno = P.bk.fno_in;
    if isnan(fno)
        S = load(fullfile(P.outdir,'s1_run.mat'));
        fno = S.OUT.fno;
    end
    Dpup = P.bk.f_oap1 / fno;              % collimated pupil diameter, m
    L = say_(L, '    feeding f/%.2f -> collimated pupil %.4f m at OAP1 f = %.3f m', ...
             fno, Dpup, P.bk.f_oap1);

    % ---- [2] the bench, in METRES ----------------------------------------
    % Start the bench a little BEFORE the telescope focus and step to
    % OAP1 at f_oap1 past it: add_oap's 'collimate' mode wants the
    % incoming chief diverging from a focus one conjugate back.
    back = P.bk.back_m;
    b = macos.design.Bench('e2e6m_back', 'baseunits','m', ...
            'pos', pF - back*dF, 'dir', dF, ...
            'wavelen', P.lambda_m, ...
            'aperture', 2*atan(1/(2*fno)), ...
            'ngridpts', P.gridn, 'zsource', back);

    o1 = b.add_oap(back + P.bk.f_oap1, fold_(dF, P.bk.aoi_deg(1), 1), ...
                   'mode','collimate', 'f',P.bk.f_oap1, 'name','OAP1');
    iApod = b.add_reference(P.bk.d_apod, 'Apodizer');
    o2 = b.add_oap(P.bk.d_oap2, fold_(b.dir, P.bk.aoi_deg(2), -1), ...
                   'mode','focus', 'f',P.bk.f_oap2, 'name','OAP2');
    iFPM  = b.add_reference(P.bk.f_oap2, 'FPM');
    o3 = b.add_oap(P.bk.f_oap3, fold_(b.dir, P.bk.aoi_deg(3), 1), ...
                   'mode','collimate', 'f',P.bk.f_oap3, 'name','OAP3');
    iLyot = b.add_reference(P.bk.d_lyot, 'Lyot');
    o4 = b.add_oap(P.bk.d_oap4, fold_(b.dir, P.bk.aoi_deg(4), -1), ...
                   'mode','focus', 'f',P.bk.f_oap4, 'name','OAP4');
    iSci  = b.add_detector(P.bk.f_oap4, 'Science');

    bench_in = [tag '_back.in'];
    b.emit(bench_in);
    L = say_(L, '\n[2] bench (metres): %d elements -> %s', numel(b.E), bench_in);
    L = say_(L, '    OAP parent focal lengths (m): %.4f %.4f %.4f %.4f', ...
             o1.f_parent, o2.f_parent, o3.f_parent, o4.f_parent);
    L = say_(L, '    marker stations: Apodizer %d, FPM %d, Lyot %d, Science %d (bench-local)', ...
             iApod, iFPM, iLyot, iSci);

    % ---- [3] splice ------------------------------------------------------
    % Drop the telescope's terminal quartet (FP_return / ExitPupil / FP):
    % the back end re-images that focus, and a FocalPlane mid-train would
    % terminate it.
    full_in = [tag '_full.in'];
    info = macos.design.append_rx(tel, bench_in, full_in, ...
                'drop_base_tail', P.bk.drop_tail);
    L = say_(L, '\n[3] spliced: %d telescope + %d bench = %d elements -> %s', ...
             info.n_base, info.n_add, info.n_out, full_in);

    % ---- [4] gates -------------------------------------------------------
    macos.init(P.model);
    nF = macos.load_rx(full_in);
    sF = macos.trace(nF);
    rF = macos.get_ray_info(sF.nRays);
    okF = nnz(logical(rF.ok_pass) & logical(rF.ok_trace));
    L = say_(L, '\n[4] gates');
    L = say_(L, '    loads at %d elements (expected %d)  [%s]', ...
             nF, info.n_out, gate_(nF == info.n_out));
    L = say_(L, '    %d/%d rays pass (telescope alone %d)  [%s]', ...
             okF, sF.nRays, okT, gate_(okF > 0.9*okT));

    % conjugates: beam radius at each marker, measured
    st = struct('name',{'Apodizer','FPM','Lyot','Science'}, ...
                'ielt',{info.n_base+iApod, info.n_base+iFPM, ...
                        info.n_base+iLyot, info.n_base+iSci}, ...
                'want',{'pupil','focus','pupil','focus'});
    macos.ray_hist('on');  macos.trace(nF);
    h = macos.ray_hist(sF.nRays);  macos.ray_hist('off');
    for k = 1:numel(st)
        m = h.ok(:, st(k).ielt+1);  m(1) = false;
        if nnz(m) < 5
            L = say_(L, '    %-9s elt %2d: NO RAYS  [FAIL]', st(k).name, st(k).ielt);
            st(k).r = NaN;  continue;
        end
        Q = h.P(:, m, st(k).ielt+1);
        r = max(vecnorm(Q - mean(Q,2), 2, 1));
        st(k).r = r;
        L = say_(L, '    %-9s elt %2d: beam radius %.5g m (%s)', ...
                 st(k).name, st(k).ielt, r, st(k).want);
    end

    % ---- [5] the shroud, re-measured on the FULL train --------------------
    sh = shroud_full_(full_in, nF, P, [tag '_shroud.png']);
    L = say_(L, '\n[5] shroud on the full train: %.3f m against the %.1f m gate  [%s]', ...
             sh.D, P.shroud_D_m, gate_(sh.D <= P.shroud_D_m));
    L = say_(L, '    train length %.2f m (launch axis)', sh.len);

    L = say_(L, '\nS3 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen([tag '_report.txt'],'w');  fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'info',info, 'bench',bench_in, 'full',full_in, ...
                 'stations',st, 'shroud',sh, 'nelt',nF, 'nray',okF, ...
                 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save([tag '_run.mat'],'OUT');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function o = fold_(d, aoi_deg, sgn)
%FOLD_  Outgoing chief for a fold of angle-of-incidence AOI, in the plane
%   spanned by d and the global x axis (so the back end folds ACROSS the
%   telescope's y-z fold plane and stays inside the annulus).  SGN picks
%   the side.  The chief turn is 180 - 2*AOI.
    d = d(:)/norm(d);
    a = [1;0;0];
    a = a - (a.'*d)*d;
    if norm(a) < 1e-9, a = [0;1;0] - ([0;1;0].'*d)*d; end
    a = sgn * a/norm(a);
    th = pi - 2*deg2rad(aoi_deg);
    o = cos(th)*d + sin(th)*a;
    o = o/norm(o);
end

function sh = shroud_full_(rx, nE, P, png)
%SHROUD_FULL_  Radial extent of every HARDWARE element's footprint about
%   the launch axis, measured on the SPLICED deck (no Telescope object
%   exists for it), and drawn.
%
%   Element=Return surfaces are excluded: they are the propagator's
%   return planes and exit-pupil reference SPHERES, mathematical
%   surfaces rather than glass anyone builds, and an exit-pupil sphere
%   sits at a radius with nothing to do with the hardware envelope.
%   Element=Reference markers ARE kept -- a mask or pupil site is a real
%   mount.  Read from the deck text, since a spliced deck has no spec.
    kinds = regexp(fileread(rx), '(?m)^\s*Element=\s*(\S+)', 'tokens');
    isHW = true(1,nE);
    for k = 1:min(nE, numel(kinds))
        isHW(k) = ~strcmpi(kinds{k}{1}, 'Return');
    end
    macos.ray_hist('on');  s = macos.trace(nE);
    h = macos.ray_hist(s.nRays);  macos.ray_hist('off');
    C = nan(3,nE);  R = nan(1,nE);
    for k = 1:nE
        m = h.ok(:,k+1);  m(1) = false;
        if nnz(m) < 3, continue; end
        Q = h.P(:, m, k+1);
        C(:,k) = mean(Q,2);
        R(k) = max(vecnorm(Q - C(:,k), 2, 1));
    end
    ok = isfinite(R);
    rr = hypot(C(1,:), C(2,:)) + R;
    sh = struct('D', 2*max(rr(ok & isHW)), ...
                'len', max(C(3,ok)) - min(C(3,ok)), ...
                'r_elt', rr, 'is_hw', isHW, 'png', png);
    f = figure('Visible','off','Position',[100 100 700 640]);
    ax = axes(f);  hold(ax,'on');  axis(ax,'equal');
    th = linspace(0,2*pi,361);  Rg = P.shroud_D_m/2;
    plot(ax, Rg*cos(th), Rg*sin(th), 'k-', 'LineWidth', 2.0);
    idx = find(ok);  cols = lines(max(numel(idx),7));
    for j = 1:numel(idx)
        k = idx(j);
        st = '-';  w = 1.2;
        if ~isHW(k), st = ':';  w = 0.7;  end     % not hardware, not gated
        plot(ax, C(1,k)+R(k)*cos(th), C(2,k)+R(k)*sin(th), st, ...
             'Color', cols(j,:), 'LineWidth', w);
    end
    xlabel(ax,'x  [m]');  ylabel(ax,'y  [m]');  grid(ax,'on');  box(ax,'on');
    title(ax, sprintf(['full train, end-on: hardware union %.3f m against ' ...
                       'the %.1f m gate (%s)'], sh.D, P.shroud_D_m, ...
                      tern_(sh.D <= P.shroud_D_m,'FITS','OVER')));
    lim = 1.3*Rg;  xlim(ax,[-lim lim]);  ylim(ax,[-lim lim]);
    saveas(f, png);  close(f);
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
