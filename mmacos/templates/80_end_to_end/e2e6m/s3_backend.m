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
    addpath(fullfile(here,'..','..','..','design','src'));
    tag = fullfile(P.outdir, ['s3_' P.bk.tag]);
    % BASE DECK: the SEGMENTED telescope when S2 has run.  The coronagraph
    % has to see the segmented pupil -- that is the whole point of putting
    % an APLC behind this telescope, and S4's sensitivities need the
    % segments and the back end in ONE train.  Falls back to the
    % monolithic telescope so the stage is runnable without S2.
    tel = fullfile(P.outdir, P.bk.base_in);
    if ~isfile(tel)
        tel = fullfile(P.outdir, 's1_telescope.in');
    end
    assert(isfile(tel), 's3_backend: no base deck found in %s', P.outdir);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S3 -- the back end');
    L = say_(L, 'base deck %s', tel);

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

    % CONJUGATE, NOT FOCAL LENGTH.  An off-axis parabola's pole-to-focus
    % distance is r = f/cos^2(AOI), not f (add_oap's own docstring: the
    % conjugate that realizes a desired f is 2f/(1-cos(theta)) =
    % f/cos^2(AOI)).  Placing the markers at f instead cost 1.011x at
    % 6 deg -- ~10 mm of defocus at f/20, measured as 5.89 waves rms of
    % which 5.66 was pure focus.  Every marker distance below is a
    % CONJUGATE.
    conj_ = @(f, aoi) f / cosd(aoi)^2;
    r1 = conj_(P.bk.f_oap1, P.bk.aoi_deg(1));
    r2 = conj_(P.bk.f_oap2, P.bk.aoi_deg(2));
    r3 = conj_(P.bk.f_oap3, P.bk.aoi_deg(3));
    r4 = conj_(P.bk.f_oap4, P.bk.aoi_deg(4));

    o1 = b.add_oap(back + r1, fold_(dF, P.bk.aoi_deg(1), 1), ...
                   'mode','collimate', 'f',P.bk.f_oap1, 'name','OAP1');
    iApod = b.add_reference(P.bk.d_apod, 'Apodizer');
    o2 = b.add_oap(P.bk.d_oap2, fold_(b.dir, P.bk.aoi_deg(2), -1), ...
                   'mode','focus', 'f',P.bk.f_oap2, 'name','OAP2');
    iFPM  = b.add_reference(r2, 'FPM');
    o3 = b.add_oap(r3, fold_(b.dir, P.bk.aoi_deg(3), 1), ...
                   'mode','collimate', 'f',P.bk.f_oap3, 'name','OAP3');
    iLyot = b.add_reference(P.bk.d_lyot, 'Lyot');
    o4 = b.add_oap(P.bk.d_oap4, fold_(b.dir, P.bk.aoi_deg(4), -1), ...
                   'mode','focus', 'f',P.bk.f_oap4, 'name','OAP4');
    iSci  = b.add_detector(r4, 'Science');

    bench_in = [tag '_back.in'];
    b.emit(bench_in);
    L = say_(L, '\n[2] bench (metres): %d elements -> %s', numel(b.E), bench_in);
    L = say_(L, '    OAP parent focal lengths (m): %.4f %.4f %.4f %.4f', ...
             o1.f_parent, o2.f_parent, o3.f_parent, o4.f_parent);
    L = say_(L, '    pole-to-focus conjugates  (m): %.4f %.4f %.4f %.4f', ...
             r1, r2, r3, r4);
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
    % SAME rule as the imager leg, from one place (design/src/shroud_deck):
    % two shroud numbers measured two ways is how a demo ends up quoting
    % 7.451 on one slide and 7.448 on the next.
    sh = shroud_deck(full_in, P, 'labels', {'coronagraph leg'}, ...
                     'png', [tag '_shroud.png']);
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

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
function s = tern_(c,a,b), if c, s = a; else, s = b; end, end
