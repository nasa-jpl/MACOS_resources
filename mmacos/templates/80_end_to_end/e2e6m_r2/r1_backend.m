function OUT = r1_backend(over)
%R1_BACKEND  e2e6m round 2, R1: the DM-bearing coronagraph back end.
%
%   Round 1's back end carried apodizer / FPM / Lyot only; its close-out
%   named the missing capability and called the extension parameter
%   work.  This runner does that work: the CTB topology (8 OAPs + two
%   flat-fold DMs, DST2R-class near-normal folds) instanced at
%   observatory scale behind the 6 m telescope, built in METRES with
%   `macos.design.Bench` and spliced with `macos.design.append_rx` so
%   ONE train carries a telescope perturbation to a contrast number.
%
%   Stations (light order): OAP1 collimate -> DM1 pupil -> DM2 ->
%   OAP2 focus -> OAP3 collimate -> Apodizer -> OAP4 focus -> FPM ->
%   OAP5 collimate -> Lyot -> OAP6 focus -> FieldStop -> OAP7
%   collimate -> Backend pupil -> OAP8 focus -> Science.  All relays
%   1:1, so the 47 mm pupil is preserved at every pupil station.
%
%   Mask sites stay passive Reference markers (masks are applied in
%   MATLAB, never declared in the deck -- the ctb silent-failure rule);
%   the DMs are REAL flat mirrors, GridData-augmented later on the
%   diffraction deck (r1_dm) for actuator figures.
%
%   GATES (round 1's, plus the DM stations): spliced element count;
%   ray count vs the telescope alone; beam radius at every station in
%   its expected conjugate; the shroud re-measured on the full train.
%
%   OUT = R1_BACKEND()      seg base, default knobs
%   OUT = R1_BACKEND(OVER)  with e2e6m_r2_params overrides
%
%   See also E2E6M_R2_PARAMS, R1_CORO, ../e2e6m/s3_backend.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','..','design','src'));
    tag = fullfile(P.outdir, ['r1_' P.b2.tag]);

    tel = fullfile(P.r1dir, P.b2.base_in);
    assert(isfile(tel), 'r1_backend: base deck %s not found', tel);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R1 -- the DM-bearing back end');
    L = say_(L, 'base deck %s', tel);

    % ---- [1] the telescope's exit state ---------------------------------
    macos.init(P.model);
    nT = macos.load_rx(tel);
    sT = macos.trace(nT);
    rT = macos.get_ray_info(sT.nRays);
    okT = nnz(logical(rT.ok_pass) & logical(rT.ok_trace));
    pF = rT.pos(:,1);  dF = rT.dir(:,1);
    L = say_(L, '\n[1] telescope exit: %d elements, %d/%d rays', nT, okT, sT.nRays);
    L = say_(L, '    focus at [%.4f %.4f %.4f] m, chief along [%.4f %.4f %.4f]', pF, dF);

    fno = P.b2.fno_in;
    if isnan(fno)
        S = load(fullfile(P.r1dir,'s1_run.mat'));
        fno = S.OUT.fno;
    end
    Dpup = P.b2.f_oap1 / fno;
    L = say_(L, '    feeding f/%.2f -> collimated pupil %.4f m at OAP1 f = %.3f m', ...
             fno, Dpup, P.b2.f_oap1);

    % ---- [2] the bench, in METRES ---------------------------------------
    back = P.b2.back_m;
    b = macos.design.Bench('e2e6m_r2_back', 'baseunits','m', ...
            'pos', pF - back*dF, 'dir', dF, ...
            'wavelen', P.lambda_m, ...
            'aperture', 2*atan(1/(2*fno)), ...
            'ngridpts', P.gridn, 'zsource', back);

    % CONJUGATE, NOT FOCAL LENGTH (the round-1 lesson): pole-to-focus of
    % an OAP is r = f/cos^2(AOI).  Every focus-leg distance below is a
    % conjugate; OAP-to-OAP through an intermediate focus is 2r.
    aoi = P.b2.aoi_deg;
    r1c = P.b2.f_oap1  / cosd(aoi)^2;
    rrc = P.b2.f_relay / cosd(aoi)^2;

    % FOLD SIDES: the SAME side every fold.  A near-retro fold turns the
    % chief by 180-2*AOI; because the beam direction REVERSES at each
    % fold, keeping the same geometric side makes the turn sense
    % alternate by itself and the chain PING-PONGS between two fixed
    % directions -- a bench accordion in a leg-sized pocket.
    % ALTERNATING the side (round 1's rule, fine for its 5 folds) adds
    % -2*AOI of net rotation per fold: over this chain's 10 folds the
    % accordion fanned 120 deg and walked to 5.2 m radius (shroud
    % 10.33 m vs the 8 m gate; measured, see the LOG).
    o1 = b.add_oap(back + r1c, fold_(dF, aoi, +1), ...
                   'mode','collimate', 'f',P.b2.f_oap1, 'name','OAP1');
    iDM1 = b.add_mirror(P.b2.d_dm1, 'out', fold_(b.dir, aoi, +1), ...
                        'name','DM1', 'aprad',P.b2.aprad_dm);
    iDM2 = b.add_mirror(P.b2.d_dm2, 'out', fold_(b.dir, aoi, +1), ...
                        'name','DM2', 'aprad',P.b2.aprad_dm);
    o2 = b.add_oap(P.b2.d_oap2, fold_(b.dir, aoi, +1), ...
                   'mode','focus', 'f',P.b2.f_relay, 'name','OAP2');
    o3 = b.add_oap(2*rrc, fold_(b.dir, aoi, +1), ...
                   'mode','collimate', 'f',P.b2.f_relay, 'name','OAP3');
    iApod = b.add_reference(P.b2.d_mark, 'Apodizer');
    o4 = b.add_oap(P.b2.d_mark, fold_(b.dir, aoi, +1), ...
                   'mode','focus', 'f',P.b2.f_relay, 'name','OAP4');
    iFPM = b.add_reference(rrc, 'FPM');
    o5 = b.add_oap(rrc, fold_(b.dir, aoi, +1), ...
                   'mode','collimate', 'f',P.b2.f_relay, 'name','OAP5');
    iLyot = b.add_reference(P.b2.d_mark, 'Lyot');
    o6 = b.add_oap(P.b2.d_mark, fold_(b.dir, aoi, +1), ...
                   'mode','focus', 'f',P.b2.f_relay, 'name','OAP6');
    iFS = b.add_reference(rrc, 'FieldStop');
    o7 = b.add_oap(rrc, fold_(b.dir, aoi, +1), ...
                   'mode','collimate', 'f',P.b2.f_relay, 'name','OAP7');
    iBck = b.add_reference(P.b2.d_mark, 'Backend');
    o8 = b.add_oap(P.b2.d_mark, fold_(b.dir, aoi, +1), ...
                   'mode','focus', 'f',P.b2.f_relay, 'name','OAP8');
    iSci = b.add_detector(rrc, 'Science');

    bench_in = [tag '_back.in'];
    b.emit(bench_in);
    L = say_(L, '\n[2] bench (metres): %d elements -> %s', numel(b.E), bench_in);
    L = say_(L, '    OAP parents (m): %.4f + 7 x %.4f (1:1 relays)', ...
             o1.f_parent, o2.f_parent);
    L = say_(L, '    DM legs: OAP1 -%.2f-> DM1 -%.2f-> DM2 -%.2f-> OAP2 (pupil %.1f mm on %.0f mm DMs)', ...
             P.b2.d_dm1, P.b2.d_dm2, P.b2.d_oap2, Dpup*1e3, 2*P.b2.aprad_dm*1e3);

    % ---- [3] splice -----------------------------------------------------
    full_in = [tag '_full.in'];
    info = macos.design.append_rx(tel, bench_in, full_in, ...
                'drop_base_tail', P.b2.drop_tail);
    L = say_(L, '\n[3] spliced: %d telescope + %d bench = %d elements -> %s', ...
             info.n_base, info.n_add, info.n_out, full_in);

    % ---- [4] gates ------------------------------------------------------
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

    st = struct('name',{'DM1','DM2','Apodizer','FPM','Lyot','FieldStop','Backend','Science'}, ...
                'loc', {iDM1,iDM2,iApod,iFPM,iLyot,iFS,iBck,iSci}, ...
                'want',{'pupil','pupil','pupil','focus','pupil','focus','pupil','focus'});
    macos.ray_hist('on');  macos.trace(nF);
    h = macos.ray_hist(sF.nRays);  macos.ray_hist('off');
    for k = 1:numel(st)
        st(k).ielt = info.n_base + st(k).loc;
        m = h.ok(:, st(k).ielt+1);  m(1) = false;
        if nnz(m) < 5
            L = say_(L, '    %-9s elt %2d: NO RAYS  [FAIL]', st(k).name, st(k).ielt);
            st(k).r = NaN;  continue;
        end
        Q = h.P(:, m, st(k).ielt+1);
        st(k).r = max(vecnorm(Q - mean(Q,2), 2, 1));
        L = say_(L, '    %-9s elt %2d: beam radius %.5g m (%s)', ...
                 st(k).name, st(k).ielt, st(k).r, st(k).want);
    end

    % ---- [5] the shroud on the full train -------------------------------
    sh = shroud_deck(full_in, P, 'labels', {'coronagraph leg (DM-bearing)'}, ...
                     'png', [tag '_shroud.png']);
    L = say_(L, '\n[5] shroud on the full train: %.3f m against the %.1f m gate  [%s]', ...
             sh.D, P.shroud_D_m, gate_(sh.D <= P.shroud_D_m));
    L = say_(L, '    train length %.2f m (launch axis)', sh.len);

    L = say_(L, '\nR1 backend (%s) DONE in %.1f min', P.b2.tag, toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen([tag '_report.txt'],'w');  fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'info',info, 'bench',bench_in, 'full',full_in, ...
                 'stations',st, 'shroud',sh, 'nelt',nF, 'nray',okF, ...
                 'Dpup',Dpup, 'fno',fno, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save([tag '_run.mat'],'OUT');
end

% =========================================================================
function o = fold_(d, aoi_deg, sgn)
%FOLD_  Outgoing chief for a fold of angle-of-incidence AOI, in the plane
%   spanned by d and global x (folds ACROSS the telescope's y-z fold
%   plane, round 1's shroud-friendly accordion).  SGN picks the side;
%   the chief turn is 180 - 2*AOI.
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
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
