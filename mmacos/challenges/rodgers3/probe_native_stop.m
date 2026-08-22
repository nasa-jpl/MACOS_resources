function OUT = probe_native_stop()
%PROBE_NATIVE_STOP  A/B: native STOP-element aiming vs Stage-0 Newton aiming.
%
%   The Stage-0 runner aims every field's chief through the stop centre
%   by a hand-rolled Newton iteration (aim_chief_ in rodgers3.m) because
%   the CODE V stop is a bare dummy plane and no stop ELEMENT existed to
%   aim at.  Dave's 2026-08-19 correction: the engine's element-bound
%   stop path (`STOP <elt>` / `ApStop= dx dy`, veneer macos.stop) drives
%   ChiefRayAiming (tracesub.F) -- ITERATIVE REAL-RAY aiming at a
%   designated stop element.  Template rule: emit a Reference element at
%   the stop plane, bind the stop to it, use native aiming -- IF this A/B
%   shows (a) chief-through-stop-centre agreement and (b) an unchanged
%   gate table on the r2 deck.
%
%   Note: the veneer route REQUIRES the inserted Reference element here:
%   stop_info_set (macos_api_mod.F90) rejects iElt >= nElt-2, so on the
%   4-element rodgers3 deck the stop cannot be bound to m2 (iElt 2 ==
%   nElt-2) even though the stop plane coincides with it (.seq th5 = 0).
%
%   Per r2 field (his 9 points):
%     Newton leg  -- original deck, aim_chief_ Newton iteration, strict
%                    WFE exactly as rodgers3.m computes it.
%     native leg  -- variant deck with elt 2 = Reference AT the stop
%                    centre (VptElt = stopC), macos.stop(2,[0 0]), then
%                    the same strict WFE.
%   Reports: stop-plane miss (both legs), chief FP landing delta, and
%   the WFE delta.  Saves probe_native_stop.mat.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);

    B = load(fullfile(here,'r3_build.mat'));  G = B.D.r2;
    S = rodgers3_seq();  R = S.r2;

    macos.init(256);

    txt0 = fileread(G.deck);
    txtN = insert_stop_ref_(txt0, G);
    fid = fopen(fullfile(here,'r3_r2_refstop.in'),'w');
    fprintf(fid,'%s',txtN);  fclose(fid);

    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));

    Nd = G.psi_si/norm(G.psi_si);  Vd = G.V_si;
    nF = numel(R.XAN);
    T = struct('miss_newton',nan(nF,1),'miss_native',nan(nF,1), ...
               'wfe_newton',nan(nF,1),'wfe_native',nan(nF,1), ...
               'dFP_mm',nan(nF,1),'dpos_mm',nan(nF,1));

    fprintf('\n  A/B native STOP-element aiming vs Newton aiming -- r2 deck\n');
    fprintf('  field (XAN,YAN)  |  stop miss N (m)  stop miss V (m) | chief dFP (mm) | WFE N (nm)  WFE V (nm)   dWFE (nm)\n');
    fprintf('  -----------------+-------------------------------------+----------------+------------------------------------\n');

    for q = 1:nF
        dq = tancomp_(R.XAN(q), R.YAN(q));

        % ---------------- Newton leg (original deck) --------------------
        [pq, aimq] = aim_chief_(txt0, tmp, G, dq);
        sq = trace_full_(txt0, tmp, pq, dq, 4);
        ok = sq.ok;  ok(1) = false;
        bx = asin(dq(1));  by = asin(dq(2));
        dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
        pp = aim_chief_(txt0, tmp, G, dp, pq);
        sp = trace_full_(txt0, tmp, pp, dp, 4);
        X  = fex_cross_(sq.pos(:,1), sq.dir(:,1), sp.pos(:,1), sp.dir(:,1));
        rf = strict_refs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                         sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
        wN = rf.wfe_centroid*1e9;
        cN = fp_land_(sq, Vd, Nd);

        % ---------------- native leg (Reference-stop deck) --------------
        [sv, missV] = native_trace_(txtN, tmp, G, dq, 5);
        okv = sv.ok;  okv(1) = false;
        [spv, ~]    = native_trace_(txtN, tmp, G, dp, 5);
        Xv = fex_cross_(sv.pos(:,1), sv.dir(:,1), spv.pos(:,1), spv.dir(:,1));
        rv = strict_refs(sv.pos(:,okv), sv.dir(:,okv), sv.opl(okv), ...
                         sv.pos(:,1), sv.dir(:,1), Vd, Nd, Xv);
        wV = rv.wfe_centroid*1e9;
        cV = fp_land_(sv, Vd, Nd);

        T.miss_newton(q) = aimq.miss;   T.miss_native(q) = missV;
        T.wfe_newton(q)  = wN;          T.wfe_native(q)  = wV;
        T.dFP_mm(q)      = norm(cV-cN)*1e3;

        fprintf('  [%+5.1f %+5.1f]     |  %13.3e    %13.3e   |  %12.6f  | %10.3f %10.3f  %+10.4f\n', ...
                R.XAN(q), R.YAN(q), aimq.miss, missV, T.dFP_mm(q), wN, wV, wV-wN);
    end

    fprintf('\n  max |dWFE| = %.4f nm   max chief dFP = %.6f mm   max native stop miss = %.3e m\n', ...
            max(abs(T.wfe_native-T.wfe_newton)), max(T.dFP_mm), max(T.miss_native));
    OUT = T;
    save(fullfile(here,'probe_native_stop.mat'),'T','G');
end

% =====================================================================
function [st, miss] = native_trace_(txtN, tmp, G, cdir, nE)
%NATIVE_TRACE_  Load the Reference-stop deck with a crude seed, bind the
%   stop to elt 2 (native ChiefRayAiming fires inside the STOP command),
%   then trace.  Returns full-train ray states + the stop-plane miss of
%   the traced chief.
    % crude geometric seed (same constructor as aim_chief_'s cold start)
    cdR  = [cdir(1); cdir(2); -cdir(3)];
    tq   = (G.z_m1 - G.stopC(3))/cdir(3);
    q    = G.stopC - tq*cdR;
    seed = q - (0.75/cdir(3))*cdir;

    emit_src_(txtN, tmp, seed, cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('probe_native_stop:load','deck failed to load'); end
    macos.stop(2, [0 0]);           % native aiming: chief through VptElt(2)

    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));

    % stop-plane miss: chief state at the Reference element
    tr2 = macos.trace(2);
    r2  = macos.get_ray_info(tr2.nRays);
    miss = norm(r2.pos(1:2,1) - G.stopC(1:2));
end

function c = fp_land_(st, Vd, Nd)
    p1 = st.pos(:,1);  d1 = st.dir(:,1);
    c  = p1 + d1*(dot(Nd, Vd - p1)/dot(Nd, d1));
end

function txt = insert_stop_ref_(txt0, G)
%INSERT_STOP_REF_  nElt 4 -> 5; renumber elts 2..4 -> 3..5; insert a flat
%   Reference element AT the stop centre between m1 and m2.
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    txt = txt0;
    txt = regexprep(txt, '(nElt=\s*)4', '$15');
    % renumber from the end so we never collide
    txt = regexprep(txt, 'iElt=  4', 'iElt=  5');
    txt = regexprep(txt, 'iElt=  3', 'iElt=  4');
    txt = regexprep(txt, 'iElt=  2', 'iElt=  3');
    blk = sprintf([ ...
        '             iElt=  2\n' ...
        '          EltName=  Stop\n' ...
        '          Element=  Reference\n' ...
        '          Surface=  Flat\n' ...
        '            KrElt=-1.0000000000000000E+22\n' ...
        '            KcElt=0.0000000000000000E+00\n' ...
        '           psiElt=  %s\n' ...
        '           VptElt=  %s\n' ...
        '           RptElt=  %s\n' ...
        '           IndRef=1.0E+00\n' ...
        '           Extinc=0.0E+00\n' ...
        '             nObs=  0\n' ...
        '           ApType=  None\n' ...
        '         PropType=  Geometric\n' ...
        '             zElt=0.0000000000000000E+00\n'], ...
        v3([0;0;-1]), v3(G.stopC), v3(G.stopC));
    % insert before the (renumbered) m2 block
    txt = regexprep(txt, '             iElt=  3\n', [blk '             iElt=  3\n'], 'once');
end

% ---- helpers copied verbatim from rodgers3.m (local functions there) --
function d = tancomp_(xan_deg, yan_deg)
    d = [tand(xan_deg); tand(yan_deg); 1];
    d = d/norm(d);
end

function [p0, aim] = aim_chief_(txt0, tmp, G, cdir, seed)
    if nargin < 5
        cdR  = [cdir(1); cdir(2); -cdir(3)];
        tq   = (G.z_m1 - G.stopC(3))/cdir(3);
        q    = G.stopC - tq*cdR;
        seed = q - (0.75/cdir(3))*cdir;
    end
    p0 = seed;
    h = 1e-4;  tol = 1e-9;  aim = struct('niter',0,'miss',inf);
    r0 = stop_miss_(txt0, tmp, G, p0, cdir);
    if norm(r0) >= tol
        rx = stop_miss_(txt0, tmp, G, p0+[h;0;0], cdir);
        ry = stop_miss_(txt0, tmp, G, p0+[0;h;0], cdir);
        J  = [(rx-r0)/h, (ry-r0)/h];
        for it = 1:8
            dp = -J\r0;
            p0 = p0 + [dp(1); dp(2); 0];
            r0 = stop_miss_(txt0, tmp, G, p0, cdir);
            aim.niter = it;
            if norm(r0) < tol, break; end
        end
    end
    aim.miss = norm(r0);
end

function r = stop_miss_(txt0, tmp, G, p0, cdir)
    st = trace_elt_(txt0, tmp, p0, cdir, 1);
    p = st.pos(:,1);  d = st.dir(:,1);
    t = (G.stopC(3) - p(3))/d(3);
    q = p + d*t;
    r = q(1:2) - G.stopC(1:2);
end

function st = trace_elt_(txt0, tmp, p0, cdir, ie)
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    if ~macos.has_rx(), error('probe_native_stop:load','deck failed to load: %s', tmp); end
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));
end

function st = trace_full_(txt0, tmp, p0, cdir, nE)
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    tr = macos.trace(nE);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));
end

function emit_src_(txt0, tmp, p0, cdir)
    v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
    s = regexprep(txt0, '(ChfRayDir=\s*)[^\n]*', ['$1' v3(cdir)]);
    s = regexprep(s,    '(ChfRayPos=\s*)[^\n]*', ['$1' v3(p0)]);
    fid = fopen(tmp,'w');  fprintf(fid,'%s',s);  fclose(fid);
end

function X = fex_cross_(p1,d1,p2,d2)
    d1 = d1/norm(d1);  d2 = d2/norm(d2);
    w0 = p1 - p2;  b = dot(d1,d2);  den = 1 - b^2;
    if abs(den) < 1e-14, X = p1; return; end
    s1 = ( b*dot(d2,w0) - dot(d1,w0)) / den;
    s2 = ( dot(d2,w0) - b*dot(d1,w0)) / den;
    X  = 0.5*((p1 + d1*s1) + (p2 + d2*s2));
end

function delete_if_(p), if exist(p,'file'), delete(p); end, end
