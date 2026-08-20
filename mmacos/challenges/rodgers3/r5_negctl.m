function OUT = r5_negctl()
%R5_NEGCTL  Negative control: the r5 gate must have teeth.
%
%   The r5 rung is the ZRN-convention gate: BornWolf modes = SCO C-index
%   MINUS ONE (C1 = the NRADIUS slot).  This probe rebuilds the r5 deck
%   with that offset DELIBERATELY DROPPED (modes = C-index verbatim) and
%   scores both decks on Mike's 9 optimization fields.  A correct
%   convention lands near his 53 nm; the dropped offset must MISS by a
%   large factor -- that miss is what proves the tRodgers3 gate is
%   testing the convention rather than passing vacuously.
%
%   OUT: .max_ok_nm, .max_bad_nm, .factor (bad/ok).  Saves nothing.
%   Requires macos.init to have been called by the caller (tRodgers3's
%   class setup does; standalone: macos.init(256) first).
%
%   See also RODGERS3, BUILD_R3, tests/tRodgers3.m.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','mmacos_setup.m'));
    addpath(here);

    B = load(fullfile(here,'r3_build.mat'));  G = B.D.r5;
    S = rodgers3_seq();  R = S.r5;

    txt_ok  = fileread(G.deck);
    txt_bad = bump_modes_(txt_ok);

    w_ok  = score9_(txt_ok,  G, R);
    w_bad = score9_(txt_bad, G, R);

    OUT = struct('max_ok_nm', max(w_ok), 'max_bad_nm', max(w_bad), ...
                 'factor', max(w_bad)/max(w_ok), ...
                 'w_ok_nm', w_ok, 'w_bad_nm', w_bad);
    fprintf(['  r5 negative control: 9pt max %.2f nm (correct C-offset) vs ' ...
             '%.2f nm (offset dropped) -- factor %.1fx\n'], ...
            OUT.max_ok_nm, OUT.max_bad_nm, OUT.factor);
end

% =========================================================================
function txt = bump_modes_(txt)
%BUMP_MODES_  ZernModes k -> k+1 on every mirror (drops the C1 offset).
    lines = strsplit(txt, '\n', 'CollapseDelimiters', false);
    for i = 1:numel(lines)
        t = lines{i};
        j = strfind(t, 'ZernModes=');
        if isempty(j), continue; end
        v = sscanf(t(j+10:end), '%d');
        lines{i} = [t(1:j+9) sprintf(' %d', v + 1)];
    end
    txt = strjoin(lines, newline);
end

function w = score9_(txt0, G, R)
%SCORE9_  Strict WFE (centroid ref, nm) on the 9 .seq fields -- the
%   Stage-0 metric, Newton aiming (these decks carry no stop element).
    tmp = [tempname '.in'];
    cu  = onCleanup(@() delete_if_(tmp));
    Nd = G.psi_si/norm(G.psi_si);  Vd = G.V_si;
    nF = numel(R.XAN);
    w = nan(nF,1);
    for q = 1:nF
        dq = tancomp_(R.XAN(q), R.YAN(q));
        pq = aim_chief_(txt0, tmp, G, dq);
        sq = trace_full_(txt0, tmp, pq, dq);
        ok = sq.ok;  ok(1) = false;
        if nnz(ok) < 10, continue; end
        bx = asin(dq(1));  by = asin(dq(2));
        dp = [sin(bx+1e-5); sin(by); sqrt(1-sin(bx+1e-5)^2-sin(by)^2)];
        pp = aim_chief_(txt0, tmp, G, dp, pq);
        sp = trace_full_(txt0, tmp, pp, dp);
        X  = fex_cross_(sq.pos(:,1), sq.dir(:,1), sp.pos(:,1), sp.dir(:,1));
        rf = strict_refs(sq.pos(:,ok), sq.dir(:,ok), sq.opl(ok), ...
                         sq.pos(:,1), sq.dir(:,1), Vd, Nd, X);
        w(q) = rf.wfe_centroid*1e9;
    end
end

% ---- helpers copied verbatim from rodgers3.m ----------------------------
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
    if ~macos.has_rx(), error('r5_negctl:load','deck failed to load: %s', tmp); end
    tr = macos.trace(ie);
    ri = macos.get_ray_info(tr.nRays);
    st = struct('pos',ri.pos,'dir',ri.dir,'opl',ri.opl, ...
                'ok', ri.ok_trace(:) & ri.ok_pass(:));
end

function st = trace_full_(txt0, tmp, p0, cdir)
    emit_src_(txt0, tmp, p0, cdir);
    macos.load_rx(tmp);
    nE = macos.num_elt();
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
