function OUT = r0_context_bisect()
%R0_CONTEXT_BISECT  e2e6m R0.1b: which context flips the Seg19 Ry poke?
%
%   r0_frame_probe measured: the SAME macos.perturb(19,Ry,'local') call
%   gives a pure segment PISTON of theta x 2.07 m (= the rpt lever arm;
%   rotation effectively about the shared parent VERTEX) in the s5 check
%   context, and a clean TILT about the segment center (the S4 Jacobian
%   column) in the dw_dx_multi harvest context.  This script bisects the
%   context: opd_ref('chief'), FEX before the poke, forward-vs-central
%   difference, channel-object-vs-raw call.
%
%   For each context: poke Ry, poke Tz (the piston basis measured in the
%   SAME context), and report |dW|, corr with the piston shape, and corr
%   with the STORED S4 tilt column.  Plus a linearity sweep in the s5
%   context (1e-9 / 1e-8 / 1e-7 rad forward).

    here = fileparts(mfilename('fullpath'));
    r1   = fullfile(here, '..', 'e2e6m');
    run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
    addpath(r1);
    P  = e2e6m_params(struct());
    rx = fullfile(r1, P.sn.rx);
    S4 = load(fullfile(r1, 's4_sens.mat'), 'ox');
    ox = S4.ox;
    IE = 19;

    ic   = find(strcmp(ox.field_names, 'C'), 1);
    Wn   = ox.per_field_w_nom_2d{ic};
    mnom = finite_(Wn);
    idx  = find(mnom);
    qRy  = find(ox.iElt == IE & ox.dof_idx == 1 & ...
                strcmp(ox.kind(:), 'RigidBody'), 1);
    A0   = ox.per_field_dwdx{ic};

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R0.1b -- context bisect, elt %d Ry', IE);

    C = {
      'a: s5-exact (chief ref, fwd 1e-9)',      struct('chief',1,'fex',0,'meth','fwd','d',1e-9)
      'b: no chief ref (fwd 1e-9)',             struct('chief',0,'fex',0,'meth','fwd','d',1e-9)
      'c: fex first (fwd 1e-9)',                struct('chief',0,'fex',1,'meth','fwd','d',1e-9)
      'd: fex + central 1e-8 (harvest-like)',   struct('chief',0,'fex',1,'meth','ctr','d',1e-8)
      'e: central 1e-8, no fex',                struct('chief',0,'fex',0,'meth','ctr','d',1e-8)
      'f: chief ref + central 1e-8',            struct('chief',1,'fex',0,'meth','ctr','d',1e-8)
    };
    n_c = size(C,1);
    res = struct('rms',nan(1,n_c), 'c_pis',nan(1,n_c), 'c_col',nan(1,n_c));
    for k = 1:n_c
        cfg = C{k,2};
        [dRy, dTz, both] = ctx_(rx, P, IE, cfg);
        res.rms(k)  = rms_(dRy);
        res.c_pis(k) = corr_(dRy, dTz);
        rows = find(ismember(idx, find(both)));
        col  = A0(rows, qRy);  col = col - mean(col);
        res.c_col(k) = corr_(dRy, col);
        L = say_(L, '  %-38s |dW| %.4g /nrad   corr(piston) %+.4f   corr(S4 col) %+.4f', ...
                 C{k,1}, res.rms(k), res.c_pis(k), res.c_col(k));
    end

    % channel-object call in context b
    m = macos.Session(P.sn.model);
    n = m.load_rx(rx);
    m.trace(n);  W0 = m.opd();
    ch = macos.channels.RigidBodyChannel(m, IE, 1);
    ch.apply(1e-9);   m.trace(n);  W1 = m.opd();
    ch.restore();
    b  = finite_(W0) & finite_(W1) & mnom;
    v1 = W1(b) - mean(W1(b));  v0 = W0(b) - mean(W0(b));
    dW = v1 - v0;
    L = say_(L, '  %-38s |dW| %.4g /nrad', ...
             'h: channel object, fwd 1e-9 (ctx b)', rms_(dW));

    % linearity sweep in the s5 context
    L = say_(L, '\n  linearity of the s5-context Ry response (forward pokes):');
    for d = [1e-9 1e-8 1e-7]
        cfg = struct('chief',1,'fex',0,'meth','fwd','d',d);
        [dRy, ~, ~] = ctx_(rx, P, IE, cfg);
        L = say_(L, '    theta %.0e rad : |dW| %.4g   (/theta: %.4g m/rad)', ...
                 d, rms_(dRy)*d/1e-9, rms_(dRy)*d/1e-9/d);
    end

    L = say_(L, '\nR0.1b DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(here,'r0_bisect_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('res',res, 'text',txt);
    save(fullfile(here,'r0_bisect.mat'), 'OUT');
end

function [dRy, dTz, both] = ctx_(rx, P, IE, cfg)
%CTX_  One context: fresh load, optional chief ref / FEX, then a Ry and
%   a Tz poke.  Returns dW vectors NORMALISED to a 1e-9 poke, on the
%   common-finite mask (intersected with the caller's use).
    macos.init(P.sn.model);
    n = macos.load_rx(rx);
    if cfg.chief, macos.opd_ref('chief'); end
    if cfg.fex,   macos.fex(1);           end
    macos.trace(n);
    W0 = macos.opd();
    [dRy, bR] = poke_(n, IE, 1, cfg, W0);
    [dTz, bT] = poke_(n, IE, 5, cfg, W0);
    both = bR & bT;
    % re-cut both vectors onto the common mask
    dRy = recut_(dRy, bR, both);
    dTz = recut_(dTz, bT, both);
end

function [dW, b] = poke_(n, IE, dof, cfg, W0)
    d = zeros(6,1);  d(dof+1) = cfg.d;
    if strcmp(cfg.meth, 'fwd')
        pk_(IE, d);   macos.trace(n);  Wp = macos.opd();
        pk_(IE, -d);
        b  = finite_(W0) & finite_(Wp);
        vp = Wp(b) - mean(Wp(b));  v0 = W0(b) - mean(W0(b));
        dW = (vp - v0) * (1e-9/cfg.d);
    else
        pk_(IE, d);    macos.trace(n);  Wp = macos.opd();
        pk_(IE, -2*d); macos.trace(n);  Wm = macos.opd();
        pk_(IE, d);
        b  = finite_(Wp) & finite_(Wm);
        vp = Wp(b) - mean(Wp(b));  vm = Wm(b) - mean(Wm(b));
        dW = (vp - vm) / 2 * (1e-9/cfg.d);
    end
end

function pk_(IE, d)
    macos.perturb(IE, 'rotation', d(1:3), 'translation', d(4:6), ...
                  'frame','local');
    macos.modify();
end

function v = recut_(v, bown, both)
    f = find(bown);  keep = ismember(f, find(both));
    v = v(keep);  v = v - mean(v);
end

function c = corr_(a, b)
    nmin = min(numel(a), numel(b));
    a = a(1:nmin);  b = b(1:nmin);
    c = (a(:).'*b(:)) / max(norm(a)*norm(b), realmin);
end
function m = finite_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
