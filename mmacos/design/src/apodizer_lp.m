function [A, info] = apodizer_lp(Pamp, r_px, opts)
%APODIZER_LP  Globally optimal amplitude apodizer for an arbitrary pupil.
%   [A, INFO] = APODIZER_LP(PAMP, R_PX) returns an N-by-N real amplitude
%   transmission A (0..1) that MAXIMIZES throughput subject to a
%   dark-zone contrast bound, for the pupil amplitude PAMP with beam
%   radius R_PX pixels, through a FIXED hard occulter + Lyot stop.
%
%   THE METHOD (Carlotti, Vanderbei & Kasdin 2011, Opt. Express 19,
%   26796; arXiv:1108.4050).  The pupil-to-focal map is LINEAR, so
%   "maximize transmission subject to bounds on the focal field" is a
%   linear program and its solution is the GLOBAL optimum for that
%   aperture -- no local minima, no initial guess.  Their Eq. 6 is
%
%       maximize   E(0,0)
%       subject to  -10^(-c/2) <= E(u,v)/E(0,0) <= 10^(-c/2)   in the
%                   dark zone, and 0 <= A(x,y) <= 1 on the aperture.
%
%   E(0,0) is itself linear in A, so the ratio constraint stays linear.
%
%   WHAT IS DIFFERENT HERE, and why it is still an LP.  Carlotti bound
%   the DIRECT focal field (a shaped pupil).  We bound the field AFTER
%   the existing occulter and Lyot stop -- the APLC operator of N'Diaye,
%   Zimmerman & Soummer (2016, ApJ 818, 163; arXiv:1601.02614).  Every
%   stage of that chain is linear in A, so the LP structure survives:
%
%       Lyot-plane field   l = p - T^{-1}( M .* T p )        (Babinet)
%       dark-zone field    E = S( L .* l )
%
%   with p = PAMP.*A, T the pupil->occulter matrix Fourier transform, M
%   the occulter support, L the Lyot stop, S the Lyot->dark-zone MFT.
%   BABINET IS NOT A STYLE CHOICE: it puts the only fine focal sampling
%   inside the occulter (a few lambda/D across), which is what keeps the
%   operator small.  We do NOT co-optimize the occulter or the Lyot --
%   that is the deferred half of N'Diaye's problem.
%
%   COMPLEX FIELD.  Carlotti's pupils are two-axis symmetric, so their
%   focal field is REAL and one +/- bound per sample suffices.  A real
%   traced pupil generally is not, so the bound is applied to Re and Im
%   SEPARATELY at 10^(-c/2)/sqrt(2) each, whose sum of squares is
%   exactly the modulus bound.  This is a bound on the modulus, not an
%   approximation of it -- but it is a SQUARE inscribed in the feasible
%   disc, so it is conservative by up to sqrt(2) in the corner
%   directions.  Stated, not hidden; INFO.bound_form records it.
%
%   SIZE CONTROL, in the escalation order the caller can drive:
%     1. SYMMETRY FOLD.  If the pupil is even in y, an optimal A even in
%        y exists (the problem is symmetric and the feasible set convex,
%        so symmetrizing any solution keeps feasibility and throughput),
%        and the dark-zone field is even in v -- so the variables halve
%        AND the constraint rows halve.  'fold_y'.
%     2. COARSE VARIABLE BASIS.  A is held constant on BLOCK x BLOCK
%        tiles.  Note this coarsens only the VARIABLES; the operator
%        keeps the pupil at full resolution.  Coarsening the PUPIL
%        instead would be wrong here for a measurable reason: on the
%        e2e6m 6 m pupil the 25 mm segment gaps are 1.06 px wide at
%        model 1024 and vanish below it, so a coarse-pupil LP would
%        optimize against an aperture with no gaps -- exactly the
%        feature it is being asked to work around.
%     3. ONE-SIDED DARK ZONE (Por 2019, arXiv:1908.02585): a D-shaped
%        zone halves the rows again and buys throughput.  'onesided'.
%
%   Args:
%     Pamp   N-by-N pupil amplitude, peak-normalised (from the ENGINE).
%     r_px   pupil beam radius in pixels.
%   Name-value:
%     'iwa','owa'      dark zone, lambda/D (default 3, 15).
%     'contrast'       target contrast 10^-c (default 1e-9).
%     'r_occ_lamD'     occulter radius, lambda/D (default 2.8).
%     'r_lyot_frac'    Lyot stop radius / pupil radius (default 0.90).
%     'block'          variable tile size in px (default 0 = auto from
%                      'nvar_target').
%     'nvar_target'    variable-count aim for the auto block (2500).
%     'dz_per_lamD'    dark-zone samples per lambda/D (default 2).
%     'n_fpm'          occulter-grid samples across its diameter (48).
%     'fold_y'         exploit y-symmetry (default 'auto': measured).
%     'onesided'       D-shaped dark zone, u >= iwa (default false).
%     'origin'         [x0 y0] pupil centre in 1-based pixels (default
%                      [] = measured centroid; see ORIGIN below).
%     'lp_maxiter'     linprog iteration cap (default 2e5).
%     'verbose'        progress printing (default true).
%
%   ORIGIN.  The Fourier origin is MEASURED from the amplitude centroid,
%   not assumed.  On an even grid the two candidate conventions --
%   floor(N/2)+1 (the engine's FFT centre) and (N+1)/2 (the symmetry
%   centre of an even array) -- differ by half a pixel, and a half-pixel
%   error is a linear phase ramp across the pupil, i.e. a dark-zone
%   shift.  INFO.origin_offset reports the distance from both.
%
%   INFO fields: .contrast_pred (the LP's own dark-zone contrast,
%     normalised to the APODIZED on-axis peak), .contrast_pred_bare
%     (normalised to the UNAPODIZED peak -- the convention ctb_contrast
%     and the e2e6m record use), .throughput (Phi^2-weighted fill over
%     the geometric pupil, same definition as ctb_aplc), .thru_amp
%     (sum(A.*P)/sum(P), the LP's own objective), .nvar, .nrow, .block,
%     .fold_y, .t_build, .t_solve, .exitflag, .kernel_check,
%     .roundtrip_check, .origin, .origin_offset, .bound_form.
%
%   See also CTB_APOD_PROLATE, CTB_APLC, macos.apodize.

    arguments
        Pamp              (:,:)                       % pupil FIELD (complex
                                                     % is expected) or a real
                                                     % amplitude
        r_px              (1,1) double {mustBePositive}
        opts.iwa          (1,1) double = 3.0
        opts.owa          (1,1) double = 15.0
        opts.contrast     (1,1) double {mustBePositive} = 1e-9
        opts.r_occ_lamD   (1,1) double = 2.8
        opts.r_lyot_frac  (1,1) double = 0.90
        opts.block        (1,1) double = 0
        opts.nvar_target  (1,1) double = 2500
        opts.dz_per_lamD  (1,1) double = 2.0
        opts.n_fpm        (1,1) double = 48
        opts.fold_y       = 'auto'
        opts.onesided     (1,1) logical = false
        opts.origin       (1,:) double = []
        opts.lp_maxiter   (1,1) double = 2e5
        opts.babinet_scale (1,1) double = 1   % DIAGNOSTIC: scale the
                                              % Babinet (occulter) term.
                                              % 1 is the physics; anything
                                              % else is an experiment.
        opts.verify_dense (1,1) double = 2    % after solving, re-evaluate
                                              % the dark zone on a grid this
                                              % many times finer.  An LP
                                              % only bounds the field AT ITS
                                              % SAMPLES; between them it is
                                              % free to ring.  0 disables.
        opts.predict      (:,:) double = []   % skip the LP: just return
                                              % what the operator predicts
                                              % for THIS apodizer.  The
                                              % model-vs-engine gate needs
                                              % exactly this, and so does
                                              % any diagnosis of it.
        opts.verbose      (1,1) logical = true
    end
    N = size(Pamp,1);
    assert(size(Pamp,2)==N, 'apodizer_lp: pupil must be square');
    vb = opts.verbose;

    % ---- 0. origin, measured ------------------------------------------
    % PAMP IS THE COMPLEX FIELD, not just its amplitude.  Modelling the
    % pupil as real cost this slice a full cycle: the traced field at the
    % apodizer carries 0.108 rad rms of phase, the amplitude-only operator
    % tracked the engine only to ~4x on a smooth (prolate) apodizer, and
    % the LP then optimized against that 4x error -- producing a mask that
    % the model scored at 3e-10 and the engine at 2e-6.  An optimizer will
    % always find the modelling error; the model has to be right first.
    Pmag = abs(Pamp);
    [X1,Y1] = meshgrid(1:N, 1:N);
    wsum = sum(Pmag(:));
    if isempty(opts.origin)
        org = [sum(X1(:).*Pmag(:))/wsum, sum(Y1(:).*Pmag(:))/wsum];
    else
        org = opts.origin;
    end
    info.origin = org;
    info.origin_offset = struct('fft_centre', norm(org - (floor(N/2)+1)), ...
                                'array_centre', norm(org - (N+1)/2));
    if vb
        fprintf('[lp] origin (measured centroid) [%.2f %.2f] px\n', org);
        fprintf('[lp]   offset from floor(N/2)+1 = %.3f px | from (N+1)/2 = %.3f px\n', ...
                info.origin_offset.fft_centre, info.origin_offset.array_centre);
    end
    X = X1 - org(1);   Y = Y1 - org(2);          % pupil coords, pixels
    RR = hypot(X,Y);

    % ---- 1. supports ---------------------------------------------------
    lit  = Pmag > 0.02*max(Pmag(:));              % illuminated pupil
    Lyot = disc_(RR, opts.r_lyot_frac*r_px, 8, X, Y);   % soft-edged stop
    a    = 1/(2*r_px);                            % cycles/px per lambda/D

    % ---- 2. symmetry fold ---------------------------------------------
    if ischar(opts.fold_y) || isstring(opts.fold_y)
        % fold on the FIELD, not the amplitude: a pupil whose amplitude
        % is y-symmetric but whose phase is not does not admit the fold.
        Pf = interp_flip_(Pamp, org(2));
        d  = Pamp(lit) - Pf(lit);
        fold = sqrt(mean(abs(d).^2))/max(sqrt(mean(abs(Pamp(lit)).^2)),eps) < 1e-3;
    else
        fold = logical(opts.fold_y);
    end
    info.fold_y = fold;

    % ---- 3. variable basis: BLOCK-constant tiles -----------------------
    blk = opts.block;
    if blk <= 0
        nlit = nnz(lit);
        % nvar ~ nlit / (blk^2 * fold_factor), so invert that -- DIVIDE by
        % the fold factor, do not multiply (getting this backwards asks for
        % 2500 variables and quietly delivers 612, i.e. a fifth of the
        % design freedom, which shows up only as a worse contrast).
        blk = max(1, round(sqrt(nlit / (max(opts.nvar_target,1) * tern_(fold,2,1)))));
    end
    [vid, nvar] = block_index_(lit, X, Y, blk, fold);
    info.block = blk;  info.nvar = nvar;
    if vb
        fprintf('[lp] %d lit px | block %d px | %d variables%s\n', ...
                nnz(lit), blk, nvar, tern_(fold,' (y-folded)',''));
    end

    % ---- 4. dark-zone samples ------------------------------------------
    du = 1/opts.dz_per_lamD;
    ug = -opts.owa:du:opts.owa;
    vg = 0:du:opts.owa;                     % v>=0 always: E is even in v
    if ~fold                                 % ... unless the pupil is not
        vg = -opts.owa:du:opts.owa;          %     even, then take both
    end
    [UU,VV] = meshgrid(ug,vg);
    rho = hypot(UU,VV);
    keep = rho >= opts.iwa & rho <= opts.owa;
    if opts.onesided, keep = keep & UU >= opts.iwa; end
    u_dz = UU(keep);  v_dz = VV(keep);
    ndz  = numel(u_dz);
    if vb
        fprintf('[lp] dark zone %g-%g lambda/D%s at %g samples/(lambda/D): %d samples\n', ...
                opts.iwa, opts.owa, tern_(opts.onesided,' (one-sided)',''), ...
                opts.dz_per_lamD, ndz);
    end

    % ---- 5. occulter grid ----------------------------------------------
    nf = opts.n_fpm;
    fg = linspace(-opts.r_occ_lamD, opts.r_occ_lamD, nf);
    dfo = fg(2)-fg(1);
    [FU,FV] = meshgrid(fg,fg);
    inocc = hypot(FU,FV) <= opts.r_occ_lamD;
    u_f = FU(inocc);  v_f = FV(inocc);  nfpm = numel(u_f);
    if vb, fprintf('[lp] occulter grid %dx%d -> %d samples inside %.2f lambda/D\n', ...
                   nf, nf, nfpm, opts.r_occ_lamD); end

    % ---- 6. build the operator -----------------------------------------
    tb = tic;
    W = Pamp;  W(~lit) = 0;                       % complex pupil weight
    % ROTATE OUT THE GLOBAL PHASE.  A traced field carries whatever
    % absolute phase the propagation left it with, and the objective
    % below maximizes Re(E00).  If the on-axis field happens to sit near
    % the imaginary axis, Re(E00) ~ 0 for every tile, the contrast bound
    % b*Re(E00) collapses with it, and the LP correctly reports that the
    % only feasible apodizer is A = 0 -- a trivial answer that looks like
    % an infeasible design problem.  A global phase is unobservable, so
    % rotate it away and the degeneracy Por removes stays removed.
    ph0 = sum(W(:));
    if abs(ph0) > 0
        W = W * conj(ph0)/abs(ph0);
    end
    info.global_phase_removed = angle(ph0);
    % 6a. per-variable pupil->occulter block  (sparse blocks, cheap)
    Ufpm = block_mft_(W, X, Y, vid, nvar, u_f, v_f, a);          % nfpm x nvar
    % 6b. per-variable DIRECT Lyot->dark-zone term
    Gdir = block_mft_(W.*Lyot, X, Y, vid, nvar, u_dz, v_dz, a);  % ndz  x nvar
    % 6c. Lyot-plane kernel: occulter sample -> dark-zone sample through L.
    %     L is radially symmetric, so this depends only on |u_dz - u_f|
    %     and is a Hankel transform of the stop's radial profile -- one
    %     1-D quadrature, then interpolate.  (Doing it pixel-wise would
    %     need a 40000 x 2300 complex matrix, ~1.5 GB.)
    dd = hypot(u_dz - u_f.', v_dz - v_f.');       % ndz x nfpm
    Kfun = lyot_kernel_(Lyot, X, [], a, max(dd(:))*1.02);
    K  = Kfun(dd) * (dfo^2) * (a^2);              % quadrature weight
    G  = Gdir - opts.babinet_scale * (K * Ufpm);  % ndz x nvar
    % 6d. on-axis (non-coronagraphic) field: linear in the variables
    w = accum_(vid, W(:), nvar);                  % nvar x 1, sum of P per tile
    info.t_build = toc(tb);
    info.scale = struct('w_sum', sum(w), 'Gdir_rms', rms_(Gdir(:)), ...
                        'Kterm_rms', rms_(reshape(K*Ufpm,[],1)), ...
                        'G_rms', rms_(G(:)), ...
                        'G_over_E00_flat', rms_(G*ones(nvar,1))/max(sum(w),eps));
    if vb, fprintf('[lp] operator %dx%d built in %.1f s\n', ndz, nvar, info.t_build); end

    % ---- 6e. self-tests -------------------------------------------------
    info.kernel_check   = kernel_check_(Kfun, Lyot, X, Y, a, dfo);
    info.roundtrip_check = roundtrip_check_(W, X, Y, a, r_px);
    if vb
        fprintf('[lp] kernel vs direct sum: rel %.2e | MFT round-trip: rel %.2e\n', ...
                info.kernel_check, info.roundtrip_check);
    end

    % ---- 6f. predict-only exit ------------------------------------------
    if ~isempty(opts.predict)
        A = opts.predict;
        assert(isequal(size(A),[N N]), ...
            'apodizer_lp: predict apodizer is %dx%d, pupil is %dx%d', ...
            size(A,1), size(A,2), N, N);
        % tile-mean transmission: weight by |W| so a tile straddling a gap
        % reports the transmission the light actually sees.
        aw = accum_(vid, reshape(A.*abs(W),[],1), nvar);
        nw = accum_(vid, reshape(abs(W),[],1), nvar);
        t  = aw ./ max(nw, eps);
        opts.u_dz = u_dz; opts.v_dz = v_dz;
        info = finish_(info, A, t, w, G, W, RR, r_px, nvar, ndz, nfpm, opts, vb);
        return
    end

    % ---- 7. the linear program -----------------------------------------
    b = sqrt(opts.contrast)/sqrt(2);          % per-part bound (see header)
    info.bound_form = sprintf(['Re,Im each |.| <= %.3e * Re(E00) ' ...
        '(square inscribed in the |E| <= %.3e disc)'], b, sqrt(opts.contrast));
    % With a COMPLEX pupil the on-axis field is complex, and "maximize
    % E(0,0)" is ambiguous by a global phase.  Maximize its REAL part
    % (Por 2019 sec. 2.2): it is linear, it removes exactly that
    % degeneracy, and the choice of the real axis is arbitrary only in
    % the sense that any fixed linear functional would do.
    wr = real(w);
    Aub = [ real(G) - b*wr.';
           -real(G) - b*wr.';
            imag(G) - b*wr.';
           -imag(G) - b*wr.' ];
    bub = zeros(size(Aub,1),1);
    info.nrow = size(Aub,1);
    f = -wr;                                   % maximize Re(sum(A.*W))
    lb = zeros(nvar,1);  ub = ones(nvar,1);
    if vb
        fprintf('[lp] LP: %d variables, %d inequality rows (%.0f MB dense)\n', ...
                nvar, info.nrow, numel(Aub)*8/1e6);
    end
    o = optimoptions('linprog','Algorithm','dual-simplex-highs', ...
                     'Display', tern_(vb,'iter','none'), ...
                     'MaxIterations', opts.lp_maxiter);
    ts = tic;
    [t, fval, ef, lpout] = linprog(f, Aub, bub, [], [], lb, ub, o);
    info.t_solve = toc(ts);
    info.exitflag = ef;
    info.lp_message = lpout.message;
    if vb
        fprintf('[lp] solved in %.1f s, exitflag %d (%s)\n', ...
                info.t_solve, ef, strtrim(lpout.message));
    end
    if isempty(t)
        error('apodizer_lp:infeasible', ...
              'linprog returned no solution (exitflag %d): %s', ef, lpout.message);
    end

    % ---- 8. unpack + predict --------------------------------------------
    A = zeros(N);
    A(lit) = t(vid(lit));
    A = min(max(A,0),1);
    opts.u_dz = u_dz; opts.v_dz = v_dz;
    info = finish_(info, A, t, w, G, W, RR, r_px, nvar, ndz, nfpm, opts, vb);

    % ---- 9. dense re-evaluation ------------------------------------------
    if opts.verify_dense > 1
        f2 = opts.verify_dense;
        du2 = du/f2;
        ug2 = -opts.owa:du2:opts.owa;
        vg2 = tern_(fold, 0:du2:opts.owa, -opts.owa:du2:opts.owa);
        [U2,V2] = meshgrid(ug2,vg2);
        r2 = hypot(U2,V2);
        k2 = r2 >= opts.iwa & r2 <= opts.owa;
        if opts.onesided, k2 = k2 & U2 >= opts.iwa; end
        u2 = U2(k2);  v2 = V2(k2);
        Gd2 = block_mft_(W.*Lyot, X, Y, vid, nvar, u2, v2, a);
        dd2 = hypot(u2 - u_f.', v2 - v_f.');
        K2  = lyot_kernel_(Lyot, X, [], a, max(dd2(:))*1.02);
        G2  = Gd2 - K2(dd2)*(dfo^2)*(a^2) * Ufpm;
        E2  = G2*t;
        info.ndz_dense = numel(u2);
        info.contrast_dense_mean = mean(abs(E2).^2)/max(info.E00^2,eps);
        info.contrast_dense_worst = max(abs(E2).^2)/max(info.E00^2,eps);
        info.contrast_dense_bare = mean(abs(E2).^2)/max(info.E00_bare^2,eps);
        info.ring_factor = info.contrast_dense_mean / ...
                           max(info.contrast_pred_mean, eps);
        if vb
            fprintf(['[lp] dense check (%gx finer, %d samples): mean %.3e ' ...
                     'worst %.3e -> ringing %.1fx\n'], f2, info.ndz_dense, ...
                     info.contrast_dense_mean, info.contrast_dense_worst, ...
                     info.ring_factor);
        end
    end
end

% ======================================================================
function info = finish_(info, A, t, w, G, W, RR, r_px, nvar, ndz, nfpm, opts, vb)
%FINISH_  The reported quantities, shared by the LP and the predict path
%   so a gate can never be comparing two different definitions.
    E00  = abs(w.'*t);                         % apodized on-axis amplitude
    E00b = abs(sum(W(:)));                     % UNapodized on-axis amplitude
    Edz  = G*t;
    info.contrast_pred      = max(abs(Edz).^2) / max(E00^2, eps);
    info.contrast_pred_mean = mean(abs(Edz).^2) / max(E00^2, eps);
    info.contrast_pred_bare = mean(abs(Edz).^2) / max(E00b^2, eps);
    info.thru_amp   = E00 / max(E00b, eps);
    geo = RR <= r_px;
    info.throughput = sum(A(geo).^2) / max(nnz(geo),1);   % ctb_aplc's definition
    info.E00 = E00;  info.E00_bare = E00b;  info.Edz = Edz;
    info.u_dz = opts.u_dz;  info.v_dz = opts.v_dz;   % sample coords, so a
                                                     % caller can bin the
                                                     % predicted field
                                                     % radially and compare
                                                     % with a measured curve
    info.nvar = nvar;
    info.ndz = ndz;  info.nfpm = nfpm;  info.iwa = opts.iwa;  info.owa = opts.owa;
    info.contrast_target = opts.contrast;
    if vb
        fprintf(['[lp] predicted DZ contrast: worst %.3e, mean %.3e ' ...
                 '(apodized-peak norm)\n'], info.contrast_pred, info.contrast_pred_mean);
        fprintf('[lp]   mean %.3e in BARE-peak normalisation (the ctb convention)\n', ...
                info.contrast_pred_bare);
        fprintf('[lp] throughput: %.4f (Phi^2 fill) | %.4f (amplitude ratio)\n', ...
                info.throughput, info.thru_amp);
    end
end

% ======================================================================
function D = disc_(RR, r, K, X, Y)
%DISC_  Supersampled hard disc, K x K sub-samples per pixel.
    D = double(RR <= r);
    edge = abs(RR - r) < 1.5;                 % only re-do the rim
    [ii,jj] = find(edge);
    off = ((1:K)-0.5)/K - 0.5;
    [ox,oy] = meshgrid(off,off);
    for m = 1:numel(ii)
        p = ii(m); q = jj(m);
        D(p,q) = mean(hypot(X(p,q)+ox(:), Y(p,q)+oy(:)) <= r);
    end
end

function B = interp_flip_(Pamp, y0)
%INTERP_FLIP_  Pamp reflected about the row y0 (nearest-row; the caller
%   only needs this to DECIDE whether to fold, at 1e-3).
    N = size(Pamp,1);
    src = round(2*y0 - (1:N));
    ok = src >= 1 & src <= N;
    B = zeros(N);
    B(ok,:) = Pamp(src(ok),:);
end

function [vid, nvar] = block_index_(lit, X, Y, blk, fold)
%BLOCK_INDEX_  Map each lit pixel to a variable id.  Tiles are blk x blk;
%   when folding, a tile and its y-mirror share one id.
    N = size(lit,1);
    bx = floor(X/blk);
    by = floor(Y/blk);
    if fold
        by = floor(abs(Y)/blk) .* 1;            % |y| -> same tile as -y
    end
    key = (bx - min(bx(:))) * (max(by(:))-min(by(:)) + 2) + (by - min(by(:)));
    k = key(lit);
    [~,~,ic] = unique(k);
    vid = zeros(N);
    vid(lit) = ic;
    nvar = max(ic);
end

function M = block_mft_(Wt, X, Y, vid, nvar, u, v, a)
%BLOCK_MFT_  numel(u) x nvar matrix whose (k,j) entry is
%   sum_{pixels in tile j} Wt(px) * exp(-2i*pi*a*(x*u_k + y*v_k)).
%   CHUNKED over pixels: the full nu x npix phase array would be ~900 MB
%   on the e2e6m pupil (1400 samples x 40356 lit px), so it is built
%   2^13 pixels at a time and accumulated straight into the nu x nvar
%   result, which is three orders of magnitude smaller.
    idx = find(vid > 0);
    np  = numel(idx);
    u = u(:);  v = v(:);
    M = complex(zeros(numel(u), nvar));
    % chunk so the transient phase array stays ~130 MB regardless of how
    % many focal samples the caller asked for (the occulter grid has ~2x
    % the samples the dark zone does, and 1800 x 40356 complex is 1.2 GB).
    chunk = max(512, floor(8e6/max(numel(u),1)));
    for lo = 1:chunk:np
        hi = min(lo+chunk-1, np);
        s  = idx(lo:hi);
        ph = exp(-2i*pi*a*(u*X(s).' + v*Y(s).'));
        M  = M + (ph .* Wt(s).') * ...
             sparse(1:numel(s), vid(s), 1, numel(s), nvar);
    end
end

function s = accum_(vid, Wcol, nvar)
    idx = find(vid > 0);
    s = accumarray(vid(idx), Wcol(idx), [nvar 1]);
end

function f = lyot_kernel_(Lyot, X, ~, a, rho_max)
%LYOT_KERNEL_  Focal-plane kernel of the Lyot stop, by PROJECTION-SLICE.
%   The stop is radially symmetric, so the kernel that carries an
%   occulter-plane sample to a dark-zone sample depends only on their
%   separation rho, and equals the stop's Fourier transform along ANY
%   direction.  Take it along x and the 2-D sum collapses to a 1-D one:
%
%       H(rho) = sum_{x,y} L(x,y) e^{-2i pi a x rho}
%              = sum_x [ sum_y L(x,y) ] e^{-2i pi a x rho}
%
%   i.e. the transform of the stop's COLUMN SUMS.  This is exact for the
%   array as built -- supersampled rim included -- where a Hankel
%   quadrature over a binned radial profile is not: the earlier version
%   of this routine did that and KERNEL_CHECK_ caught it at 2.7e-2.
    proj = sum(Lyot, 1);                        % 1 x N column sums
    xs   = X(1,:);                              % x of each column
    rho  = linspace(0, rho_max, 4001).';
    H    = exp(-2i*pi*a*(rho*xs)) * proj.';     % 4001 x 1
    f = @(d) reshape(interp1(rho, H, min(abs(d(:)),rho(end)), 'linear', 0), size(d));
end

function rel = kernel_check_(Kfun, Lyot, X, Y, a, ~)
%KERNEL_CHECK_  The projection-slice kernel against a DIRECT 2-D sum, at
%   separations spanning the dark zone AND at an oblique one (so a
%   kernel that is only right along x is caught).  Non-vacuous: a wrong
%   scale, a flipped exponent sign or a dropped 2*pi lands here rather
%   than 200 lines later inside a contrast number.
    d  = [0 1 3 7 15 22].';
    th = [0 0 0 pi/4 pi/3 pi/2].';            % oblique separations too
    got = Kfun(d);
    ref = zeros(size(d));
    for k = 1:numel(d)
        ref(k) = sum(sum(Lyot .* ...
            exp(-2i*pi*a*(X*(d(k)*cos(th(k))) + Y*(d(k)*sin(th(k)))))));
    end
    rel = max(abs(got - ref)) / max(abs(ref(1)), eps);
end

function rel = roundtrip_check_(~, X, Y, a, r_px)
%ROUNDTRIP_CHECK_  MFT forward-then-back must return what it was given.
%   This is the QUADRATURE-NORMALISATION test: the inverse carries a
%   du*dv*a^2 weight, and if it is wrong Babinet subtracts the wrong
%   amount and every contrast the LP predicts is wrong by that factor.
%
%   Run on a GAUSSIAN, not on the pupil: a hard-edged disc's spectrum
%   decays as 1/rho^(3/2) and never fits in a finite focal window, so a
%   pupil round-trip measures window truncation (~1e-2) and would hide
%   a genuine normalisation error underneath it.
    sig = r_px/3;
    G   = exp(-(X.^2+Y.^2)/(2*sig^2));
    ss  = max(1, round(r_px/16));
    m   = false(size(G)); m(1:ss:end,1:ss:end) = true;
    idx = find(m & G > 1e-6);
    fmax = 1/(2*a*ss) * 0.98;
    nf = 128; g = linspace(-fmax, fmax, nf); df = g(2)-g(1);
    [U,V] = meshgrid(g,g);
    ph = exp(-2i*pi*a*(U(:)*X(idx).' + V(:)*Y(idx).'));
    F  = ph * G(idx) * ss^2;
    back = (ph' * F) * (df^2) * (a^2);
    rel = norm(back - G(idx)) / max(norm(G(idx)), eps);
end

function r = rms_(v), r = sqrt(mean(abs(v).^2)); end

function s = tern_(c,a,b), if c, s=a; else, s=b; end, end
