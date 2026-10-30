function [Ls, info] = iodc2010_solve(L, p, opts)
%IODC2010_SOLVE  The inner solve: CALIB on the spacings, in rounds, with a thickness floor.
%   [Ls, info] = iodc2010_solve(L, p, 'fields_deg', [0 2.5 5], ...) takes a
%   green lens L (its radius, sign pattern and stop semi-diameter are kept)
%   and returns Ls, the same lens with its axial spacings (every surface,
%   the stop when opts.var_stop, the image plane when opts.var_fp) solved
%   by MACOS CALIB for the smallest RMS wavefront error at the design
%   fields, then SCALED to |EFL| = p.efl_mm (a scale keeps the common radius
%   and the merit's D*theta form).
%
%   Why rounds.  The deck CALIB solves carries three things that depend on
%   the lens it started from: the beam diameter (the paraxial entrance-pupil
%   diameter D), the pupil position the field chief rays are aimed at, and
%   the source plane.  All three move as the surfaces move, so the solve is
%   re-emitted from its own result (the solved lens's paraxials) and run
%   again until D settles to opts.D_tol (relative) or opts.rounds is reached.
%   The first ladder's open item (R = 130: CALIB's own off-axis evaluation
%   lost every ray) was the source plane being overtaken by a surface that
%   moved upstream; iodc2010_emit now backs the CALIB source off by the
%   lens's length and this solve re-emits per round.
%
%   The thickness floor.  CALIB knows no bounds and the problem's loophole
%   invites zero thickness.  A round whose solved lens respects every floor
%   is ACCEPTED as solved; a round that drives a glass thickness below
%   opts.floor_glass_R x R or an air space below opts.floor_air_R x R is not:
%   the gap is clamped to its floor (the downstream surfaces shift together),
%   BOTH surfaces of that gap leave the variable set (the gap is now rigid at
%   its floor), and the next round re-solves the rest from there (an active
%   set).  If the last round still clamps, one more round runs with the
%   frozen set so the lens returned is a solved one.  The floors are in units of the radius
%   because the solve runs at the seed's own scale (a 3 mm lens at EFL 100
%   is 1.4 mm long at the solve's EFL 45; an absolute 2 mm floor wrecked
%   every round of the first smoke runs).  A clamped gap is reported in
%   info.clamped.
%
%   A round whose after-solve WFE at any field reads the engine's
%   lost-rays value (>= 1e30) is not accepted: the lens goes back to the
%   last accepted round and the solve stops with info.stop = 'lost-field'.
%
%   Holding D (opts.D_target).  CALIB moves the surfaces, so the focal
%   length at the solve's scale drifts and the final scale to |EFL| = 100
%   would carry D with it.  With opts.D_target given (mm, at EFL 100) the
%   stop is re-sized every round so the f-number stays 100 / D_target at
%   the solve's own scale, and the scaled lens comes out at D_target: the
%   (D, theta) point the caller asked for is the point it gets.  NaN keeps
%   the seed's stop.
%
%   Filling the stop (opts.fit_stop).  The beam is the PARAXIAL entrance
%   pupil; where the pupil's own aberration carries its edge rays outside
%   the stop, the metric counts them lost ("vignetting") and the lens fails
%   on axis.  After the solve the stop semi-diameter is shrunk (D with it)
%   until no on-axis ray is lost, in at most five steps; info.stop_fit
%   records the factor.
%
%   info: .rounds (table: round, n_iter, stop_reason, wfe_before, wfe_after,
%   D, efl), .stop, .clamped (1xN-1 logical), .n_rounds, .deck, .scale,
%   .efl_solved (before scaling), .ok.
arguments
    L struct
    p struct
    opts.fields_deg (1,:) double = [0 2.5 5]
    opts.rounds (1,1) double = 4
    opts.iters_per_round (1,1) double = 12
    opts.D_tol (1,1) double = 3e-3
    opts.floor_glass_R (1,1) double = 0.03
    opts.floor_air_R (1,1) double = 0.01
    opts.var_fp (1,1) logical = true
    opts.var_stop (1,1) logical = false
    opts.ray_grid (1,1) double = 31
    opts.deck (1,:) char = ''
    opts.fit_stop (1,1) logical = true
    opts.D_target (1,1) double = NaN
    opts.quiet (1,1) logical = true
end
say = @(varargin) iif_(~opts.quiet, varargin{:});
if isempty(opts.deck), opts.deck = fullfile(p.outdir, [L.name '_calib.in']); end
N = L.N;
isglass = ~strcmpi(L.glass(1:N-1), 'air');
floor_t = opts.floor_air_R * L.R * ones(1, N - 1);  floor_t(isglass) = opts.floor_glass_R * L.R;

Lc = L;  Lc.semidia = inf(1, N);           % no clear apertures during the solve
Lc.t(1:N-1) = max(Lc.t(1:N-1), floor_t);  Lc.z = [0, cumsum(Lc.t(1:N-1))];
Lc = hold_D_(Lc, p, opts.D_target);
frozen = false(1, N);                      % surfaces removed from the variable set
clamped = false(1, N - 1);
rows = {};  info.stop = 'rounds';  D_prev = NaN;
accepted = Lc;  clean = false;  n_accepted = 0;
r = 0;
while r < opts.rounds || (~clean && r < opts.rounds + 1)
    r = r + 1;
    P = iodc2010_paraxial(Lc, p.lambda_nm);
    if ~isfinite(P.efl) || ~isfinite(P.D) || P.D <= 0
        info.stop = 'paraxial';  break
    end
    var_surf = find(~frozen);
    opt = struct('fields_deg', opts.fields_deg, 'max_iters', opts.iters_per_round, ...
        'var_surf', var_surf, 'var_fp', opts.var_fp, 'var_stop', opts.var_stop, 'ray_grid', opts.ray_grid);
    E = iodc2010_emit(Lc, P, p, opts.deck, opt);
    macos.load_rx(E.path);
    try
        res = macos.calib();
    catch err
        rows{end+1} = struct('round', r, 'n_iter', -1, 'stop_reason', ['calib error: ' err.message], ...
            'wfe_before', NaN, 'wfe_after', NaN, 'D', P.D, 'efl', P.efl); %#ok<AGROW>
        info.stop = 'calib-failed';  macos.modify();  break
    end
    row = struct('round', r, 'n_iter', res.n_iter, 'stop_reason', res.stop_reason, ...
        'wfe_before', max(res.old_wfe(:)), 'wfe_after', max(res.new_wfe(:)), 'D', P.D, 'efl', P.efl);
    rows{end+1} = row; %#ok<AGROW>
    if any(res.new_wfe(:) >= 1e30)
        info.stop = 'lost-field';  macos.modify();  break
    end
    % read the solved positions back
    z = zeros(1, N);
    for k = 1:N, v = macos.get_elt_vpt(E.iSurf(k));  z(k) = v(3); end
    vs = macos.get_elt_vpt(E.iStop);  vf = macos.get_elt_vpt(E.iFP);
    macos.modify();
    t = [diff(z), vf(3) - z(N)];
    zs = vs(3) - z(1);
    js = find(z - z(1) <= zs, 1, 'last');               % the gap the stop sits in (empty: ahead of surface 1)
    if isempty(js), off_s = zs; else, off_s = zs - (z(js) - z(1)); end
    % the floor: clamp, shift what follows, freeze both surfaces of the gap
    clean = true;
    for k = 1:N-1
        if t(k) < floor_t(k) * (1 - 1e-9)       % a gap sitting AT its floor re-reads a round-off below it
            t(k) = floor_t(k);  clamped(k) = true;  frozen(k) = true;  frozen(k+1) = true;  clean = false;
        end
    end
    if t(N) <= 0, info.stop = 'image-inside';  break; end
    Ln = Lc;  Ln.t = t;  Ln.z = [0, cumsum(t(1:N-1))];
    if isempty(js), Ln.stop(1) = off_s;                % keeps its place in its own gap when gaps shift
    elseif js < N, Ln.stop(1) = Ln.z(js) + min(off_s, t(js));
    else, Ln.stop(1) = Ln.z(N) + off_s; end
    Pn = iodc2010_paraxial(Ln, p.lambda_nm);
    if ~isfinite(Pn.efl) || Pn.z_img <= Ln.z(N) || ~isfinite(Pn.D)
        info.stop = 'paraxial';  break
    end
    Ln = hold_D_(Ln, p, opts.D_target);  Pn = iodc2010_paraxial(Ln, p.lambda_nm);
    Lc = Ln;
    if clean, accepted = Ln;  n_accepted = n_accepted + 1; end
    say('    round %d: %2d it (%s) wfe %.3g -> %.3g mm, D %.3f -> %.3f, EFL %.2f%s\n', r, res.n_iter, ...
        res.stop_reason, row.wfe_before, row.wfe_after, P.D, Pn.D, Pn.efl, ...
        tern_(clean, ' (accepted)', sprintf(' -> clamped gap(s) %s, frozen %s', mat2str(find(clamped)), mat2str(find(frozen)))));
    if clean && ~isnan(D_prev) && abs(Pn.D - P.D) <= opts.D_tol * P.D && ...
            any(strcmp(res.stop_reason, {'converged', 'floor', 'stalled'}))
        info.stop = 'settled';  break
    end
    if clean && all(frozen(1:N)), info.stop = 'all-frozen';  break; end
    D_prev = Pn.D;
end
if ~clean && ~strcmp(info.stop, 'settled'), info.stop = [info.stop ' (last round clamped; the previous clean round is returned)']; end
% scale the accepted lens to |EFL| = p.efl_mm
Ls = accepted;
Pa = iodc2010_paraxial(Ls, p.lambda_nm);
info.efl_solved = Pa.efl;
sc = p.efl_mm / abs(Pa.efl);
Ls.R = Ls.R * sc;  Ls.t = Ls.t * sc;  Ls.z = Ls.z * sc;  Ls.stop = Ls.stop * sc;
Ls.semidia = [];
info.scale = sc;
info.stop_fit = 1;
if opts.fit_stop && isfinite(Pa.efl)
    for k = 1:5
        Pf = iodc2010_paraxial(Ls, p.lambda_nm);
        Lw = Ls;  Lw.semidia = inf(1, N);
        Ef = iodc2010_emit(Lw, Pf, p, regexprep(opts.deck, '\.in$', '_fit.in'));   % keep the CALIB deck on disk
        macos.load_rx(Ef.path);
        F0 = iodc2010_eval(Ef, Pf, p, 0);
        macos.modify();
        if F0.nLost == 0, break; end
        fac = 0.995 * sqrt(1 - F0.nLost / F0.nLaunched);   % the lost fraction is an area
        Ls.stop(2) = Ls.stop(2) * fac;  info.stop_fit = info.stop_fit * fac;
        say('    stop fit: %d/%d on-axis rays outside the stop -> a x %.4f\n', F0.nLost, F0.nLaunched, fac);
    end
end
info.clamped = clamped;
info.n_rounds = numel(rows);
info.deck = opts.deck;
if isempty(rows)
    info.rounds = table();
else
    info.rounds = struct2table([rows{:}], 'AsArray', true);
end
info.n_accepted = n_accepted;
% solved = at least one accepted (clean) round; a later failed round returns the last accepted lens
info.ok = n_accepted >= 1 && all(Ls.t(1:N-1) > 0) && Ls.t(N) > 0;
end

function L = hold_D_(L, p, D_target)
% the stop re-sized so that f/# = p.efl_mm / D_target at the lens's own scale
if ~isfinite(D_target), return; end
P = iodc2010_paraxial(L, p.lambda_nm);
if ~isfinite(P.efl) || ~isfinite(P.D) || P.D <= 0, return; end
D_here = D_target * abs(P.efl) / p.efl_mm;
L.stop(2) = L.stop(2) * D_here / P.D;
end

function out = tern_(c, a, b)
if c, out = a; else, out = b; end
end

function iif_(c, varargin)
if c, fprintf(varargin{:}); end
end
