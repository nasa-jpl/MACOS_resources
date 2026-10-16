function out = pol_contrast_floor(pupil, det, opts)
%MACOS.POL_CONTRAST_FLOOR  Polarization-limited contrast floor at a detector.
%   OUT = macos.pol_contrast_floor(PUPIL, DET) propagates the loaded
%   prescription with vector diffraction on and splits the detector field
%   into CO-polarized, CROSS-polarized and LONGITUDINAL channels.  The
%   cross-polarized channel is the part of the light no scalar DM control
%   can touch, so its peak-normalized level IS the polarization contrast
%   floor (PLAN_POLARIZATION.md §2c).
%
%   WHY THIS IS DONE AT THE DETECTOR, NOT AS A PUPIL MULTIPLIER.  The
%   chain is linear in the input Jones state (measured 4.2e-16 on
%   Rx_Coro), and Phase 3a Tranche 1 propagates all three component
%   planes with the identical scalar kernel.  A spatially UNIFORM
%   analyzer therefore commutes with propagation, so projecting at the
%   detector gives the same answer as projecting in the pupil -- without
%   ever building a pupil multiplier.  That matters: the Jones pupil
%   cannot be used as one (it is assembled from RayE and carries the
%   accumulated OPL phase, and the RayE<->WFElt phase relation is
%   train-dependent), which is what blocked the first two 2c designs.
%
%   THE ANALYZER IS DERIVED, NOT ASSUMED.  "Co-polarized" is referenced
%   to the MEAN OUTPUT state, never to the input state: a real train
%   rotates polarization geometrically with zero diattenuation and zero
%   retardance, and billing that uniform rotation as cross-polarized
%   light reports an aberration where there is none (you would simply
%   orient the analyzer to it).  The analyzer is the dominant eigenvector
%   of the 2x2 pupil COHERENCY matrix C_ij = sum_pupil E_i conj(E_j).
%   Coherency is phase-insensitive -- the common wavefront cancels in
%   E_i conj(E_j) -- so unlike a plain pupil mean it does not collapse on
%   an aberrated pupil, and by construction it is the analyzer that
%   MINIMIZES cross-polarized power.  Its degree of polarization is
%   reported (.per_state.dop); a value below 'dop_min' means the output
%   is not close to fully polarized and no analyzer is well defined, so
%   the run is flagged rather than silently reported.
%
%   INPUT STATES.  'input' selects what is launched:
%     'x'            single run, (Ex0,Ey0) = (1,0)          [default]
%     'y'            single run, (0,1)
%     'unpolarized'  TWO runs, x and y, summed in INTENSITY.  The second
%                    state is never synthesized from the first -- an
%                    unpolarized source is two independent traces.
%     [ex; ey]       single run at an arbitrary complex Jones state
%                    (normalized to unit power -- absolute channel powers
%                    scale with it, every reported ratio does not).
%   With 'unpolarized' each run gets its OWN analyzer (each input state
%   has its own mean output state); the channel maps are then summed.
%
%   Name-value pairs:
%     'input'      as above (default 'x').
%     'coatings'   cell array of coating sets to sweep for the coating
%                  sensitivity the §2c contract asks for.  Each set is a
%                  struct array with fields .elt .index .extinc
%                  .thickness (macos.coating arguments; thickness in
%                  element BaseUnits), optionally .label.  Entry k is
%                  applied on top of the as-loaded prescription, the
%                  whole floor is recomputed, and OUT.sweep(k) holds the
%                  scalar summary plus .d_cross_rel, the fractional
%                  change in cross-polarized power vs the as-loaded
%                  baseline.  Default {} = no sweep.  Every set must
%                  cover the SAME elements: a coating can be overwritten
%                  but not cleared (coat_set takes >= 1 layer), so sets
%                  over different elements would accumulate.  For the
%                  same reason the sweep LEAVES THE LAST SET APPLIED --
%                  macos.load_rx to get back to the as-loaded stack.
%     'dark_zone'  [r_in r_out] annulus radii in PIXELS at the detector.
%                  When given, OUT.floor.dark_zone reports mean/median/
%                  peak peak-normalized contrast per channel there.
%                  Default [] = skip (no lambda/D convention is assumed;
%                  use macos.lambda_over_D_pixels
%                  helpers to convert).
%     'pupil_tol'  pixels with transverse intensity below this fraction
%                  of the pupil peak are excluded from the coherency sum
%                  (default 1e-12).
%     'floor_tol'  detector pixels whose total intensity is below this
%                  fraction of the peak get NaN in the RATIO maps -- a
%                  small denominator is masked, never zero-filled
%                  (default 1e-12).
%     'dop_min'    minimum pupil degree of polarization for the analyzer
%                  to be considered well defined (default 0.99).
%     'scope_tol'  tolerance on the carried-fraction check below
%                  (default 0.05).
%
%   SCOPE -- READ THIS BEFORE QUOTING A NUMBER.  Phase 3a Tranche 1 seeds
%   the three component planes from RayE at the FIRST physical-optics leg
%   of a trace and thereafter applies only a common scalar phase.  Any
%   polarizing surface AFTER that first leg therefore transforms the RAYS
%   but not the diffraction GRID, so its contribution never reaches the
%   detector field.  This is exactly the Tranche-2 gap
%   (PLAN_POLARIZATION.md §3a.3), and on a chain with mirrors between
%   propagation legs it makes the floor a LOWER BOUND.
%
%   It is measured, not assumed: OUT.scope compares the cross-polarized
%   fraction of the pupil COHERENCY computed from the grid planes (what
%   diffraction carries) against the same quantity computed from RayE at
%   the same element (the full train).  OUT.scope.carried is their ratio
%   PER INPUT STATE (averaging the states hides the shortfall),
%   OUT.scope.worst is the entry furthest from 1, OUT.scope.full_chain is
%   true only when every state is 1 within 'scope_tol', and a warning
%   (macos:pol_contrast_floor:tranche1) fires when it is not.
%   Prescriptions whose polarizing elements all precede the first
%   propagation leg (Rx_Cass_FarField: two mirrors, then one far-field
%   hop) carry the full train; Rx_Coro does not.
%
%   Returns struct:
%     .I_co .I_cross .I_long   N x N detector intensity per channel,
%                              summed over the run set
%     .I_total                 their sum (== macos.intensity(DET))
%     .contrast_co/_cross/_long   each channel divided by peak(.I_co)
%     .frac_cross              .I_cross ./ .I_total, NaN where the
%                              denominator is below 'floor_tol'
%     .floor      .co .cross .long   total power per channel
%                 .cross_over_co     the headline ratio
%                 .contrast_cross_peak   max(.contrast_cross)
%                 .dark_zone             (when 'dark_zone' given)
%     .per_state(k)  .state .analyzer .complement .coherency .dop
%                    .power (.co .cross .long)
%     .scope      .grid_cross_frac .ray_cross_frac .carried .full_chain
%     .checks     .parseval  max |I_co+I_cross - (|E1|^2+|E2|^2)| / peak
%                 .closure   max |I_co+I_cross+I_long - intensity| / peak
%     .sweep(k)   (when 'coatings' given) .label .floor .scope
%                 .d_cross_rel and the channel maps .I_co .I_cross .I_long
%     .pupil .det .input .mask
%
%   The pre-call polarization state is restored on exit.
%
%   See also: macos.jones_pupil, macos.pol_maps, macos.complex_field,
%   macos.vector_diffraction, macos.coating.
arguments
    pupil          (1,1) double {mustBeInteger, mustBePositive}
    det            (1,1) double {mustBeInteger, mustBePositive}
    opts.input           = 'x'
    opts.coatings  (1,:) cell   = {}
    opts.dark_zone (1,:) double = []
    opts.pupil_tol (1,1) double {mustBePositive} = 1e-12
    opts.floor_tol (1,1) double {mustBePositive} = 1e-12
    opts.dop_min   (1,1) double = 0.99
    opts.scope_tol (1,1) double {mustBePositive} = 0.05
end

states = input_states_(opts.input);
if ~isempty(opts.dark_zone) && numel(opts.dark_zone) ~= 2
    error('macos:pol_contrast_floor:darkZone', ...
        '''dark_zone'' must be [r_in r_out] in pixels.');
end

s0 = macos.polarization();                 % restore on exit
cleanup = onCleanup(@() restore_pol_(s0)); %#ok<NASGU> lifetime only

% ---- baseline (as-loaded coatings) --------------------------------------
out = floor_run_(pupil, det, states, opts);
out.input = opts.input;

% ---- optional coating sensitivity sweep ---------------------------------
if ~isempty(opts.coatings)
    check_sweep_(opts.coatings);
    base = out.floor.cross;
    sw = struct('label', {}, 'floor', {}, 'scope', {}, 'd_cross_rel', {}, ...
                'I_co', {}, 'I_cross', {}, 'I_long', {});
    for k = 1:numel(opts.coatings)
        set_coatings_(opts.coatings{k});
        r = floor_run_(pupil, det, states, opts);
        lbl = sprintf('set %d', k);
        if isfield(opts.coatings{k}, 'label') && ~isempty(opts.coatings{k}(1).label)
            lbl = opts.coatings{k}(1).label;
        end
        sw(k) = struct('label', lbl, 'floor', r.floor, 'scope', r.scope, ...
                       'd_cross_rel', (r.floor.cross - base) / base, ...
                       'I_co', r.I_co, 'I_cross', r.I_cross, 'I_long', r.I_long);
    end
    out.sweep = sw;
end
end

% =========================================================================
function r = floor_run_(pupil, det, states, opts)
% One full floor computation over the run set, at the current coatings.
ns = numel(states);
ps = struct('state', {}, 'analyzer', {}, 'complement', {}, ...
            'coherency', {}, 'dop', {}, 'power', {});
I_co = []; I_cross = []; I_long = []; I_eng = [];
par = 0; clo = 0;
gcf_ = nan(1, ns);  rcf_ = nan(1, ns);     % scope, PER STATE (see below)
pmask = [];

for k = 1:ns
    v = states{k};
    macos.polarization('on', 'Ex', [real(v(1)) imag(v(1))], ...
                             'Ey', [real(v(2)) imag(v(2))]);
    macos.vector_diffraction(true);

    % --- pupil grid planes -> coherency -> analyzer ----------------------
    P1 = macos.complex_field(pupil, 'plane', 1);
    P2 = macos.complex_field(pupil, 'plane', 2, 'reset_trace', false);
    Pt = abs(P1).^2 + abs(P2).^2;
    pm = Pt > opts.pupil_tol * max(Pt(:));
    if ~any(pm(:))
        error('macos:pol_contrast_floor:emptyPupil', ...
            'no pupil pixels above ''pupil_tol'' at element %d', pupil);
    end
    Cg = coherency_(P1(pm), P2(pm));
    [a, lam] = dominant_(Cg);
    b = [-conj(a(2)); conj(a(1))];         % [a b] unitary -> exact Parseval
    dop = (lam(1) - lam(2)) / max(lam(1) + lam(2), realmin);
    if dop < opts.dop_min
        warning('macos:pol_contrast_floor:dop', ...
            ['pupil degree of polarization %.4g < dop_min %.4g for input ' ...
             'state %d -- the dominant eigenvector is not a well-defined ' ...
             'analyzer and the co/cross split is ill-conditioned.'], ...
            dop, opts.dop_min, k);
    end
    if isempty(pmask), pmask = pm; else, pmask = pmask | pm; end
    gcf_(k) = lam(2) / max(lam(1) + lam(2), realmin);

    % --- ray-level coherency at the same element (scope diagnostic) ------
    macos.trace(pupil);
    rf = macos.ray_field(pupil);
    rk = rf.status == 0;
    if any(rk(:))
        Cr = coherency_(rf.Ex(rk), rf.Ey(rk));
        [~, lr] = dominant_(Cr);
        rcf_(k) = lr(2) / max(lr(1) + lr(2), realmin);
    end

    % --- detector component planes ---------------------------------------
    D1 = macos.complex_field(det, 'plane', 1);
    D2 = macos.complex_field(det, 'plane', 2, 'reset_trace', false);
    D3 = macos.complex_field(det, 'plane', 3, 'reset_trace', false);
    Ie = macos.intensity(det);             % independent engine total

    co = abs(conj(a(1))*D1 + conj(a(2))*D2).^2;
    cr = abs(conj(b(1))*D1 + conj(b(2))*D2).^2;
    lo = abs(D3).^2;

    pk = max(co(:) + cr(:) + lo(:));
    par = max(par, max(abs(co(:) + cr(:) - (abs(D1(:)).^2 + abs(D2(:)).^2))) / pk);
    clo = max(clo, max(abs(co(:) + cr(:) + lo(:) - Ie(:))) / pk);

    if isempty(I_co)
        I_co = co;  I_cross = cr;  I_long = lo;  I_eng = Ie;
    else
        I_co = I_co + co;  I_cross = I_cross + cr;
        I_long = I_long + lo;  I_eng = I_eng + Ie;
    end
    ps(k) = struct('state', v, 'analyzer', a, 'complement', b, ...
                   'coherency', Cg, 'dop', dop, ...
                   'power', struct('co', sum(co(:)), 'cross', sum(cr(:)), ...
                                   'long', sum(lo(:))));
end

I_total = I_co + I_cross + I_long;
peak = max(I_co(:));

% ratio maps: NaN where the denominator is small -- never zero-filled
small = I_total < opts.floor_tol * max(I_total(:));
frac = I_cross ./ I_total;
frac(small) = NaN;

fl = struct('co', sum(I_co(:)), 'cross', sum(I_cross(:)), ...
            'long', sum(I_long(:)));
fl.cross_over_co = fl.cross / fl.co;
fl.contrast_cross_peak = max(I_cross(:)) / peak;
if ~isempty(opts.dark_zone)
    fl.dark_zone = dark_zone_(I_co, I_cross, I_long, peak, opts.dark_zone);
end

% Scope is judged PER INPUT STATE and reported at its WORST.  Averaging the
% states hides the shortfall: on Rx_Coro the x run carries 0.84 and the y
% run over-reads, and their power-weighted mean is 1.02 -- which would
% declare a chain healthy that is not.
% A cross-polarized fraction at or below NEG is round-off, not physics --
% there is nothing for the grid to fail to carry, so the ratio would be
% 0/0.  Declare full carry rather than dividing.
NEG = 1e-12;
carried = ones(1, ns);
for k = 1:ns
    if isnan(rcf_(k)), continue, end
    if rcf_(k) <= NEG && gcf_(k) <= NEG, continue, end
    carried(k) = gcf_(k) / max(rcf_(k), realmin);
end
[~, iw] = max(abs(carried - 1));
worst = carried(iw);
full_chain = all(abs(carried - 1) <= opts.scope_tol);
if ~full_chain
    warning('macos:pol_contrast_floor:tranche1', ...
        ['the diffraction grid carries %.4g of the ray-level cross-' ...
         'polarized fraction at element %d (grid %.4g vs ray %.4g, ' ...
         'input state %d of %d).  Phase 3a Tranche 1 freezes the ' ...
         'component planes at the first physical-optics leg, so ' ...
         'polarizing surfaces after it act on rays only -- this floor ' ...
         'is a LOWER BOUND until Tranche 2.'], ...
        worst, pupil, gcf_(iw), rcf_(iw), iw, ns);
end

r = struct('I_co', I_co, 'I_cross', I_cross, 'I_long', I_long, ...
           'I_total', I_total, ...
           'contrast_co', I_co/peak, 'contrast_cross', I_cross/peak, ...
           'contrast_long', I_long/peak, 'frac_cross', frac, ...
           'floor', fl, 'per_state', ps, ...
           'scope', struct('grid_cross_frac', gcf_, ...
                           'ray_cross_frac', rcf_, ...
                           'carried', carried, 'worst', worst, ...
                           'full_chain', full_chain), ...
           'checks', struct('parseval', par, 'closure', clo, ...
                            'engine_total', sum(I_eng(:))), ...
           'pupil', pupil, 'det', det, 'mask', pmask);
end

% =========================================================================
function st = input_states_(spec)
if ischar(spec) || isstring(spec)
    switch lower(char(spec))
        case 'x',            st = {[1; 0]};
        case 'y',            st = {[0; 1]};
        case 'unpolarized',  st = {[1; 0], [0; 1]};
        otherwise
            error('macos:pol_contrast_floor:input', ...
                '''input'' must be ''x'', ''y'', ''unpolarized'' or [ex; ey].');
    end
    return
end
v = spec(:);
if numel(v) ~= 2 || ~any(abs(v))
    error('macos:pol_contrast_floor:input', ...
        '''input'' vector must be a nonzero 2-element complex Jones state.');
end
st = {complex(double(v)) / norm(v)};
end

function C = coherency_(Ex, Ey)
% C_ij = sum E_i conj(E_j).  MIND THE ORDER: in MATLAB's ' the CONJUGATE
% sits on the LEFT operand, so C_12 = sum E_1 conj(E_2) is Ey'*Ex, not
% Ex'*Ey.  Getting it backwards builds conj(C), whose dominant eigenvector
% is the CONJUGATE analyzer -- identical for any linear state (real
% eigenvector) and exactly orthogonal to the truth for a circular one.
% test_analyzer_tracks_input_state's circular case is what catches it.
Ex = Ex(:);  Ey = Ey(:);
C = [Ex'*Ex, Ey'*Ex; Ex'*Ey, Ey'*Ey];
C = (C + C')/2;                            % hermitian by construction
end

function [a, lam] = dominant_(C)
[V, D] = eig(C);
[lam, ix] = sort(real(diag(D)), 'descend');
lam = max(lam, 0);
a = V(:, ix(1));
[~, ip] = max(abs(a));
a = a * exp(-1i*angle(a(ip)));             % fix the arbitrary global phase
a = a / norm(a);
end

function dz = dark_zone_(I_co, I_cross, I_long, peak, rad)
N = size(I_co, 1);
c = (N - 1)/2;
[xx, yy] = meshgrid(0:N-1, 0:N-1);
rr = hypot(xx - c, yy - c);
in = rr >= rad(1) & rr <= rad(2);
dz = struct('r_in', rad(1), 'r_out', rad(2), 'n_pix', nnz(in));
for f = {'co', 'cross', 'long'}
    switch f{1}
        case 'co',    A = I_co;
        case 'cross', A = I_cross;
        case 'long',  A = I_long;
    end
    v = A(in) / peak;
    dz.(f{1}) = struct('mean', mean(v), 'median', median(v), 'peak', max(v));
end
end

function check_sweep_(sets)
% A coating can be overwritten but never CLEARED (coat_set takes >= 1
% layer), so a set cannot be undone.  Every set must therefore cover the
% SAME elements: then set k+1 fully overwrites set k and each sweep point
% is a clean configuration.  The as-loaded baseline is measured before any
% set is applied.
e0 = sort(unique([sets{1}.elt]));
for k = 2:numel(sets)
    if ~isequal(sort(unique([sets{k}.elt])), e0)
        error('macos:pol_contrast_floor:sweepElts', ...
            ['every ''coatings'' set must cover the same elements -- a ' ...
             'coating cannot be cleared, only overwritten, so sets over ' ...
             'different elements would accumulate.']);
    end
end
end

function set_coatings_(spec)
for i = 1:numel(spec)
    macos.coating(spec(i).elt, 'index', spec(i).index, ...
        'extinc', spec(i).extinc, 'thickness', spec(i).thickness);
end
end

function restore_pol_(s0)
if s0.on
    macos.polarization('on', 'Ex', [real(s0.Ex) imag(s0.Ex)], ...
                             'Ey', [real(s0.Ey) imag(s0.Ey)]);
else
    macos.polarization('off');
end
end
