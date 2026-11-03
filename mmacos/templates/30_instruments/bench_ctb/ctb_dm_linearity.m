function R = ctb_dm_linearity(ch, J, a, pattern, amps_nm, opts)
%CTB_DM_LINEARITY  Is the plant linear in a DM pattern at the sub-nm scale?  Counts the rays that step.
%   R = CTB_DM_LINEARITY(CH, J, A, PATTERN, AMPS_NM) applies A + s*PATTERN
%   to DM 'probe_dm' of J (a ctb_dm_jacobian struct, or a cell of ctb_dm objects) (default 1) at each amplitude in
%   AMPS_NM (peak, nm) and both signs, traces to the plane 'plane' (default
%   the chain's Apodizer, pre-mask) and reports, per amplitude:
%     odd_dev     ||odd/A - odd_ref/A_ref|| / ||odd_ref/A_ref||, the odd
%                 (first + third order) field response per unit amplitude
%                 against the smallest amplitude's (0 = linear)
%     n_step      rays whose path length to the plane changed by more than
%                 'step_factor' (default 10) times the pattern's own
%                 largest OPD (2 A, a reflector) -- a ray that STEPPED, not
%                 one the pattern moved
%     max_step_nm the largest such per-ray path-length change
%     worst_ray, worst_elt   the ray with the largest step and the element
%                 at which its leaving direction first departs by more than
%                 'kink_rad' (1e-5) from the unperturbed trace: the
%                 discontinuity's site (a grid DM)
%   A plant made of C0 (bilinear) grid surfaces is piecewise linear in the
%   DM state: a ray whose footprint on a downstream grid DM crosses a cell
%   boundary takes the two cells' slope difference as a direction kink
%   (1e-6 .. 1e-4 rad on a 10-nm-rms state) and walks off its path by
%   10-100 nm of OPD for a sub-nm change (e2e stage 11's cycle-4 state:
%   ray 21226 steps 30 nm at the apodizer between 0.42 and 0.43 nm of
%   probe 1; 68 rays by 1 nm, 156 by 2 nm).  The probe estimate at depth
%   and the Jacobian's probe model carry those steps (PLAN_CONSOLIDATION 3a,
%   CCL 2026-10-31).  Zero steps at every amplitude is the gate a
%   C1 (continuous-normal) grid interpolation must pass.
%
%   Name-value: 'probe_dm' (1), 'plane' (ch.elt.Apodizer), 'step_factor' (10), 'kink_rad' (1e-5),
%   'u_nm' (1e9: nm per deck unit; the e2e deck is in metres), 'say' (printer; [] = silent).
    arguments
        ch struct
        J                                   % a ctb_dm_jacobian struct (its .dm geometry) or a cell of ctb_dm objects
        a  cell
        pattern (:, 1) double
        amps_nm (1, :) double {mustBePositive}
        opts.probe_dm (1, 1) double = 1
        opts.plane    (1, 1) double = ch.elt.Apodizer
        opts.step_factor (1, 1) double = 10
        opts.u_nm     (1, 1) double = 1e9
        opts.kink_rad (1, 1) double = 1e-5     % a cell-edge kink is 1e-5..1e-4 rad; a 5-nm pattern's own slope 5e-6
        opts.say = @(varargin) fprintf('%s\n', sprintf(varargin{:}))
    end
    say = opts.say;  if isempty(say), say = @(varargin) []; end
    if iscell(J)
        dm = J;
    else
        dm = cell(1, numel(J.dm));
        for k = 1:numel(J.dm)
            d = J.dm(k);
            dm{k} = ctb_dm('ielt', d.ielt, 'ng', d.ng, 'gdx_mm', d.gdx_mm, 'nact', d.nact, ...
                           'pitch_mm', d.pitch_mm, 'beam_d_mm', d.beam_d_mm, 'coupling', d.coupling);
        end
    end
    apply = @(c) cellfun(@(q, v) q.apply(v), dm, c, 'UniformOutput', false);
    s = pattern / max(abs(pattern));
    amps_nm = sort(amps_nm);  nA = numel(amps_nm);
    apply(a);  t = macos.trace(opts.plane);  nR = t.nRays;  i0 = macos.get_ray_info(nR);
    E0 = double(macos.complex_field(opts.plane));
    R = struct('amps_nm', amps_nm, 'odd_dev', nan(1, nA), 'n_step', zeros(1, nA), 'max_step_nm', zeros(1, nA), ...
               'worst_ray', zeros(1, nA), 'worst_elt', zeros(1, nA), 'odd', {cell(1, nA)});
    say('[linearity] %d rays; pattern on DM%d, plane elt %d; a step is a ray whose path changed by > %g x 2A', nR, opts.probe_dm, opts.plane, opts.step_factor);
    say('    %-8s %-12s %-8s %-12s %-10s %-8s', 'A nm', 'odd_dev', 'steps', 'max step nm', 'worst ray', 'at elt');
    for i = 1:nA
        A = amps_nm(i);  Ep = cell(1, 2);  st = zeros(nR, 1);
        for sg = [1 -1]
            at = a;  at{opts.probe_dm} = at{opts.probe_dm} + sg * (A / opts.u_nm) * s;  apply(at);
            macos.trace(opts.plane);  ri = macos.get_ray_info(nR);
            d = abs(ri.opl - i0.opl) * opts.u_nm;  ok = ri.ok_trace & i0.ok_trace;
            st = max(st, d .* ok);
            Ek = double(macos.complex_field(opts.plane));  Ep{(3 - sg) / 2} = Ek;
        end
        R.odd{i} = (Ep{1} - Ep{2}) / 2 / A;
        stepped = st > opts.step_factor * 2 * A;
        R.n_step(i) = nnz(stepped);
        if any(stepped)
            [R.max_step_nm(i), q] = max(st .* stepped);  R.worst_ray(i) = q;
            % the element where the worst ray's direction departs
            at = a;  at{opts.probe_dm} = at{opts.probe_dm} + (A / opts.u_nm) * s;  apply(at);  macos.trace();  r1 = macos.ray(q);
            apply(a);  macos.trace();  r0 = macos.ray(q);
            dd = sqrt(sum((r1.dir - r0.dir).^2, 1));  k = find(dd > opts.kink_rad, 1);  if isempty(k), k = 0; end
            R.worst_elt(i) = k;
        end
        R.odd_dev(i) = norm(R.odd{i}(:) - R.odd{1}(:)) / norm(R.odd{1}(:));
        say('    %-8.3f %-12.3e %-8d %-12.2f %-10d %-8d', A, R.odd_dev(i), R.n_step(i), R.max_step_nm(i), R.worst_ray(i), R.worst_elt(i));
    end
    apply(a);
    R.E0 = E0;
end
