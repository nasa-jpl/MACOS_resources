function chk = jacobian_check(rx, ox, opts)
%JACOBIAN_CHECK  Engine-vs-Jacobian closure check AT THE HARVEST SURFACE.
%
%   chk = jacobian_check(RX, OX, ...) pokes rigid-body DOFs on the
%   engine and compares the OPD change against the matching column of a
%   dw_dx / dw_dx_multi harvest OX.  The pokes ARE the model, so every
%   sampled DOF must close at finite-difference level -- IF the engine
%   OPD is evaluated where the harvest evaluated it.
%
%   THE ONE RULE THIS FUNCTION EXISTS TO ENFORCE: the engine trace runs
%   to OX.wf_elt, the surface the Jacobian's OPD rows live on.  Pairing
%   the Jacobian with an OPD traced to any OTHER element is the e2e6m
%   round-1 "slide 11" defect: on that deck, a segment tilt is a clean
%   tilt-about-center at the exit pupil (nElt-1) and a segment-footprint
%   PISTON with lever = the segment's pupil radius at the Science FOCAL
%   plane (nElt).  Same poke, same frames -- different surface.  Piston
%   (Tz) closes at any surface, which is exactly what made the defect
%   look like a frame/clocking error.  See
%   templates/80_end_to_end/e2e6m_r2/e2e6m_r2_LOG.md (R0.1).
%
%   OX may be a single-field artifact (fields dwdx / w_nom_2d) or a
%   multi-field one (per_field_dwdx / per_field_w_nom_2d + field_names);
%   for multi-field the 'field' option picks the block ('C' default --
%   the check runs at the deck's NOMINAL source state, which IS the
%   center field).
%
%   Name-value options:
%     'elts'     elements to poke (default: every RigidBody element in
%                OX -- restrict for cost).
%     'dofs'     0-based DOF subset (default (0:5)': Rx..Tz).
%     'd_rot'    rotation poke, rad (default 1e-9).
%     'd_trans'  translation poke, SI metres (default 1e-9).
%     'field'    multi-field block name (default 'C').
%     'model'    model size for macos.init; [] = use the live session
%                state (default []).
%     'wf_elt'   OVERRIDE of the evaluation element.  Default
%                OX.wf_elt.  Overriding exists so a test can PIN the
%                wrong-surface failure mode; production callers should
%                never pass it.
%     'w_floor'  null-response floor, SI metres (default 1e-12, the
%                run_compare convention): a poke whose engine response
%                is below it reports rel = NaN ('null') instead of an
%                FD-noise ratio.  e5hex1's segment Rz (clocking) is the
%                canonical case: 1.7e-13 m per nrad.
%     'verbose'  print per-DOF lines (default false).
%
%   Output CHK: .elt/.dof/.n_eng/.n_mod/.rel row vectors over the
%   sampled (element, DOF) pairs, plus .worst, .wf_elt, .tags.
%
%   See also: macos.dw_dx, macos.dw_dx_multi, run_sensitivities,
%   run_compare.

    arguments
        rx (1,:) char {mustBeNonempty}
        ox struct
        opts.elts    (:,1) double = []
        opts.dofs    (:,1) double = (0:5).'
        opts.d_rot   (1,1) double = 1e-9
        opts.d_trans (1,1) double = 1e-9
        opts.field   (1,:) char = 'C'
        opts.model   double {mustBeScalarOrEmpty} = []
        opts.wf_elt  double {mustBeScalarOrEmpty} = []
        opts.w_floor (1,1) double {mustBePositive} = 1e-12
        opts.verbose (1,1) logical = false
    end
    assert(isfile(rx), 'jacobian_check: %s not found', rx);

    % ---- pick the Jacobian block ----------------------------------------
    if isfield(ox, 'per_field_dwdx')
        ic = find(strcmp(ox.field_names, opts.field), 1);
        assert(~isempty(ic), 'jacobian_check: no field ''%s'' in OX', ...
               opts.field);
        A0   = ox.per_field_dwdx{ic};
        Wnom = ox.per_field_w_nom_2d{ic};
    else
        A0   = ox.dwdx;
        Wnom = ox.w_nom_2d;
    end
    mnom = finite_(Wnom);
    idx  = find(mnom);
    assert(nnz(mnom) == size(A0,1), ...
        'jacobian_check: nominal mask (%d px) vs Jacobian rows (%d)', ...
        nnz(mnom), size(A0,1));

    kind = ox.kind(:);
    isrb = strcmp(kind, 'RigidBody');
    elts = opts.elts;
    if isempty(elts), elts = unique(ox.iElt(isrb), 'stable'); end

    wf = opts.wf_elt;
    if isempty(wf)
        assert(isfield(ox, 'wf_elt'), ...
            'jacobian_check: OX carries no wf_elt -- re-harvest');
        wf = ox.wf_elt;
    end

    % ---- engine at nominal ----------------------------------------------
    if ~isempty(opts.model), macos.init(opts.model); end
    macos.load_rx(rx);
    macos.trace(wf);
    W0 = macos.opd();
    % the OPD (and so the floor comparison) is in deck BaseUnits
    if isfield(ox, 'cbm'), cbm = ox.cbm; else, cbm = macos.cbm(); end
    floor_bu = opts.w_floor / cbm;

    dofn = {'Rx','Ry','Rz','Tx','Ty','Tz'};
    chk = struct('elt',[], 'dof',[], 'n_eng',[], 'n_mod',[], 'rel',[]);
    tags = {};
    for e = elts(:).'
        for j0 = opts.dofs(:).'
            q = find(ox.iElt == e & ox.dof_idx == j0 & isrb, 1);
            if isempty(q), continue; end
            j = j0 + 1;
            a = opts.d_rot;  if j > 3, a = opts.d_trans; end
            d = zeros(6,1);  d(j) = a;
            macos.perturb(e, 'rotation', d(1:3), 'translation', d(4:6), ...
                          'frame','local');
            macos.modify();  macos.trace(wf);
            W1 = macos.opd();
            macos.perturb(e, 'rotation', -d(1:3), 'translation', -d(4:6), ...
                          'frame','local');
            macos.modify();
            b    = mnom & finite_(W1);
            rows = find(ismember(idx, find(b)));
            v1 = W1(b);  v1 = v1 - mean(v1);
            v0 = W0(b);  v0 = v0 - mean(v0);
            dW = v1 - v0;
            md = A0(rows, q) * a;  md = md - mean(md);
            chk.elt(end+1)   = e; %#ok<*AGROW>
            chk.dof(end+1)   = j0;
            chk.n_eng(end+1) = rms_(dW);
            chk.n_mod(end+1) = rms_(md);
            if rms_(dW) < floor_bu
                chk.rel(end+1) = NaN;      % null response -- no physics to close
            else
                chk.rel(end+1) = rms_(dW - md) / rms_(dW);
            end
            tags{end+1} = sprintf('elt %d %s', e, dofn{j});
            if opts.verbose
                fprintf('  elt %2d %s: |engine| %.4g  |model| %.4g  rel %.3g\n', ...
                        e, dofn{j}, chk.n_eng(end), chk.n_mod(end), ...
                        chk.rel(end));
            end
        end
    end
    assert(~isempty(chk.rel), 'jacobian_check: no matching columns in OX');
    assert(any(~isnan(chk.rel)), ...
        'jacobian_check: every sampled response is below the null floor');
    chk.worst  = max(chk.rel);              % max ignores the NaN nulls
    chk.n_null = nnz(isnan(chk.rel));
    chk.wf_elt = wf;
    chk.tags   = tags;
end

function m = finite_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v)
    v = v(:);  if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end
end
