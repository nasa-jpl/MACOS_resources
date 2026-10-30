function dm = ctb_dm(opts)
%CTB_DM  Influence-function DM model for the CTB grid-surface DMs.
%   dm = CTB_DM('ielt',IE,'ng',NG,'gdx_mm',G) builds an actuator-lattice
%   DM model for one grid-surface DM element (emit the deck first with
%   ctb_dm_rx).  The physical DOFs are actuator commands; the surface is
%   the superposition of per-actuator influence functions (the DM
%   doctrine: influence functions + grid-data surfaces, not modal
%   bases).  Commands map to a surface DISPLACEMENT grid in base units
%   (mm) on the element's ng x ng grid, applied with macos.set_elt_grid.
%
%   Influence function: Gaussian with nearest-neighbor coupling C,
%     IF(r) = exp(ln(C) * (r/pitch)^2)         (IF(pitch) = C)
%   -- the standard parameterization (C ~ 0.10-0.15 for continuous
%   facesheet DMs).  Stamped on a local window (support truncated at
%   |dx|,|dy| > WIN*pitch) so surface assembly is O(nnz(a)), not a
%   dense [ng^2 x nact^2] basis.
%
%   Name-value:
%     'ielt'       DM element index (required)
%     'ng'         element grid size (required; = ctb_dm_rx out.ng)
%     'gdx_mm'     element grid spacing (required; = out.gdx_mm(k))
%     'nact'       actuators across the lattice (default 32)
%     'pitch_mm'   actuator pitch (default beam_d_mm/nact)
%     'beam_d_mm'  controlled beam diameter on the DM (default 21.3, the
%                  measured CTB footprint at DM1/DM2 -- gate1b probe)
%     'coupling'   nearest-neighbor coupling C (default 0.12)
%     'win'        stamp half-width in pitches (default 3)
%     'base_mm'    static base figure grid added under the DM surface
%                  (default zeros: e.g. an as-built map later)
%
%   dm fields / methods:
%     .nact, .pitch_mm, .acx, .acy   lattice geometry (act centers, mm,
%                                    element-local x/y about pData)
%     .active                        logical [nact^2]: centers within
%                                    beam_r + 1 pitch (the controlled set)
%     .nact_active                   nnz(active)
%     .surface(a)   [ng ng] mm surface from commands a ([nact^2] or
%                   [nact nact], mm of surface displacement)
%     .apply(a)     surface(a) + base -> macos.set_elt_grid (REPLACES the
%                   element grid; no accumulation state to track)
%     .clear()      apply(0)
%
%   Run:  >> r = ctb_dm_rx; dm1 = ctb_dm('ielt',r.ielt(1),'ng',r.ng, ...
%                                        'gdx_mm',r.gdx_mm(1));
%   See also: ctb_dm_rx, ctb_dm_jacobian, macos.set_elt_grid.
    arguments
        opts.ielt      (1,1) double {mustBeInteger, mustBePositive}
        opts.ng        (1,1) double {mustBeInteger, mustBePositive}
        opts.gdx_mm    (1,1) double {mustBePositive}
        opts.nact      (1,1) double {mustBeInteger, mustBePositive} = 32
        opts.pitch_mm  double = []
        opts.beam_d_mm (1,1) double {mustBePositive} = 21.3
        opts.coupling  (1,1) double {mustBeInRange(opts.coupling,0.001,0.5)} = 0.12
        opts.win       (1,1) double {mustBePositive} = 3
        opts.base_mm   double = []
    end
    if isempty(opts.pitch_mm), opts.pitch_mm = opts.beam_d_mm / opts.nact; end
    if isempty(opts.base_mm),  opts.base_mm  = zeros(opts.ng); end
    assert(isequal(size(opts.base_mm), [opts.ng opts.ng]), ...
        'ctb_dm: base_mm must be [ng x ng]');

    ng = opts.ng;  gdx = opts.gdx_mm;  p = opts.pitch_mm;  na = opts.nact;

    % element grid axes, mm about pData (engine ndgrid convention:
    % first index +x).  Center pixel = (ng+1)/2 (the DBLE center; even
    % grids sit between pixels -- same axes the deck emitter assumed).
    ax = ((1:ng) - (ng+1)/2) * gdx;

    % actuator lattice, centered on the element center
    ac = ((1:na) - (na+1)/2) * p;
    [ACX, ACY] = ndgrid(ac, ac);
    acx = ACX(:);  acy = ACY(:);
    active = hypot(acx, acy) <= opts.beam_d_mm/2 + p;

    % influence-function stamp on a local window (truncated Gaussian)
    lnC = log(opts.coupling);
    hw  = ceil(opts.win * p / gdx);              % window half-width, px
    wax = (-hw:hw) * gdx;
    [WX, WY] = ndgrid(wax, wax);

    dm = struct();
    dm.ielt = opts.ielt;  dm.ng = ng;  dm.gdx_mm = gdx;
    dm.nact = na;  dm.pitch_mm = p;  dm.coupling = opts.coupling;
    dm.beam_d_mm = opts.beam_d_mm;
    dm.acx = acx;  dm.acy = acy;  dm.active = active;
    dm.nact_active = nnz(active);
    dm.base_mm = opts.base_mm;
    dm.surface = @surface_;
    dm.apply   = @(a) macos.set_elt_grid(opts.ielt, gdx, ...
                                         surface_(a) + opts.base_mm);
    dm.clear   = @() macos.set_elt_grid(opts.ielt, gdx, opts.base_mm);

    function S = surface_(a)
        if isscalar(a), a = a * ones(na*na, 1); end
        a = a(:);
        assert(numel(a) == na*na, ...
            'ctb_dm: commands must be [%d] or [%d x %d]', na*na, na, na);
        % complex passes MATLAB 'double' validation and the mex layer
        % silently keeps only the real part -- a complex-EFC solve slip
        % then shows up as an uncorrelated achieved field, not an error.
        % Solve EFC in the real-stacked form ([Re G; Im G]) instead.
        assert(isreal(a), 'ctb_dm: commands must be REAL');
        S = zeros(ng);
        for k = find(a.' ~= 0)
            % nearest grid node to the actuator center
            [~, i0] = min(abs(ax - acx(k)));
            [~, j0] = min(abs(ax - acy(k)));
            ii = i0-hw : i0+hw;   jj = j0-hw : j0+hw;
            m  = ii >= 1 & ii <= ng;   mj = jj >= 1 & jj <= ng;
            % stamp centered on the TRUE actuator position (not the node):
            % offsets of the window nodes from the actuator center
            dxk = ax(ii(m))  - acx(k);
            dyk = ax(jj(mj)) - acy(k);
            [DXK, DYK] = ndgrid(dxk, dyk);
            S(ii(m), jj(mj)) = S(ii(m), jj(mj)) + ...
                a(k) * exp(lnC * (DXK.^2 + DYK.^2) / p^2);
        end
    end
end
