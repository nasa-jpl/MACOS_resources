function out = proper_ctb_check(mode, opts)
%PROPER_CTB_CHECK  External-user interface check: consume the CTB phase
%   export in a MATLAB PROPER run and verify it plane by plane.
%
%   This is the DEMONSTRATION an external PROPER user runs.  It reads ONLY
%   the exported .mat (ctb_phase_export_N1024.mat) -- NO mmacos, no macos
%   deck, no engine.  It requires MATLAB PROPER (~/dev/proper_matlab); if
%   PROPER is absent it prints a skip message and returns empty (like the
%   ctb_proper_compare arbiter).
%
%   out = PROPER_CTB_CHECK(MODE) with MODE:
%     's2s'       replicate every inter-optic propagation.  Through-focus
%                 and FarField legs are replayed in PROPER from the exported
%                 FEEDING SPHERE (spheres struct): prop_begin at the sphere,
%                 inject its field, prop_lens(R)+prop_propagate(R) -> compare
%                 to the exported focus-station field.  Collimated NFPlane
%                 p2p legs are replayed with prop_propagate from the pupil.
%     'collapsed' ignore the inter-optic legs: consume the exported field E
%                 directly at each station as the hand-off (compact-model
%                 style), replay ONLY the through-focus legs.  The measured
%                 cost of ignoring s2s.
%
%   METRICS per station (printed as a table, returned in out.rows):
%     corr_I   peak-normalised INTENSITY correlation  -- the robust,
%              reference-phase-immune metric; the primary gate.
%     rms_I    RMS intensity error (fraction of peak).
%     corr_E   RAW COMPLEX-FIELD correlation -- EXPOSES the reference-phase
%              convention gap on collimated legs (see below); ~1 at focus
%              stations, can be strongly negative on collimated pupil legs.
%
%   *** THE CONVENTION YOU MUST UNDERSTAND (from the export meta) ***
%   macos's collimated NFPlane plane-to-plane propagator reads the field on
%   a PLANAR reference (local-plane curvature re-zeroed).  PROPER's
%   prop_propagate accumulates the full Fresnel quadratic reference-sphere
%   phase.  So across a COLLIMATED pupil->pupil leg the INTENSITY matches
%   (corr_I ~ 0.95) but the RAW COMPLEX FIELDS differ by a large quadratic
%   reference-phase term (corr_E strongly negative).  This is a convention
%   difference, NOT an error (the NF p2p propagator is validated to 2.4e-14
%   macos-vs-macos).  => Judge collimated legs by corr_I, NOT corr_E.  The
%   through-focus / FarField legs, replayed from the feeding sphere, match
%   at corr_I = 1.000000 (the arbiter class).  If you need bit-faithful
%   fields at a pupil, consume the exported E directly (that is exactly what
%   'collapsed' mode does, and why it is always valid).
%
%   WHAT IS AND IS NOT REPLAYABLE (measured; MATLAB PROPER, N=1024, 500 nm):
%     - THROUGH-FOCUS + FarField legs (Focus23/FPM/FieldStop/FPA), replayed
%       from the feeding SPHERE: corr_I = 1.000000 in BOTH modes -- the
%       arbiter class.  This is the core interface guarantee.
%     - PUPIL stations in COLLAPSED mode (consume our exported E as the
%       hand-off): corr_I >= 0.96 -- the robust, always-valid path.
%     - PUPIL stations in S2S mode: only the DIRECT pupil->pupil NFPlane
%       legs replay well (Apodizer/Lyot/CheckPoint corr_I ~ 0.999); a pupil
%       fed THROUGH A POWERED OAP (e.g. ExitPupil via OAP8) does NOT replay
%       by bare prop_propagate -- the OAP's focal-length phase acts between
%       planes and is the EXTERNAL USER'S OWN prop_lens to model, not ours
%       to carry.  So s2s deliberately does NOT gate the OAP-fed pupils;
%       an external user models their OAPs as prop_lens in their own script
%       and uses our fields as the plane-by-plane cross-check.
%     - OPTIC stations (the 8 OAPs) are MID-BEAM through a powered mirror --
%       not valid hand-off planes at all; printed as 'info', never gated.
%   GATES: focus corr_I >= 0.999999 (both modes); collapsed-mode pupil
%   corr_I >= 0.94.  (s2s-mode OAP-fed pupils and all optic stations are
%   reported, not gated -- see above.)
%
%   Name-value: 'mat' (export path), 'outdir', 'visible'.
%
%   See also: ctb_phase_export, ctb_proper_compare, README.md.
    arguments
        mode (1,:) char {mustBeMember(mode,{'s2s','collapsed'})} = 's2s'
        opts.mat     (1,:) char = ''
        opts.outdir  (1,:) char = ''
        opts.visible (1,1) logical = false
        opts.figure  (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.mat)
        opts.mat = fullfile(here, 'ctb_phase_export_N1024.mat');
        if ~isfile(opts.mat)                              % fall back to preview
            pv = fullfile(here, 'ctb_phase_export_preview.mat');
            if isfile(pv)
                opts.mat = pv;
                fprintf('[check] full export absent; using the committed preview (%d px).\n', ...
                    getfield(load(pv,'meta').meta,'preview_n'));      %#ok<GFLD>
            end
        end
    end
    if isempty(opts.outdir), opts.outdir = here; end
    assert(isfile(opts.mat), 'export .mat not found: %s (run ctb_phase_export first)', opts.mat);

    have_proper = exist('prop_begin','file')==2 && exist('prop_propagate','file')==2;
    if ~have_proper
        fprintf(['[check] MATLAB PROPER not on path (~/dev/proper_matlab) -- ', ...
                 'skipping.  Add PROPER and re-run.\n']);
        out = []; return;
    end

    d = load(opts.mat);
    lam = d.meta.lambda_m; N = d.meta.N;
    fprintf('[check] mode=%s  export=%s  N=%d  lambda=%.4e m\n', mode, opts.mat, N, lam);
    fprintf('[check] OPD sign: %s\n', d.meta.opd_sign);

    % index helpers
    stn = @(nm) d.stations(find(strcmp({d.stations.name}, nm), 1));
    sph = @(nm) d.spheres(find(strcmp({d.spheres.feeds_station}, nm), 1));
    is_focus = @(k) any(strcmp(d.stations(k).kind, {'focus'}));

    % ---- verify the stored orientation assertion first ----------------
    o = d.meta.orientation;
    fprintf('[check] orientation assertion: +X pupil ramp -> FPA peak dcol=%+d drow=%+d (col%scenter)\n', ...
        o.dcol, o.drow, ternary_(o.dcol<0,'<','>='));

    % ---- walk the stations, replay each leg per mode ------------------
    rows = struct('station',{},'kind',{},'corr_I',{},'rms_I',{},'corr_E',{},'note',{});
    for k = 1:numel(d.stations)
        b = d.stations(k);
        if k == 1
            rows(k) = mkrow_(b.name,b.kind,1,0,1,'export origin (no leg)'); continue;
        end
        a = d.stations(k-1);
        leg = d.legs(k-1);
        if is_focus(k)
            % THROUGH-FOCUS / FarField: replay from the feeding sphere
            s = sph(b.name);
            wf = replay_focus_(s, lam, N);
            [cI,rI,cE] = compare_(wf, b, 'intensity');
            rows(k) = mkrow_(b.name,b.kind,cI,rI,cE, ...
                sprintf('focus: replay from sphere R=%.4f m',s.R_m));
        else
            switch mode
                case 's2s'
                    % collimated pupil / optic: PROPER prop_propagate the leg
                    wf = replay_p2p_(a, leg.chief_len_m, lam, N);
                    [cI,rI,cE] = compare_(wf, b, 'field');
                    rows(k) = mkrow_(b.name,b.kind,cI,rI,cE, ...
                        sprintf('p2p replay %.4f m (corr_E exposes ref-phase gap)',leg.chief_len_m));
                case 'collapsed'
                    % ignore the leg: consume the exported field directly
                    [cI,rI,cE] = compare_(a.E, b, 'field_handoff');
                    rows(k) = mkrow_(b.name,b.kind,cI,rI,cE, ...
                        'collapsed: prev exported E used as hand-off (leg skipped)');
            end
        end
    end

    % ---- print the table ----------------------------------------------
    fprintf('\n  %-11s | %-6s | %-9s | %-9s | %-9s | note\n', ...
        'station','kind','corr_I','rms_I','corr_E');
    fprintf('  %s\n', repmat('-',1,78));
    for k = 1:numel(rows)
        r = rows(k);
        fprintf('  %-11s | %-6s | %9.6f | %9.2e | %+9.4f | %s\n', ...
            r.station, r.kind, r.corr_I, r.rms_I, r.corr_E, r.note);
    end
    fprintf('  %s\n', repmat('-',1,78));

    % ---- gate check ----------------------------------------------------
    % Only PUPIL and FOCUS stations are valid comparison planes.  OPTIC
    % stations sit MID-BEAM through a powered mirror (converging/diverging):
    % the optic's focal-length phase acts BETWEEN planes, so a single
    % prop_propagate cannot cross it and an optic-plane field is not a valid
    % PROPER hand-off.  Optic rows are printed for completeness but flagged
    % 'info' and excluded from the gates (see README).
    foc = arrayfun(@(r) strcmp(r.kind,'focus'), rows);
    gate_focus = all([rows(foc).corr_I] >= 0.999999);
    fprintf('[check] GATE focus stations corr_I>=0.999999: %s (min %.6f)\n', ...
        tf_(gate_focus), min([rows(foc).corr_I]));

    % Pupil gate: in COLLAPSED mode gate every pupil (hand-off is always
    % valid).  In S2S mode gate only pupils reached by a DIRECT NFPlane
    % pupil->pupil leg -- OAP-fed pupils are the user's own prop_lens to
    % model (see header), so they are reported, not gated.
    pup = find(arrayfun(@(r) strcmp(r.kind,'pupil'), rows));
    if strcmp(mode,'s2s')
        keep = arrayfun(@(k) contains(d.legs(k-1).prop_type,'NFPlane'), pup);
        gpup = pup(keep);
        label = 'S2S direct-NFPlane pupil stations';
    else
        gpup = pup;
        label = 'collapsed-mode pupil stations';
    end
    gate_pupil = ~isempty(gpup) && all([rows(gpup).corr_I] >= 0.94);
    fprintf('[check] GATE %s corr_I>=0.94: %s (min %.6f)\n', ...
        label, tf_(gate_pupil), min([rows(gpup).corr_I]));
    fprintf(['[check] (optic stations = mid-beam through a powered OAP; s2s OAP-fed pupils = ', ...
             'user''s own prop_lens -- both reported, not gated. See README.)\n']);

    out = struct('mode',mode,'rows',rows,'gate_focus',gate_focus, ...
        'gate_pupil',gate_pupil,'lambda_m',lam,'N',N,'mat',opts.mat);

    if opts.figure, out.figure = plot_check_(out, opts); end
end

% ======================================================================
function figpath = plot_check_(out, opts)
%PLOT_CHECK_  Deck-grade station-by-station agreement figure for one mode.
    rows = out.rows;  n = numel(rows);
    corr_I = [rows.corr_I];  kinds = {rows.kind};
    % color by station kind
    col = zeros(n,3);
    for k=1:n
        switch kinds{k}
            case 'focus', col(k,:) = [0.10 0.45 0.80];   % blue
            case 'pupil', col(k,:) = [0.20 0.60 0.25];   % green
            otherwise,    col(k,:) = [0.70 0.70 0.72];   % grey (optic, info)
        end
    end
    vis='off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[60 60 1500 620]);
    set(fig,'DefaultAxesFontSize',15);
    ax = axes(fig); hold(ax,'on');
    b = bar(ax, 1:n, corr_I, 0.7, 'FaceColor','flat');
    b.CData = col;
    yline(ax, 0.999999, '--', 'focus gate 1.000000', 'Color',[0.10 0.45 0.80], ...
        'LabelHorizontalAlignment','left','FontSize',12);
    yline(ax, 0.94, ':', 'pupil gate 0.94', 'Color',[0.20 0.60 0.25], ...
        'LabelHorizontalAlignment','left','FontSize',12);
    set(ax,'XTick',1:n,'XTickLabel',{rows.station},'XTickLabelRotation',40);
    ylim(ax,[-0.3 1.05]); grid(ax,'on'); box(ax,'on');
    ylabel(ax,'intensity correlation  corr_I');
    title(ax, sprintf(['CTB phase-export interface check -- mode ''%s'' ', ...
        '(N=%d, 500 nm): focus %s / pupil-gate %s'], out.mode, out.N, ...
        tf_(out.gate_focus), tf_(out.gate_pupil)), 'Interpreter','none','FontSize',16);
    % legend by kind
    hf=patch(ax,nan,nan,[0.10 0.45 0.80]); hp=patch(ax,nan,nan,[0.20 0.60 0.25]);
    ho=patch(ax,nan,nan,[0.70 0.70 0.72]);
    legend(ax,[hf hp ho],{'focus (gated 1.0)','pupil (gated 0.94)', ...
        'optic / OAP-fed (info, not gated)'},'Location','southeast','FontSize',12);
    figpath = fullfile(opts.outdir, sprintf('proper_ctb_check_%s.png', out.mode));
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[check] wrote %s\n', figpath);
end

% ======================================================================
function wf = replay_focus_(s, lam, N)
%REPLAY_FOCUS_  PROPER replay of a through-focus/FarField leg from the
%   feeding sphere: inject the sphere field, lens(R)+propagate(R).
    bm = prop_begin(N * s.dx_sphere_m, lam, N, 'beam_diam_fraction',1.0);
    bm = prop_multiply(bm, s.AMP);
    bm = prop_add_phase(bm, s.OPD_m);                    % OPD_m already sign-flipped
    bm = prop_define_entrance(bm);
    bm = prop_lens(bm, s.R_m);
    bm = prop_propagate(bm, s.R_m);
    wf = prop_get_wavefront(bm);
end

function wf = replay_p2p_(a, len_m, lam, N)
%REPLAY_P2P_  PROPER replay of a collimated plane-to-plane leg: inject the
%   pupil field, propagate the chief-ray length.  (corr_E will expose the
%   reference-phase convention gap -- that is the point.)
    bm = prop_begin(N * a.dx_m, lam, N, 'beam_diam_fraction',1.0);
    bm = prop_multiply(bm, a.E);                          % inject complex E
    bm = prop_define_entrance(bm);
    bm = prop_propagate(bm, len_m);
    wf = prop_get_wavefront(bm);
end

function [cI, rI, cE] = compare_(wf, b, kind)
%COMPARE_  Intensity corr + RMS + raw complex-field corr vs station b.
    Iw = abs(wf).^2;  Ib = b.AMP.^2;
    cI = ncorr_(Iw, Ib);
    rI = norm(Iw(:)/max(Ib(:)+eps) - Ib(:)/max(Ib(:)+eps)) / sqrt(numel(Ib));
    cE = ncorr_c_(wf, b.E);
    switch kind
        case 'field_handoff'
            % hand-off: wf IS the previous exported E; corr_E vs this station
            % measures how similar consecutive planes are (leg skipped).
        otherwise
    end
end

function c = ncorr_(A, B)
%NCORR_  Pearson correlation of two real images (mean-removed).
    a = double(A(:)); b = double(B(:));
    a = a - mean(a); b = b - mean(b);
    c = (a'*b) / (norm(a)*norm(b) + eps);
end

function c = ncorr_c_(A, B)
%NCORR_C_  Complex-field correlation (Hermitian, magnitude of normalised
%   inner product's real part) -- 1 = identical field, <0 = anti-phased.
    a = A(:); b = B(:);
    c = real(a'*b) / (norm(a)*norm(b) + eps);
end

function r = mkrow_(station, kind, cI, rI, cE, note)
    r = struct('station',station,'kind',kind,'corr_I',cI,'rms_I',rI, ...
               'corr_E',cE,'note',note);
end

function v = ternary_(c,a,b), if c, v=a; else, v=b; end, end
function s = tf_(b), if b, s='PASS'; else, s='FAIL'; end, end
