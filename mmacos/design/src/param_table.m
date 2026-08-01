function T = param_table(t, opts)
%PARAM_TABLE  The stage's parameter-provenance table: what each surface IS.
%
%   T = PARAM_TABLE(t) reads the built design T (a macos.design.Telescope)
%   and returns the per-element parameter row set that a design stage must
%   publish -- radius, conic, vertex, orientation, aperture -- plus the
%   focal-plane pose measured TWO ways, each against a named reference.
%
%   THE PARAMETER TABLE IS THE SOLUTION.  WFE numbers only score it: they
%   are reproducible from the parameters, and the parameters are not
%   reproducible from them.  A stage that reports only its WFE has not
%   reported its design.  (The Rodgers deck's slide-5 lesson; every
%   reconciliation in design/rodgers1/PACKET.md ran off the parameters.)
%
%   T = PARAM_TABLE(t, 'prev', T0) also computes the DELTA against a
%   previous stage's table, matched by element NAME, so each stage
%   publishes what IT changed rather than restating the whole design.
%   Elements absent from T0 are marked new.
%
%   FRAME BEFORE ANGLE.  The focal plane gets both angles it has, because
%   they drive different things and have been confused before (PACKET §4b:
%   a "14.3 deg tilted detector" that was the same surface as CODE V's
%   -0.07 deg, measured against a different reference):
%     .fp_beam_aoi_deg  the arriving CHIEF ray vs the detector normal --
%                       the FPA acceptance angle, the assembly driver
%     .fp_axis_deg      the detector normal vs the OPTICAL AXIS -- the
%                       mechanical mount requirement
%   Both are measured live from the trace, not inferred from the spec.
%
%   HELD vs SOLVED.  Each row carries .prov, the builder's own provenance
%   string for that element, so a radius a solve HELD is visibly held.
%   Pass 'held' with a cellstr of parameter names to annotate further.
%
%   Name-value:
%     'prev'    a previous PARAM_TABLE result to difference against
%     'held'    cellstr of parameter names this stage held fixed
%     'title'   heading for the formatted text
%     'quiet'   suppress printing (default false)
%
%   Returns T with .row (struct array: name, kind, Kr_m, Kc, Vpt_m,
%   psi, tilt_deg, ap_r_m, prov), .fp_beam_aoi_deg, .fp_axis_deg,
%   .held, .title, .text (the formatted block, for the stage report).
%
%   .tilt_deg is each element's own normal vs the optical axis (+z),
%   signed in the y-z plane by atan2d(psi_y, -psi_z) -- the same
%   convention design/rodgers1 decodes CODE V's ADE into.
%
%   See also design_report, macos.design.Telescope.

    arguments
        t
        opts.prev  struct = struct([])
        opts.held  (1,:) cell = {}
        opts.title (1,:) char = 'PARAMETER PROVENANCE'
        opts.quiet (1,1) logical = false
    end

    e = t.spec.elt;
    n = numel(e);
    row = struct('name',{},'kind',{},'Kr_m',{},'Kc',{},'Vpt_m',{}, ...
                 'psi',{},'tilt_deg',{},'ap_r_m',{},'prov',{});
    for k = 1:n
        psi = e(k).psi(:).';   psi = psi/norm(psi);
        row(k).name     = e(k).name;
        row(k).kind     = e(k).kind;
        row(k).Kr_m     = e(k).Kr;
        row(k).Kc       = e(k).Kc;
        row(k).Vpt_m    = e(k).Vpt(:).';
        row(k).psi      = psi;
        row(k).tilt_deg = atan2d(psi(2), -psi(3));
        row(k).ap_r_m   = e(k).ap_r;
        if isfield(e(k),'provenance'), row(k).prov = e(k).provenance;
        else,                          row(k).prov = ''; end
    end

    % ---- focal-plane pose, measured live, both references named -------
    ifp = find(strcmp({e.kind},'FocalPlane'), 1, 'last');
    aoi = NaN;  ax = NaN;
    if ~isempty(ifp)
        nrm = e(ifp).psi(:)/norm(e(ifp).psi);
        ax  = acosd(min(1, abs(dot(nrm, [0;0;1]))));      % vs the AXIS
        try
            tr = macos.trace(n);
            ri = macos.get_ray_info(tr.nRays);
            d1 = ri.dir(:,1);  d1 = d1/norm(d1);           % the arriving CHIEF
            aoi = acosd(min(1, abs(dot(d1, nrm))));
        catch
        end
    end
    T = struct('row',row, 'fp_elt',ifp, 'fp_beam_aoi_deg',aoi, ...
               'fp_axis_deg',ax, 'held',{opts.held}, 'title',opts.title);

    % ---- deltas vs the previous stage, matched by NAME ----------------
    T.delta = struct('name',{},'dKc',{},'dKr_m',{},'dVpt_mm',{},'dtilt_deg',{},'isnew',{});
    if ~isempty(fieldnames(opts.prev)) && isfield(opts.prev,'row')
        p = opts.prev.row;
        for k = 1:numel(row)
            j = find(strcmp({p.name}, row(k).name), 1);
            d = struct('name',row(k).name,'dKc',NaN,'dKr_m',NaN, ...
                       'dVpt_mm',[NaN NaN NaN],'dtilt_deg',NaN,'isnew',isempty(j));
            if ~isempty(j)
                d.dKc       = row(k).Kc - p(j).Kc;
                d.dKr_m     = row(k).Kr_m - p(j).Kr_m;
                d.dVpt_mm   = (row(k).Vpt_m - p(j).Vpt_m)*1e3;
                d.dtilt_deg = row(k).tilt_deg - p(j).tilt_deg;
            end
            T.delta(end+1) = d; %#ok<AGROW>
        end
    end

    % ---- format --------------------------------------------------------
    L = {};
    L{end+1} = sprintf('---------------- %s ----------------', T.title);
    L{end+1} = sprintf('%-11s %-11s %12s %12s %27s %9s %9s  %s', ...
        'element','kind','Kr [m]','Kc','Vpt (x,y,z) [m]','tilt[deg]','ap_r[m]','provenance');
    for k = 1:numel(row)
        kr = row(k).Kr_m;
        if abs(kr) > 1e20, krs = sprintf('%12s','flat');
        else,              krs = sprintf('%12.6f', kr); end
        L{end+1} = sprintf('%-11s %-11s %12s %12.6f  %8.4f %8.4f %8.4f %9.4f %9.4f  %s', ...
            row(k).name, row(k).kind, krs, row(k).Kc, row(k).Vpt_m, ...
            row(k).tilt_deg, row(k).ap_r_m, row(k).prov); %#ok<AGROW>
    end
    if ~isempty(T.delta)
        L{end+1} = '';
        L{end+1} = sprintf('%-11s %13s %13s %28s %12s', ...
            'delta vs prev','dKc','dKr [m]','dVpt (x,y,z) [mm]','dtilt [deg]');
        for k = 1:numel(T.delta)
            d = T.delta(k);
            if d.isnew
                L{end+1} = sprintf('%-11s   (new this stage)', d.name); %#ok<AGROW>
            else
                L{end+1} = sprintf('%-11s %13.3e %13.3e  %8.4f %8.4f %8.4f %12.5f', ...
                    d.name, d.dKc, d.dKr_m, d.dVpt_mm, d.dtilt_deg); %#ok<AGROW>
            end
        end
    end
    L{end+1} = '';
    L{end+1} = sprintf(['focal plane: beam AOI %.4f deg (arriving CHIEF vs the ' ...
                        'detector normal -- FPA acceptance)'], T.fp_beam_aoi_deg);
    L{end+1} = sprintf(['             mechanical %.4f deg (detector normal vs the ' ...
                        'OPTICAL AXIS -- the mount)'], T.fp_axis_deg);
    if ~isempty(T.held)
        L{end+1} = sprintf('HELD this stage: %s', strjoin(T.held, ', '));
    end
    L{end+1} = repmat('-', 1, 72);
    T.text = sprintf('%s\n', L{:});
    if ~opts.quiet, fprintf('%s', T.text); end
end
