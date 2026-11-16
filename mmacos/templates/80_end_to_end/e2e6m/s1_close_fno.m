function OUT = s1_close_fno(opts)
%S1_CLOSE_FNO  Drive the CORRECTED f/# into band by moving the base layout.
%
%   The sphere+Zernike stage of S1 spends optical POWER: mode 5 is
%   defocus, and on this train CALIB will not run without it (every
%   variant that pinned power out aborted with "DOFs for optimization are
%   correlated" and core-dumped).  So the freeform solve moves the first
%   order -- measured 2.15x on the base layout -- and the f/# gate cannot
%   be applied to the base spheres, where `s1_layout_search` reads it.
%   It has to be applied to the CORRECTED system, with the base layout as
%   the knob.
%
%   S1_CLOSE_FNO secants on the M3 BASE RADIUS, running a full
%   `s1_telescope` at each iterate and reading back the corrected f/#.
%   R3 is the right knob: it is the strongest EFL lever in this topology
%   and the weakest packaging one (M3 sits at the far end of the annulus,
%   and its own beam is 0.3 m).  The packaging gates are re-read at every
%   iterate anyway -- an f/# bought by breaking the shroud is not a
%   solution.
%
%   Each iterate writes a full artifact set into its own subdirectory, so
%   the search is auditable and the winner is simply the directory whose
%   numbers the report quotes.
%
%   Name-value:
%     'target'    corrected f/# to aim at (default 16, mid-band)
%     'band'      acceptance band (default from e2e6m_params)
%     'R3_0'      first R3 (default from e2e6m_params)
%     'R3_1'      second R3, to seed the secant (default 0.85*R3_0)
%     'iters'     outer iteration cap (default 5)
%     'outdir'    parent directory for the iterate subdirectories
%
%   See also S1_TELESCOPE, S1_LAYOUT_SEARCH, E2E6M_PARAMS.

    arguments
        opts.target (1,1) double = 16
        opts.band   (1,2) double = [0 0]
        opts.R3_0   (1,1) double = 0
        opts.R3_1   (1,1) double = 0
        opts.iters  (1,1) double = 5
        opts.outdir (1,:) char   = ''
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P0 = e2e6m_params();
    if all(opts.band == 0), opts.band = P0.fno_band; end
    if opts.R3_0 == 0, opts.R3_0 = P0.tel.R_m(3); end
    if opts.R3_1 == 0, opts.R3_1 = 0.85*opts.R3_0; end
    if isempty(opts.outdir), opts.outdir = fullfile(here, 's1_fno'); end
    if ~exist(opts.outdir,'dir'), mkdir(opts.outdir); end

    fprintf(['\n==== S1 f/# closure: secant on R3, target f/%.2f, band ' ...
             '[%.1f %.1f] ====\n'], opts.target, opts.band);
    R = struct('R3',{},'fno',{},'wfe_tilt',{},'shroud',{},'clear',{}, ...
               'dir',{},'ok',{});
    x = [opts.R3_0, opts.R3_1];
    for k = 1:opts.iters
        if k > 2
            % secant on log(f/#): the map R3 -> f/# is strongly nonlinear
            % and positive, and logs keep the step from overshooting into
            % the wrong branch on the first correction.
            f1 = log(R(k-1).fno/opts.target);
            f0 = log(R(k-2).fno/opts.target);
            if abs(f1 - f0) < 1e-9, break; end
            xn = R(k-1).R3 - f1*(R(k-1).R3 - R(k-2).R3)/(f1 - f0);
            xn = min(max(xn, 0.4*opts.R3_0), 2.5*opts.R3_0);   % keep it sane
            x(k) = xn;
        end
        r = one_(x(k), k, opts, P0);
        R(k) = r; %#ok<AGROW>
        fprintf(['iterate %d: R3 %.4f m -> corrected f/%.2f | -tilt max %.4f ' ...
                 'waves | shroud %.3f m | clear %d [%s]\n'], ...
                k, r.R3, r.fno, r.wfe_tilt, r.shroud, r.clear, gate_(r.ok));
        if r.ok, break; end
    end

    ok = find([R.ok], 1);
    if isempty(ok)
        [~, ok] = min(abs(log([R.fno]/opts.target)));
        fprintf('\nNo iterate met every gate; closest on f/# is iterate %d.\n', ok);
    else
        fprintf('\nCLOSED at iterate %d: %s\n', ok, R(ok).dir);
    end
    OUT = struct('R',R, 'best',ok, 'opts',opts, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(opts.outdir,'s1_close_fno.mat'),'OUT');
end

% =========================================================================
function r = one_(R3, k, opts, P0)
    d = fullfile(opts.outdir, sprintf('it%02d', k));
    if ~exist(d,'dir'), mkdir(d); end
    tel = P0.tel;  tel.R_m(3) = R3;
    O = s1_telescope(struct('outdir', d, 'tel', tel));
    r = struct('R3',R3, 'fno',O.fno, ...
               'wfe_tilt', max(O.map.rms_tilt), ...
               'shroud', 2*O.pack.shroud_radius_m, ...
               'clear', all([O.clip.ok]), 'dir', d, 'ok', false);
    r.ok = r.fno >= opts.band(1) && r.fno <= opts.band(2) && ...
           r.shroud <= P0.shroud_D_m && r.clear && ...
           r.wfe_tilt <= P0.dl_waves;
end

function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function s = gate_(ok), if ok, s = 'ALL GATES'; else, s = 'not yet'; end, end
