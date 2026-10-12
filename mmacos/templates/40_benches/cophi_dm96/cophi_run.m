function out = cophi_run(varargin)
%COPHI_RUN  The COPHI (heterodyne camera interferometer) DM gauge on the TG96
%   bench: the reference flat's seat taken by a TRACED, fiber-fed LO arm
%   (Dave's ruling 1; twyman_green 'ref_arm','lo').  One parameterized runner
%   (Dave's standing rule), on tg96_run's bones: the bench itself -- clearance
%   solve, build, emit, figures, the part-by-part clearance table -- IS
%   tg96_run's, called with P.bench.ref_arm = 'lo'; this file adds what COPHI
%   needs on top.
%
%   Usage (interactive; NO exit -- the -batch wrapper is cophi_run_batch):
%     cophi_run                                   % the sheet (cophi_params), lens rig
%     cophi_run('bench.optics','oap','tag','cophi_oap')   % mirror rig
%     cophi_run('stages',{'bench'})               % one stage
%
%   Stages (P.stages; the brief's order, BRIEF_to_cophi.md):
%     bench    tg96_run's bench (+ figs / clearance when listed) with the LO arm
%              placed, then the REGISTRATION of the two arms at the camera: the
%              detector grid pitch of each deck, the chief-ray pixel of each,
%              the DM->camera and seat->camera magnifications (dmg_frame on each
%              deck), the LO's footprint over the test arm's pupil image, the
%              LO's own wavefront there, and the null phase on the flat DM.
%              The two must agree to a pixel (BRIEF_to_cophi sec. 1).
%     (battery, noise, loop, capture, format, figs: to come, one commit each)
%
%   COPHI's bench = the TG96 bench of record with NO polarization phase
%   shifting: the input polarizer stays as a SOURCE polarizer, the arm
%   quarter-wave plates are set to zero retardance (they are then the record's
%   2 mm silica windows, so the glass -- and the tail of record tuned with it --
%   is unchanged), the output QWP likewise, and the analyzer is turned to PASS
%   the test arm's state (solved from three traces, T(theta) = a + b cos 2theta
%   + c sin 2theta) -- a clean-up polarizer before the camera that both arms
%   share.  The LO's launch state is the record's (Ex = Ey); what the analyzer
%   passes of it is MEASURED and reported, and is the LO power the fiber's
%   orientation has to supply.  So the polarization gain the record's snapshot
%   reading carries is absent by construction and the battery MEASURES that.
%
%   Writes runs/<tag>/<tag>_cophi_report.txt and <tag>_cophi.mat beside
%   tg96_run's own <tag>_report.txt / decks / figures.

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(exdir);
P = cophi_params();                          % adds tg_psi_dm96_oap + dm_gauge_lib
P = apply_overrides_(P, varargin{:});
% the bench OF RECORD: the deck's IFO rows (redo96_lens / redo96_oap) ran on the
% redo tails, so COPHI names them explicitly (tg96_run bench.tail_file)
if ~isfield(P.bench, 'tail_file') || isempty(P.bench.tail_file)
    P.bench.tail_file = P.cophi.record_tail.(P.bench.optics);
end
if strcmp(P.bench.optics, 'oap') && P.cophi.oap_record
    % the MIRROR RIG OF RECORD -- what every record run passes explicitly
    % (tg_psi_dm96_oap/runs/redo96_ph_seq.sh, closefinal5.sh): not the sheet's
    % Stage-A defaults, which put the input polarizer in the collimated leg
    % and leave the fold AOIs to the solve.  An explicit override of any of
    % these on the call line still wins (it is applied after).
    P.bench.POL_IN = 'source';  P.bench.D_RC_L2 = 125;
    P.oap.OAP1_AOI = 20;  P.oap.OAP2_AOI = 25;  P.oap.OAP1_SIDE = 1;  P.oap.OAP2_SIDE = -1;
    P = apply_overrides_(P, varargin{:});
end
if isempty(P.outdir), P.outdir = fullfile(exdir, 'runs', P.tag); end
if ~exist(P.outdir, 'dir'), mkdir(P.outdir); end

out = struct('P', P);
want = @(st) any(strcmp(P.stages, st));
t0 = tic;

if want('bench')
    % ---- the bench: tg96_run's, LO arm placed ---------------------------
    Pt = P;  Pt.stages = intersect({'bench','figs','clearance'}, P.stages, 'stable');
    tb = tic;
    o  = tg96_run(Pt);
    G  = o.bench.G;
    oldcd = cd(P.outdir);  cl = onCleanup(@() cd(oldcd));
    rep = fopen([P.tag '_cophi_report.txt'], 'w');  cl2 = onCleanup(@() fclose(rep));
    say = @(varargin) say_(rep, varargin{:});
    say('=== COPHI on the TG96 bench (%s optics), model %d, grid %d, tag "%s" ===\n', ...
        P.bench.optics, P.MODEL, P.NGRID, P.tag);
    say('tg96_run bench stages {%s}: %.0f s; decks %s_test.in (test arm) + %s_ref.in (the LO arm)\n\n', ...
        strjoin(Pt.stages, ','), toc(tb), P.tag, P.tag);
    Cx  = cophi_ctx(P, G, say);  reg = Cx.reg;
    out.bench = struct('G', G, 'reg', reg, 'tg96', rmfield(o, 'bench'));
    say('\nwall %.0f s\n', toc(t0));
    save([P.tag '_cophi.mat'], 'out', '-v7.3');
    clear cl cl2
end

if want('report')
    % re-print the battery table from a finished run's results (no engine)
    oldcd = cd(P.outdir);  cl = onCleanup(@() cd(oldcd));
    rep = fopen([P.tag '_cophi_report.txt'], 'a');  cl2 = onCleanup(@() fclose(rep));
    say = @(varargin) say_(rep, varargin{:});
    b = load([P.tag '_cophi_battery_results.mat']);
    say('\n=== COPHI battery table (%s rig), re-reported %s ===\n', P.bench.optics, char(datetime('now')));
    battery_table_(P, b.battery.deck, say);
    clear cl cl2
end

if want('loop')
    % ---- the loop stage: the hold servo on the H reading (cophi_ctx loop hook,
    % tg96_run hands it the set point's matrix); the record's loop knobs, the
    % camera's cadence, the drift / LO / fiber / delay rows, both estimators.
    tb = tic;
    Pt = P;  Pt.stages = {'bench', 'deck'};  Pt.reading_ctx = 'cophi_ctx';
    Pt.battery.deck = true;  Pt.battery.noise = false;  Pt.pzt.step_err = 0;
    Pt.battery.loop_ext = true;  Pt.battery.cap_ladder = P.battery.base_rms;  Pt.battery.recap_surf = [];
    o  = tg96_run(Pt);
    oldcd = cd(P.outdir);  cl = onCleanup(@() cd(oldcd));
    rep = fopen([P.tag '_cophi_report.txt'], 'a');  cl2 = onCleanup(@() fclose(rep));
    say = @(varargin) say_(rep, varargin{:});
    LP = o.deck.loop_ext;
    say('\n=== COPHI loop (%s rig, %s): %.1f min ===\n', P.bench.optics, ...
        iff_(isfield(P.bench,'polarizing') && ~P.bench.polarizing, 'the plain splitter', 'the record''s polarizing train'), toc(tb)/60);
    say('  ss / bias / sig_n: the record''s metric (rms over lit, piston included) | the same PISTON-FREE\n');
    say('  %-11s %-8s %-9s %8s %8s %8s | %8s %8s %8s %8s\n', 'camera', 'row', 'estimator', 'ss', 'bias', 'sig_n', 'ss_pf', 'bias_pf', 'sig_pf', 'cycle s');
    for r = 1:size(LP.rows, 1)
        x = LP.rows(r, :);
        say('  %-11s %-8s %-9s %8.2f %8.2f %8.2f | %8.2f %8.2f %8.2f %8.2f%s\n', x{1}, x{2}, x{3}, x{4}, x{5}, x{6}, x{9}, x{10}, x{11}, x{8}, iff_(x{7}, '  DIVERGED', ''));
    end
    out.loop = LP;
    save([P.tag '_cophi_loop_results.mat'], '-struct', 'out', 'loop');
    say('wall %.1f min\n', toc(t0)/60);
    clear cl cl2
end

if want('noise')
    % ---- the noise stage: the H reading on every named camera + the electronics
    % terms (campaign sec. 9.1), through the deck stage's own estimator (rows
    % re-made, ladders off; tg96_run hands its estimator to cophi_ctx's hook).
    % Run it on BOTH trains: the bench as built (the record's polarizing optics)
    % and 'bench.polarizing',false (Feng's plain splitter).
    tb = tic;
    Pt = P;  Pt.stages = {'bench', 'deck'};  Pt.reading_ctx = 'cophi_ctx';
    Pt.battery.deck = true;  Pt.battery.noise = false;  Pt.pzt.step_err = 0;
    Pt.battery.noise_ext = true;  Pt.battery.cap_ladder = P.battery.base_rms;  Pt.battery.recap_surf = [];
    o  = tg96_run(Pt);
    oldcd = cd(P.outdir);  cl = onCleanup(@() cd(oldcd));
    rep = fopen([P.tag '_cophi_report.txt'], 'a');  cl2 = onCleanup(@() fclose(rep));
    say = @(varargin) say_(rep, varargin{:});
    NZ = o.deck.noise_ext;
    trn = 'the record''s polarizing train (as built)';
    if isfield(P.bench, 'polarizing') && ~P.bench.polarizing, trn = 'the PLAIN splitter (no polarizing optics; Feng''s layout)'; end
    say('\n=== COPHI noise (%s rig, %s): %.1f min ===\n', P.bench.optics, trn, toc(tb)/60);
    say('  throughput source -> camera (flux 1 launched): test arm %.4f, LO arm %.4f\n', NZ.T_test, NZ.T_lo);
    say('  pm/read = ONE reading''s noise per actuator (the unpoked-lit floor; contrast-gated); pm@hold = the hold-out check (+-30 %% at %d draws)\n', NZ.nreal);
    say('  %-12s %-62s %9s %9s %9s %10s\n', 'camera', 'name', 'pm/read', 'pm@hold', 'reads/s', 's per pm');
    for c = 1:size(NZ.cams, 1)
        r = NZ.cams(c, :);
        say('  %-12s %-62s %9.3g %9.3g %9.3g %10.3g\n', r{1}, r{2}, r{9}, r{8}, r{10}, r{11});
    end
    say('  electronics, ONE reading''s error (pm at the hold-out / over lit):\n');
    for e = 1:size(NZ.elec, 1)
        say('    %-8s %-8g %9.3g %9.3g\n', NZ.elec{e,1}, NZ.elec{e,2}, NZ.elec{e,3}, NZ.elec{e,4});
    end
    out.noise = struct('noise', NZ, 'rows', {o.deck.rows});
    save([P.tag '_cophi_noise_results.mat'], '-struct', 'out', 'noise');
    say('wall %.1f min\n', toc(t0)/60);
    clear cl cl2
end

if want('battery')
    % ---- the battery: tg96_run's OWN placement + deck stages, on the H reading
    % (P.reading_ctx = cophi_ctx): the record's lit set, seeds (base 7, dev), the
    % 30 nm working surface, the 64-state matrix ON it, the estimator and
    % lambda_m -- one code path with the record's I column, only the reading
    % differs.  Also writes the capture-range ladders (the capture stage's
    % first half) since the deck stage computes them.
    tb = tic;
    Pt = P;  Pt.stages = {'bench', 'deck'};  Pt.reading_ctx = 'cophi_ctx';
    Pt.battery.deck = true;  Pt.battery.noise = false;  Pt.pzt.step_err = 0;
    o  = tg96_run(Pt);
    oldcd = cd(P.outdir);  cl = onCleanup(@() cd(oldcd));
    rep = fopen([P.tag '_cophi_report.txt'], 'a');  cl2 = onCleanup(@() fclose(rep));
    say = @(varargin) say_(rep, varargin{:});
    say('\n=== COPHI battery (%s rig): the H reading through the record''s deck stage (%.1f min) ===\n', ...
        P.bench.optics, toc(tb)/60);
    say('tail of record: %s; the rows are tg96_run''s (%s_report.txt), the H reading = dmg_cophi_frames/recon, N = %d, lo_ratio %g, noise-free\n', ...
        P.bench.tail_file, P.tag, P.cophi.N, P.cophi.lo_ratio);
    DK = o.deck;
    [I, recf] = battery_table_(P, DK, say);
    out.battery = struct('deck', DK, 'record_rows', {I}, 'record_file', {recf});
    save([P.tag '_cophi_battery_results.mat'], '-struct', 'out', 'battery');
    say('wall %.1f min\n', toc(t0)/60);
end
end

function [I, recf] = battery_table_(P, DK, say)
%BATTERY_TABLE_  the H rows beside the record's I columns.  TWO records, labelled:
%   redo96_<optics>ph = the SAME lit set (6948, the Sept-30 aperture cap), the same
%   hold-out actuator and grid, the same redo tail (md5) and seed 7 -- the SAME 30
%   nm surface, so the strict comparison; redo96_<optics> = the deck's headline
%   rows, run before the cap (7500 lit, 120 grid sites): the same seed over a
%   larger set, i.e. a different draw of the surface.
    base = fullfile(fileparts(fileparts(P.outdir)), '..', 'tg_psi_dm96_oap', 'runs');
    tg = {['redo96_' P.bench.optics 'ph'], ['redo96_' P.bench.optics]};
    recf = cellfun(@(t) fullfile(base, t, [t '_report.txt']), tg, 'UniformOutput', false);
    I = cellfun(@record_rows_, recf, 'UniformOutput', false);
    say('\n  rows on the %g nm working surface, matrix ON it (gain / err pm / floor pm / SNR); lit %d, hold-out (%d,%d):\n', ...
        DK.base_rms*1e6, DK.nlit, DK.ic(1), DK.ic(2));
    say('  %-18s | %-34s | %-34s | %-34s\n', 'row', 'H (COPHI, this run)', ...
        sprintf('I %s (same surface)', tg{1}), sprintf('I %s (deck, pre-cap)', tg{2}));
    for r = 1:size(DK.rows, 1)
        nmr = DK.rows{r,1};
        hs = sprintf('%7.4f %8.1f %6s %9s', DK.rows{r,2}, DK.rows{r,3}, f0_(DK.rows{r,4}), f2_(DK.rows{r,5}));
        c = cell(1, 2);
        for q = 1:2
            k = find(strcmp(I{q}(:,1), nmr), 1);
            if isempty(k), c{q} = '(not in the record)'; else, c{q} = I{q}{k,2}; end
        end
        say('  %-18s | %-34s | %-34s | %-34s\n', nmr, hs, c{1}, c{2});
    end
    say('  capture range to 10%%: grid %.0f nm, single site %.0f nm (the ladders are in %s_report.txt)\n', ...
        DK.r10_age*1e6, DK.r10_one*1e6, P.tag);
end

function I = record_rows_(f)
%RECORD_ROWS_  the record's deck rows (name, 'gain err flr SNR' text) from its report.
    I = cell(0, 2);
    if ~isfile(f), I = {'(record report not found)', f}; return; end
    t = splitlines(fileread(f));  on = false;
    for i = 1:numel(t)
        L = strtrim(t{i});
        if startsWith(L, 'row ') && contains(L, 'gain'), on = true; continue; end
        if on
            if isempty(L) || ~contains(L, '/'), break; end
            tk = regexp(L, '^(\S+)\s+(.*)$', 'tokens', 'once');
            I(end+1, :) = {tk{1}, tk{2}}; %#ok<AGROW>
        end
    end
end
function s = f0_(x), if isnan(x), s = '-'; else, s = sprintf('%.0f', x); end, end
function s = f2_(x), if isnan(x), s = '-'; else, s = sprintf('%.2f', x); end, end


% ---------------------------------------------------------------------
function s = iff_(c, a, b), if c, s = a; else, s = b; end, end
function say_(rep, varargin)
    fprintf(varargin{:});  fprintf(rep, varargin{:});
end
function P = apply_overrides_(P, varargin)
%APPLY_OVERRIDES_  a struct (replaces P) and/or dotted name/value pairs.
    k = 1;
    if k <= numel(varargin) && isstruct(varargin{k}), P = varargin{k}; k = k + 1; end
    while k < numel(varargin)
        key = varargin{k};  val = varargin{k+1};  k = k + 2;
        parts = strsplit(key, '.');
        P = setfield(P, parts{:}, val);   %#ok<SFLD>
    end
end
