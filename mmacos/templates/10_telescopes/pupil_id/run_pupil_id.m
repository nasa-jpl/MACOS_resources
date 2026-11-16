% run_pupil_id.m -- example runner for the beyond-FEX pupil-ID driver.
%
% Runs pupil_id.m on one or both bundled test telescopes and emits a
% consolidated report + the per-case graphics.  This is the "just run it"
% entry point; pupil_id.m is the general-purpose driver (any Rx in, a
% revised Rx out) and pupil_find.m (design/src) is the reusable core
% finder (no sensitivity driver calls it yet -- the dw_d*_multi
% supervisors re-reference the XP per field via reset_xp/macos.fex; see
% the README's wiring-status note).
%
%   run_pupil_id                 % both cases (default)
%   run_pupil_id('tma_onaxis')   % on-axis Korsch TMA (fully-lit pupil)
%   run_pupil_id('sz_tma')       % tilted-Zernike M1 (zone map auto-skips)
%   run_pupil_id('both')
%
% Each case writes, beside its deck: <case>_xp.in (the revised Rx),
% pupil_id_<case>.mat, and pupil_id_{cloud,zernikes,walk}.png.  Headless:
%   matlab -batch "run_pupil_id('both'); exit(0)"

function T = run_pupil_id(which)
    arguments
        which (1,:) char {mustBeMember(which,{'tma_onaxis','sz_tma','both'})} = 'both'
    end
    here = fileparts(mfilename('fullpath'));  if isempty(here), here = pwd; end
    addpath(fullfile(here,'..','..','..','src'));           % mmacos/src (pupil_id on this path too)
    addpath(here);                                          % pupil_id.m

    % --- preflight: the engine MEX SEGFAULTS (crashes MATLAB, not an error)
    % on macos.init if it cannot find macos_param.txt, i.e. MACOS_HOME unset.
    % GUI MATLAB launched from Finder/Dock does NOT inherit a shell profile,
    % so MACOS_HOME is commonly empty in an interactive session even when it
    % is set in the terminal.  Check it HERE, before any engine call, and
    % self-set it when the param file sits at the standard engine path. ------
    check_macos_home_();

    % test cases: name -> deck path (relative to this template dir)
    cases = struct( ...
        'tma_onaxis', fullfile(here,'..','tma_onaxis','tma_onaxis.in'), ...
        'sz_tma',     fullfile(here,'..','sz_tma','sz_tma.in'));
    if strcmp(which,'both'), names = fieldnames(cases); else, names = {which}; end

    % Keep every artifact under THIS runner's dir (one subdir per case), so
    % nothing lands in the sibling deck dirs; the revised Rx also goes here
    % (out_rx), leaving the input decks untouched.  results/ is gitignored.
    resroot = fullfile(here,'results');
    R = struct('name',{},'out',{});
    for i = 1:numel(names)
        nm = names{i};  rx = cases.(nm);
        fprintf('\n########## pupil-ID case: %s ##########\n', nm);
        assert(isfile(rx), 'deck not found for case %s: %s', nm, rx);
        cdir = fullfile(resroot,nm);
        if ~exist(cdir,'dir'), mkdir(cdir); end
        out = pupil_id(rx, 'outdir',cdir, ...
                           'out_rx',fullfile(cdir,[nm '_xp.in']));
        R(end+1) = struct('name',nm, 'out',out);            %#ok<AGROW>
    end

    % ---- consolidated report across cases ----------------------------
    fprintf('\n=====================================================================\n');
    fprintf(' PUPIL-ID RUNNER SUMMARY (%d case%s)\n', numel(R), plural_(numel(R)));
    fprintf('=====================================================================\n');
    fprintf(' %-11s | %-9s | %-10s | %-10s | %-8s | %-7s | %s\n', ...
        'case','FEX rad','conv rad','dep RMS','XPS df%','XPS as%','revised Rx');
    fprintf(' %s\n', repmat('-',1,92));
    for i = 1:numel(R)
        o = R(i).out;
        [~,rxn,rxe] = fileparts(o.out_rx);
        fprintf(' %-11s | %9.5f | %10.5f | %8.2e | %7.2f | %7.2f | %s\n', ...
            R(i).name, o.fex.rad, o.quality.conv_radius, o.quality.dep_rms, ...
            o.xps.defocus_relpct, o.xps.astig_relpct, [rxn rxe]);
    end
    fprintf(' %s\n', repmat('-',1,92));
    fprintf(' (FEX rad = XP->detector propagation radius, written to the Rx; conv rad =\n');
    fprintf('  pupil-imaging convergence-surface curvature, a QUALITY diagnostic, NOT the Rx\n');
    fprintf('  radius. XPS df/as%% = pupil_map anchor=stop vs engine pupil_quality (%% rel).)\n');

    T = R;
end

function s = plural_(n), if n==1, s=''; else, s='s'; end, end

function check_macos_home_()
%CHECK_MACOS_HOME_  Fail LOUDLY (never let the MEX segfault) if the engine
%   cannot find macos_param.txt.  If MACOS_HOME is unset but the file is at
%   the standard engine source path, self-set it; otherwise error with the
%   fix.  macos.init reads macos_param.txt from MACOS_HOME; a missing file is
%   a clean STOP standalone but a SIGSEGV inside the MEX host.
    h = getenv('MACOS_HOME');
    if ~isempty(h) && isfile(fullfile(h,'macos_param.txt')), return; end
    guess = '/Users/dcr/dev/macos/macos_f90';               % this box's engine source
    if isfile(fullfile(guess,'macos_param.txt'))
        setenv('MACOS_HOME', guess);
        fprintf('[run_pupil_id] MACOS_HOME was unset; set to %s\n', guess);
        return;
    end
    error('run_pupil_id:noMacosHome', ['MACOS_HOME is not set (or has no ' ...
        'macos_param.txt), so the engine MEX would SEGFAULT on macos.init.\n' ...
        'Fix: in MATLAB run  setenv(''MACOS_HOME'', ''<path to macos/macos_f90>'')  ' ...
        'before this runner,\nor launch MATLAB from a shell where MACOS_HOME is ' ...
        'exported.  Current value: [%s]'], getenv('MACOS_HOME'));
end
