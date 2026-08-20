% run_pupil_id.m -- example runner for the beyond-FEX pupil-ID driver.
%
% Runs pupil_id.m on one or both bundled test telescopes and emits a
% consolidated report + the per-case graphics.  This is the "just run it"
% entry point; pupil_id.m is the general-purpose driver (any Rx in, a
% revised Rx out) and pupil_find.m (design/src) is the run_dwd*-callable
% core finder.
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
