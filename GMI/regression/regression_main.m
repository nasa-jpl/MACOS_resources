% regression_main.m -- top-level runner for the GMI regression suite.
%
% Adds the lib/ + tests/ dirs + the GMI mex location to the path,
% invokes the six tests, prints a summary table, exits non-zero if
% any test failed.  Invoked via:
%
%   ./run_regression.sh
%
% Reference .mat files in ./reference/ are committed snapshots of
% the expected output.  Regenerate after intentional behavior
% changes: ./run_regression.sh --bootstrap.

here = fileparts(mfilename('fullpath'));

% Path setup: tests/ + lib/ for the regression-internal code,
% parent GMI/ for call_GMI.m + the mex, and Rx/ on the cwd so
% SMACOS's OLD command can find the prescription file.
addpath(fullfile(here, 'tests'));
addpath(fullfile(here, 'lib'));
addpath(fileparts(here));        % parent GMI/ holds call_GMI.m + GMI.mexa64

% Hard absolute tolerance.  See lib/compare_within.m.
opts.tol = 1.0e-12;

% cd into Rx/ so SMACOS('OLD', 'Rx_e5hex1', ...) and
% SMACOS('OLD', 'optiixonaxisz1_v4_pmsm_met', ...) resolve.  The
% optiix Rx lives in the parent GMI/ dir; symlink or copy as needed.
%
% Quick belt-and-suspenders: copy/symlink the optiix Rx into Rx/ on
% first use.  (The e5hex1 Rx was already placed there by the harness
% scaffolding.)
optiix_local = fullfile(here, 'Rx', 'optiixonaxisz1_v4_pmsm_met.in');
optiix_canon = fullfile(fileparts(here), 'optiixonaxisz1_v4_pmsm_met.in');
if ~isfile(optiix_local) && isfile(optiix_canon)
    copyfile(optiix_canon, optiix_local);
end

cd(fullfile(here, 'Rx'));

% --- Run all tests ---
tests = {
    @test01_smoke_optiix
    @test02_nominal_repro_optiix
    @test03_zern_response_optiix
    @test04_smoke_e5hex1
    @test05_nominal_repro_e5hex1
    @test06_zern_response_e5hex1
};

n = length(tests);
results = cell(n, 1);
n_pass = 0;
n_fail = 0;

fprintf('\n=== GMI regression suite ===\n');
fprintf('tolerance: %.3e (absolute |a-b|)\n\n', opts.tol);

for k = 1:n
    fprintf('--- %d/%d  %s\n', k, n, func2str(tests{k}));
    try
        r = tests{k}(opts);
    catch ME
        r.name = func2str(tests{k});
        r.pass = false;
        r.msg = sprintf('[%s] FAIL  harness exception: %s', r.name, ME.message);
    end
    results{k} = r;
    fprintf('    %s\n', r.msg);
    if r.pass
        n_pass = n_pass + 1;
    else
        n_fail = n_fail + 1;
    end
    fprintf('\n');
end

fprintf('=== summary: %d passed, %d failed (of %d) ===\n', n_pass, n_fail, n);

% quit force (rather than exit) skips MATLAB's mex-cleanup
% finalization.  Needed for the ifx-built mex (which SIGSEGVs at
% exit teardown -- suspected Fortran-module finalizer on second mex
% unload).  Harmless for the gfortran build, which exits cleanly
% either way.
if n_fail > 0
    quit(1, 'force');
end
quit force;
