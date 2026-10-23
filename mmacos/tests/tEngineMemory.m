classdef tEngineMemory < matlab.unittest.TestCase
%TENGINEMEMORY  Releasing the engine's memory inside a live MATLAB session.
%   Luis Marchen (2026-08-19) asked for a way to "kill the session when the
%   calculation is done" -- i.e. give the engine's memory back WITHOUT
%   quitting MATLAB.  Measured on this tree (FFSegDemoAll at model 256,
%   resident set from /proc/self/statm):
%
%     * repeated load / trace / opd at a FIXED model size does not grow the
%       process (steady-state slope +0.001 MB per iteration over 40) -- the
%       engine's arrays are module allocatables, reused in place;
%     * macos.unload() (re-init at the minimum model size) returns ~554 MB
%       of a 256-model session, repeatably and without accumulation;
%     * `clear('mmacos')` does NOT free the engine: the module allocatables
%       are orphaned when the DSO goes, so four unload/reload cycles GREW
%       the process by ~720 MB each.  That is why macos.unload does not
%       clear the mex.
%
%   This class asserts the DETERMINISTIC consequences (engine still usable,
%   model-size footprint actually changed, bad sizes rejected before they
%   reach the engine).  It deliberately does NOT assert on RSS -- that is
%   allocator-dependent and would be a flaky gate; the numbers above are
%   documented in macos.unload's help and in the reply to Luis.
%
%   MODEL-SIZE TRANSITIONS: this class walks 128 -> 256 -> 128 on purpose,
%   so the runner gives it its OWN matlab -batch (PLAN.md §0).

    properties (Constant)
        RxName = 'FFSegDemoAll.in'
    end

    methods (Test)

        function test_bad_model_size_is_rejected_in_matlab(testCase)
        % param_mod.F answers an unsupported size with `stop`, which takes
        % the HOST PROCESS down -- fatal when the engine is a mex.  (That
        % is not hypothetical: macos.init(32) killed a matlab -batch run
        % while this was being measured.)  macos.init must screen first.
            testCase.verifyError(@() macos.init(32), 'macos:init:badModelSize');
            testCase.verifyError(@() macos.init(200), 'macos:init:badModelSize');
            for n = macos.model_sizes()
                testCase.verifyTrue(isscalar(n) && n > 0);
            end
            testCase.verifyEqual(macos.model_size_min(), ...
                                 min(macos.model_sizes()));
        end

        function test_unload_shrinks_the_engine_and_keeps_it_usable(testCase)
        % The observable proof that the model-sized arrays were actually
        % rebuilt at the smaller size is mGridMat, the largest grid a
        % surface may carry.  Then: re-init, re-load, re-trace, and the OPD
        % must come back bit-identical -- releasing memory must not cost
        % numerical state.
            rx = rx_fixture_path(testCase.RxName);

            m = macos.Session(256);
            big_grid = macos.grid_size_max();
            m.load_rx(rx);
            n = m.num_elt();
            m.trace(n - 1);
            W_before = macos.opd();
            testCase.assertGreaterThan(max(abs(W_before(:))), 0, ...
                'fixture must produce a non-trivial OPD');

            macos.unload();
            small_grid = macos.grid_size_max();
            testCase.verifyLessThan(small_grid, big_grid, ...
                ['unload must actually rebuild the engine smaller ' ...
                 '(mGridMat did not drop)']);
            testCase.verifyFalse(logical(macos.has_rx()), ...
                'unload drops the loaded prescription');

            m2 = macos.Session(256);
            testCase.verifyEqual(macos.grid_size_max(), big_grid);
            m2.load_rx(rx);
            m2.trace(n - 1);
            testCase.verifyEqual(macos.opd(), W_before, 'AbsTol', 0, ...
                'the engine must be bit-for-bit usable after an unload');
        end

        function test_session_unload_tracks_the_model_size(testCase)
            m = macos.Session(256);
            testCase.assertEqual(m.model_size, 256);
            m.unload();
            testCase.verifyEqual(m.model_size, macos.model_size_min());
            % still usable through the same handle
            m.load_rx(rx_fixture_path('e5hex1.in'));
            testCase.verifyTrue(logical(macos.has_rx()));
        end

        function test_repeated_unload_is_safe(testCase)
        % unload() on a session already at the minimum is a documented
        % no-op; it must not error, and the engine must stay usable.
            macos.init(macos.model_size_min());
            macos.unload();
            macos.unload();
            macos.load_rx(rx_fixture_path('e5hex1.in'));
            macos.trace(12);
            testCase.verifyGreaterThan(max(abs(macos.opd()), [], 'all'), 0);
        end
    end
end
