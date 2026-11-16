classdef tApiCoverage < matlab.unittest.TestCase
%TAPICOVERAGE  Every macos_api_mod routine has an mmacos binding (BRIEF_to_bindings A.4;
%   twin of pymacos tests/test_api_coverage.py).
%   The routine list is read from the Fortran SOURCE (macos_api_mod.F90, $MACOS_HOME or
%   ~/dev/macos/macos_f90) with its own regex -- not from gen_mex_wrappers.py -- so a
%   routine added to the api without re-running the generator and relinking the mex
%   turns this red.  Each routine maps to a mex command: its own name (the generated
%   gen_dispatch), or a hand-written alias (HAND below; checked against mmacos_mex.F's
%   CASE list).  A generated command is probed in the BUILT mex with 40 dummy arguments
%   (no api routine takes 40 inputs): its helper refuses on nrhs before doing anything,
%   so the probe has no side effects and fails only with 'unknown command' when the
%   binary lacks it.
%   Must-fail: test_gate_names_a_removed_binding (in-suite, a doctored command set) and,
%   recorded once, a side mex built without one helper.

    properties (Constant)
        % api routine -> hand-written mex command (mmacos_mex.F)
        HAND = struct( ...
            'init', 'init', 'load_rx', 'load_rx', 'save_rx', 'save_rx', ...
            'modified_rx', 'modified_rx', 'n_elt', 'n_elt', 'opd_val', 'opd', ...
            'int_cmd', 'intensity', 'int_get', 'intensity', ...
            'cfield_cmd', 'complex_field', 'cfield_get', 'complex_field', ...
            'cfield_plane_get', 'complex_field', 'cfield_apodize', 'apodize', ...
            'elt_dx_get', 'dx_at', 'base_unit_to_metres', 'base_unit_to_metres', ...
            'trace_rays', 'trace_rays', 'prb_elt', 'prb_elt', ...
            'rayfield_get', 'ray_field', 'elt_csys_get', 'elt_csys_get')
        HELPERS = {'translatesurfaceid', 'translateeltid'}
    end

    methods (Test)
        function test_api_source_lists_routines(tc)
            r = tApiCoverage.api_routines();
            tc.verifyGreaterThanOrEqual(numel(r), 155);
            tc.verifyTrue(ismember('ray_status_get', r) && ismember('load_rx', r));
        end

        function test_every_api_routine_is_bound(tc)
            missing = tApiCoverage.unbound(tApiCoverage.api_routines(), ...
                tApiCoverage.mex_has());
            tc.verifyEmpty(missing, ['api routines with no mmacos binding: ' ...
                strjoin(missing, ' ')]);
        end

        function test_gate_names_a_removed_binding(tc)
            r = tApiCoverage.api_routines();
            has = @(c) ~strcmp(c, 'ray_status_get') && tApiCoverage.mex_has_one(c);
            tc.verifyEqual(tApiCoverage.unbound(r, has), {'ray_status_get'});
        end
    end

    methods (Static)
        function r = api_routines()
            home = getenv('MACOS_HOME');
            if isempty(home)
                home = fullfile(getenv('HOME'), 'dev', 'macos', 'macos_f90');
            end
            p = fullfile(home, 'macos_api_mod.F90');
            % fail closed: no source, no gate
            assert(isfile(p), 'macos_api_mod.F90 not found at %s (set MACOS_HOME)', p);
            t = regexp(fileread(p), '(?m)^\s+subroutine\s+(\w+)\s*\(', 'tokens', ...
                'ignorecase');
            % keep the source's case: a mex command name is case-sensitive
            % (elt_srf_zrn_FreeForm)
            r = cellfun(@(c) c{1}, t, 'UniformOutput', false);
            r = r(~ismember(lower(r), tApiCoverage.HELPERS));
        end

        function missing = unbound(routines, has)
            missing = {};
            for i = 1:numel(routines)
                if ~has(tApiCoverage.cmd_for(routines{i}))
                    missing{end+1} = routines{i}; %#ok<AGROW>
                end
            end
        end

        function c = cmd_for(routine)
            if isfield(tApiCoverage.HAND, routine)
                c = tApiCoverage.HAND.(routine);
            else
                c = routine;
            end
        end

        function f = mex_has()
            f = @tApiCoverage.mex_has_one;
        end

        function ok = mex_has_one(cmd)
            hand = struct2cell(tApiCoverage.HAND);
            if ismember(cmd, hand)
                src = fileread(fullfile(fileparts(which('mmacos')), 'mmacos_mex.F'));
                ok = contains(src, sprintf('CASE (''%s'')', cmd));
                return
            end
            z = num2cell(zeros(1, 40));
            try
                mmacos(cmd, z{:});
                ok = false;   % unreachable: every generated helper checks nrhs
            catch e
                % the generated helper's nrhs refusal = the command is bound
                ok = contains(e.message, 'expects') && ...
                    ~contains(e.message, 'unknown command');
            end
        end
    end
end
