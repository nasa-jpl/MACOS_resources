classdef tFingerprint < matlab.unittest.TestCase
%TFINGERPRINT  jac_fingerprint round-trip + the committed e2e fingerprints.
%   Large derived products (s4_jacobians.mat, e2e_pie_met.mat) are
%   gitignored; a committed <file>.fp.json fingerprint is the reviewable
%   truth.  This validates the helper (build/write/read/check, both arms)
%   and, when a rebuilt .mat is present on disk, that it matches its
%   committed fingerprint (skip-with-message when the blob is absent).

    methods (TestClassSetup)
        function addpaths(tc) %#ok<MANU>
            here = fileparts(mfilename('fullpath'));   % .../mmacos/tests
            root = fileparts(here);                     % .../mmacos
            addpath(fullfile(root, 'design', 'src'));
        end
    end

    methods (Test)
        function test_roundtrip_match_and_detect(tc)
            % build -> write -> check: a matching product passes; a
            % perturbed one fails with an actionable report.
            S = struct('A', magic(8), 'B', reshape(1:60, 12, 5) * 0.1);
            meta = struct('product', 'unit', 'reset_xp', true);
            fp = [tempname '.fp.json'];
            c = onCleanup(@() delete(fp));
            jac_fingerprint('write', fp, S, meta);
            tc.verifyTrue(isfile(fp));

            [ok, rep] = jac_fingerprint('check', S, fp);
            tc.verifyTrue(ok, sprintf('identical product must match: %s', rep));

            S2 = S;  S2.A(:, 1) = S2.A(:, 1) * 1.5;   % perturb one column
            [ok2, rep2] = jac_fingerprint('check', S2, fp);
            tc.verifyFalse(ok2, 'a perturbed column must be detected');
            tc.verifyNotEmpty(rep2);

            S3 = rmfield(S, 'B');                      % missing field
            [ok3, ~] = jac_fingerprint('check', S3, fp);
            tc.verifyFalse(ok3, 'a missing field must be detected');
        end

        function test_meta_and_dims_preserved(tc)
            S = struct('X', ones(4, 3));
            meta = struct('product', 'unit', 'note', 'hi', 'n', 7);
            fp = [tempname '.fp.json'];
            c = onCleanup(@() delete(fp));
            jac_fingerprint('write', fp, S, meta);
            r = jac_fingerprint('read', fp);
            tc.verifyEqual(r.meta.product, 'unit');
            tc.verifyEqual(r.meta.n, 7);
            tc.verifyEqual(r.fields(1).size(:).', [4 3]);
        end

        function test_committed_e2e_fingerprints_present(tc)
            % the committed fingerprints must exist and be readable, and
            % (when the gitignored .mat is present on disk) must match it.
            here = fileparts(mfilename('fullpath'));   % .../mmacos/tests
            e2e  = fullfile(fileparts(here), 'templates', ...
                            '80_end_to_end', 'e2e');
            cases = { ...
                's4_jacobians', {'ox','oz','og'}, ...
                    {'dwdxall','dwdzall','dwdgall'}; ...
                'e2e_pie_met',  {},                {}};
            for i = 1:size(cases, 1)
                base = cases{i, 1};
                fpp  = fullfile(e2e, [base '.fp.json']);
                tc.verifyTrue(isfile(fpp), ...
                    sprintf('committed fingerprint %s.fp.json must exist', base));
                fp = jac_fingerprint('read', fpp);
                tc.verifyTrue(isfield(fp, 'meta') && isfield(fp, 'fields'));
                matp = fullfile(e2e, [base '.mat']);
                tc.assumeTrue(isfile(matp), ...
                    sprintf('%s.mat absent (gitignored) -- regenerate to check', base));
                M = load(matp);
                S = build_check_struct_(base, M);
                [ok, rep] = jac_fingerprint('check', S, fpp);
                tc.verifyTrue(ok, sprintf('%s.mat vs fingerprint: %s', base, rep));
            end
        end
    end
end


function S = build_check_struct_(base, M)
% Reproduce the field set each regen script fingerprints.
switch base
    case 's4_jacobians'
        S = struct('dwdxall', M.ox.dwdxall, 'dwdzall', M.oz.dwdxall, ...
                   'dwdgall', M.og.dwdxall);
    case 'e2e_pie_met'
        S = struct('dedx', M.dedx, 'dldx', M.dldx, 'dldx_opt', M.dldx_opt, ...
                   'dxde', M.dxde, 'dxdl', M.dxdl);
    otherwise
        S = struct();
end
end
