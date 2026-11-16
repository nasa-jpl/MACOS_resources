classdef tBudgetKernel < matlab.unittest.TestCase
    % Error-budget kernel (macos.budget + run_budget), pure linear algebra
    % -- NO engine, so it is fast and robust.  Reproduces Redding 2014's
    % headline result: a properly-formed RSS budget is a covariance
    % analysis, and Monte-Carlo gives the identical answer (Fig. 6).
    %
    % Checks:
    %   (1) covariance == Monte-Carlo, per group and total; RSS identity;
    %       WFE = sigma*alpha  (the Fig.6 gate)
    %   (2) control projection reduces WFE and cov == MC under control
    %   (3) cov_budget on one full-covariance group reproduces run_met.m's
    %       rmsw(X) = sqrt(trace(X*G)/nw)  (refactor-safety of the kernel)
    %   (4) uniform & worstcase distributions: cov == MC
    %   (5) unit reconciliation through assemble + spec_from_table
    %       (nrad / nm with cbm); orphan/overlap column warnings
    %   (6) run_budget end-to-end on a synthetic struct: artifacts written

    methods (TestClassSetup)
        function paths(~)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(fileparts(here));
            addpath(fullfile(root, 'mmacos', 'src'));
            addpath(fullfile(root, 'mmacos', 'design', 'runners'));
        end
    end

    methods (Static)
        function [M, spec] = synth(seed, widths, sig)
            rng(seed);
            nz = sum(widths); Nw = 500;
            Jk = randn(Nw, nz);
            G = Jk' * Jk;
            M = struct('G', G, 'nw', Nw, 'Jk', Jk, 'cbm', 1, ...
                'meta', struct('channel', {repmat({'dwdx'}, nz, 1)}));
            c0 = 0; groups = struct('name', {}, 'cols', {}, 'cm', {});
            for k = 1:numel(widths)
                cols = false(nz, 1); cols(c0+(1:widths(k))) = true; c0 = c0 + widths(k);
                s = sig(k); nc = widths(k);
                cm = struct('type', 'iid', 'var', s^2, 'sigma_native', s, ...
                    'class', 'length_si', 'Sigma', [], ...
                    'sampler', @() s .* randn(nc, 1));
                groups(k) = struct('name', sprintf('g%d', k), 'cols', cols, 'cm', cm);
            end
            spec = struct('groups', groups);
        end

        function [M, dspec, H, R] = met_synth(seed, seen)
            % 12 drift states in two iid groups (1..6 at 1 um, 7..12 at
            % 2 um), 10 gauges that see the states SEEN, gauge noise 0.2 um.
            rng(seed); Nw = 300; nz = 12; Jk = randn(Nw, nz);
            M = struct('G', Jk'*Jk, 'nw', Nw, 'Jk', Jk, 'cbm', 1);
            mk = @(nc, s) struct('type', 'iid', 'var', s^2, 'sigma_native', s, ...
                'class', 'length_si', 'Sigma', [], 'sampler', @() s .* randn(nc, 1));
            c1 = false(nz, 1); c1(1:6) = true;  c2 = ~c1;
            dspec = struct('groups', struct('name', {'d1', 'd2'}, ...
                'cols', {c1, c2}, 'cm', {mk(6, 1e-6), mk(6, 2e-6)}));
            H = zeros(10, nz);  H(:, seen) = randn(10, numel(seen));
            R = (2e-7)^2 * eye(10);
        end

        function [M, spec, met, ctrl] = two_stage_model()
            % the A.3 two-stage synthetic (mm deck, both stages, sections)
            rng(15); Nw = 400;
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'}; cn = {}; iE = []; df = [];
            for e = [3 4], for d = 1:6
                cn{end+1} = sprintf('Elt %d %s', e, dofs{d}); iE(end+1)=e; df(end+1)=d-1; %#ok<AGROW>
            end, end
            ox = struct('dwdxall', randn(Nw,12), 'channel_names', {cn}, ...
                'iElt', iE, 'dof_idx', df, 'kind', {repmat({'RigidBody'},12,1)}, ...
                'cbm', 1e-3, 'base_units', 'mm', 'trans_output', 'si');
            M = macos.budget.assemble(ox);
            vn = {'Name','Elements','DOFs','Modes','Param','Channel','Dist', ...
                'Sigma','Units','Min','Max','Cov','Control','Section'};
            T = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid','WFC','Primary'; ...
                'E3 Despace','3','despace','','','dwdx','uniform',NaN,'nm',-200,200,'iid','','Primary'; ...
                'E4 Decenter','4','decenter','','','dwdx','normal',100,'nm',NaN,NaN,'iid','WFC','Secondary'}, ...
                'VariableNames', vn);
            spec = macos.budget.spec_from_table(T, M, 'wfs_sigma', {3, 'nm'}, ...
                'act_sigma', {'E3 Tilt', 0.5, 'nrad'; 'E4 Decenter', 1, 'nm'});
            D = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',5,'nrad',NaN,NaN,'iid','','Primary'; ...
                'E4 Decenter','4','decenter','','','dwdx','normal',20,'nm',NaN,NaN,'iid','','Secondary'; ...
                'E4 Tilt','4','tilt','','','dwdx','normal',10,'nrad',NaN,NaN,'iid','',''}, ...
                'VariableNames', vn);
            dspec = macos.budget.spec_from_table(D, M);
            H = zeros(6, 12);  H(:, 10:11) = randn(6, 2);
            met = struct('drift', dspec, 'H', H, 'R', (1e-10)^2 * eye(6), ...
                'gauge_units', 'nm', 'gauge_per_eng', 1e-9);
            ctrl = spec.groups(1).cols | spec.groups(3).cols;
        end

        function x = sheet_xml(f, k)
            d = tempname; mkdir(d); c = onCleanup(@() rmdir(d, 's'));
            unzip(f, d);
            x = fileread(fullfile(d, 'xl', 'worksheets', sprintf('sheet%d.xml', k)));
        end

        function V = recalc(f, wd)
            % headless LibreOffice: load (recalculating) and re-save; read
            % the recalculated values of sheet WFE back at full precision
            od = fullfile(wd, 'recalc');  if ~isfolder(od), mkdir(od); end
            prof = ['file://' fullfile(wd, 'lo_profile')];
            % env -u LD_LIBRARY_PATH: MATLAB's own libstdc++ (on its library
            % path) is older than LibreOffice's, and soffice dies at launch
            cmd = sprintf(['env -u LD_LIBRARY_PATH soffice -env:UserInstallation=%s --headless ' ...
                '--convert-to xlsx --outdir "%s" "%s" >/dev/null 2>&1'], prof, od, f);
            st = system(cmd);
            [~, b, e] = fileparts(f);
            out = fullfile(od, [b e]);
            assert(st == 0 && isfile(out), 'soffice conversion failed');
            V = readcell(out, 'Sheet', 'WFE');
            delete(out);
        end

        function S = shown(f, wd, k)
            % sheet K exported to CSV with cell contents AS SHOWN (the
            % number formats applied), every field read back as text
            od = fullfile(wd, sprintf('shown%d', k));  if ~isfolder(od), mkdir(od); end
            prof = ['file://' fullfile(wd, 'lo_profile')];
            flt = sprintf('csv:Text - txt - csv (StarCalc):44,34,76,1,,0,false,true,true,false,false,%d', k);
            cmd = sprintf(['env -u LD_LIBRARY_PATH soffice -env:UserInstallation=%s ' ...
                '--headless --convert-to ''%s'' --outdir "%s" "%s" >/dev/null 2>&1'], ...
                prof, flt, od, f);
            st = system(cmd);
            [~, b] = fileparts(f);
            c = dir(fullfile(od, [b '*.csv']));
            assert(st == 0 && ~isempty(c), 'soffice csv export failed');
            out = fullfile(od, c(1).name);
            ln = readlines(out);  delete(out);
            S = repmat({''}, numel(ln), 30);
            for i = 1:numel(ln)
                t = regexp(char(ln(i)), ',(?=(?:[^"]*"[^"]*")*[^"]*$)', 'split');
                v = cellfun(@(q) strrep(regexprep(q, '^"(.*)"$', '$1'), '""', '"'), ...
                    t, 'UniformOutput', false);
                S(i, 1:numel(v)) = v;
            end
        end

        function set_input(f, wd, ref, val, k)
            % rewrite one numeric cell's value inside the .xlsx (as an
            % engineer typing a new sigma would); K = sheet number (1 = WFE)
            if nargin < 5, k = 1; end
            d = fullfile(wd, 'edit');  if isfolder(d), rmdir(d, 's'); end
            mkdir(d);  unzip(f, d);
            sx = fullfile(d, 'xl', 'worksheets', sprintf('sheet%d.xml', k));
            x = fileread(sx);
            pat = sprintf('(<c r="%s"[^>]*><v>)[^<]*(</v>)', ref);
            assert(~isempty(regexp(x, pat, 'once')), 'input cell %s not found', ref);
            x = regexprep(x, pat, sprintf('$1%.17g$2', val), 'once');
            fid = fopen(sx, 'w');  fwrite(fid, x);  fclose(fid);
            files = {'[Content_Types].xml', '_rels/.rels', 'xl/workbook.xml', ...
                'xl/_rels/workbook.xml.rels', 'xl/styles.xml'};
            w = dir(fullfile(d, 'xl', 'worksheets', '*.xml'));
            files = [files, strcat('xl/worksheets/', {w.name})];
            z = [tempname '.zip'];  zip(z, files, d);  movefile(z, f, 'f');
        end
    end

    methods (Test)
        function test_cov_equals_mc(tc)
            [M, spec] = tBudgetKernel.synth(7, [6 10 4], [5e-8 2e-9 1e-3]);
            rc = macos.budget.cov_budget(M, spec);
            rm = macos.budget.mc_budget(M, spec, 'ndraw', 20000, 'seed', 3);
            tc.verifyEqual(rc.wfe, rm.wfe, 'RelTol', 0.05);
            tc.verifyEqual(rc.total, rm.total, 'RelTol', 0.05);
            tc.verifyEqual(rc.total^2, sum(rc.wfe.^2), 'RelTol', 1e-12);  % RSS
            tc.verifyEqual(rc.wfe, rc.sigma(:) .* rc.alpha(:), 'RelTol', 1e-12);
            tc.verifyFalse(rc.controlled);
        end

        function test_control_reduces_and_matches_mc(tc)
            rng(9); Nw = 300; nz = 14; Jk = randn(Nw, nz);
            M = struct('G', Jk'*Jk, 'nw', Nw, 'Jk', Jk, 'cbm', 1, ...
                'meta', struct('channel', {repmat({'dwdx'}, nz, 1)}));
            cols = false(nz, 1); cols(1:8) = true; nc = 8; s = 1e-6;
            cm = struct('type', 'iid', 'var', s^2, 'sigma_native', s, ...
                'class', 'length_si', 'Sigma', [], 'sampler', @() s .* randn(nc, 1));
            spec = struct('groups', struct('name', 'err', 'cols', cols, 'cm', cm));
            ctrl = false(nz, 1); ctrl(9:14) = true;
            rc0 = macos.budget.cov_budget(M, spec);
            rc1 = macos.budget.cov_budget(M, spec, 'control', ctrl);
            rm1 = macos.budget.mc_budget(M, spec, 'ndraw', 8000, 'seed', 1, 'control', ctrl);
            tc.verifyLessThan(rc1.total, rc0.total);
            tc.verifyEqual(rc1.total, rm1.total, 'RelTol', 0.08);
            tc.verifyTrue(rc1.controlled);
        end

        function test_control_projection_identities(tc)
            % Pu = I - Ju*pinv(Ju) (Eq. 23) is a projector (Pu*Pu = Pu)
            % that annihilates the control span (Pu*Ju = 0), and the
            % projected Gram is (Pu*Jk)'(Pu*Jk) -- all at Jk's size.
            rng(12); Nw = 350; nz = 16; Jk = randn(Nw, nz);
            M = struct('G', Jk'*Jk, 'nw', Nw, 'Jk', Jk, 'cbm', 1);
            ctrl = false(nz, 1); ctrl([2 5 9 10 14]) = true;
            [Gp, ctl] = macos.budget.project_control(M, ctrl);
            PJ = ctl.apply(Jk);
            tc.verifyLessThan(norm(ctl.apply(ctl.Ju)) / norm(ctl.Ju), 1e-13);
            tc.verifyLessThan(norm(ctl.apply(PJ) - PJ) / norm(PJ), 1e-13);
            tc.verifyEqual(Gp, PJ'*PJ, 'RelTol', 1e-12);
            tc.verifyLessThan(max(max(abs(Gp(ctrl, :)))) / max(abs(Gp(:))), 1e-13);
            tc.verifyEqual(ctl.rank, 5);
            % the one-output call is the v1 Gram, bit for bit
            tc.verifyEqual(macos.budget.project_control(M, ctrl), Gp);
        end

        function test_wfc_noise_rows_cov_equals_mc(tc)
            % Eq. 22 with the controller in the loop: per error group
            % (post-WFC, Eq. 30), the sensing row (Eq. 32: alpha^2 =
            % rank(Ju)/nw), one actuator row per control group (Eq. 31,
            % unprojected), and the total -- cov == MC on every row.
            rng(13); Nw = 300; nz = 14; Jk = randn(Nw, nz);
            M = struct('G', Jk'*Jk, 'nw', Nw, 'Jk', Jk, 'cbm', 1);
            mk = @(idx, s) struct('type', 'iid', 'var', s^2, 'sigma_native', s, ...
                'class', 'length_si', 'Sigma', [], 'sampler', @() s .* randn(numel(idx), 1));
            cl = @(idx) ismember((1:nz)', idx);
            gi = {1:6, 7:10, 11:14};  sg = [2e-6 1e-6 3e-6];
            groups = struct('name', {}, 'cols', {}, 'cm', {});
            for k = 1:3
                groups(k) = struct('name', sprintf('g%d', k), 'cols', cl(gi{k}), ...
                    'cm', mk(gi{k}, sg(k)));
            end
            act = struct('name', {'g2 Actuator', 'g3 Actuator'}, ...
                'group', {'g2', 'g3'}, 'cols', {cl(7:10), cl(11:14)}, ...
                'cm', {mk(7:10, 4e-7), mk(11:14, 9e-7)});
            spec = struct('groups', groups);
            spec.wfs = struct('sigma', 5e-6, 'sigma_eng', 5e-6, 'units', 'm');
            spec.act = act;
            ctrl = cl(7:14);
            rc = macos.budget.cov_budget(M, spec, 'control', ctrl);
            rm = macos.budget.mc_budget(M, spec, 'ndraw', 20000, 'seed', 5, ...
                'control', ctrl);
            tc.verifyEqual({rc.extra.kind}, {'wfs', 'act', 'act'});
            tc.verifyEqual(rc.extra(1).alpha^2, 8/Nw, 'RelTol', 1e-12);
            tc.verifyEqual(rc.extra(2).alpha, rc.alpha_pre(2), 'RelTol', 1e-12);
            tc.verifyLessThan(rc.alpha(2:3), 1e-12 * rc.alpha_pre(2:3));  % actuated: null
            tc.verifyEqual(rm.wfe(1), rc.wfe(1), 'RelTol', 0.05);
            tc.verifyLessThan(rm.wfe(2:3), 1e-12 * rm.wfe_pre(2:3));  % MC sees the null too
            tc.verifyEqual(rc.wfe_pre, rm.wfe_pre, 'RelTol', 0.05);
            tc.verifyEqual([rc.extra.wfe]', [rm.extra.wfe]', 'RelTol', 0.05);
            tc.verifyEqual(rc.total, rm.total, 'RelTol', 0.05);
            tc.verifyEqual(rc.total^2, sum(rc.var) + sum([rc.extra.var]), 'RelTol', 1e-12);
            % no noise inputs -> no rows; uncontrolled call -> v1 fields
            rc0 = macos.budget.cov_budget(M, struct('groups', groups));
            tc.verifyEmpty(rc0.extra);
            tc.verifyEqual(rc0.alpha, rc0.alpha_pre);
            tc.verifyEqual(rc0.total, rc0.total_pre);
        end

        function test_met_stage_reduces_to_run_met(tc)
            % project_met with EVERY state actuated is run_met.m's merit,
            % sqrt(trace(post*G)/nw), post = X - X*H'*((H*X*H'+R)\(H*X)),
            % and with no actuators the estimate corrects nothing.
            [M, dspec, H, R] = tBudgetKernel.met_synth(31, 1:12);
            X = blkdiag(1e-12 * eye(6), 4e-12 * eye(6));
            post = X - X*H'*((H*X*H' + R) \ (H*X));
            rmsw = sqrt(trace(post * M.G) / M.nw);
            rt = macos.budget.project_met(M, dspec, H, R, 'control', true(12, 1));
            tc.verifyEqual(rt.total_post, rmsw, 'RelTol', 1e-10);
            tc.verifyEqual(rt.total_pre, sqrt(trace(X * M.G) / M.nw), 'RelTol', 1e-12);
            tc.verifyEqual(rt.post, post, 'RelTol', 1e-12);
            r0 = macos.budget.project_met(M, dspec, H, R);
            tc.verifyEqual(r0.wfe_post, r0.wfe_pre, 'RelTol', 1e-12);
            tc.verifyEqual(r0.extra(1).wfe, 0);
        end

        function test_met_stage_cov_equals_mc_and_unseen_passes(tc)
            % Partial control (11 of 12 states actuated), the estimator in
            % the loop: cov == MC per drift group, gauge row and total; a
            % drift the metrology cannot see (group 2: H's columns 7..12
            % are zero) passes through UNCORRECTED (post == pre, not the
            % WFC stage's post value); remove the gauges and the seen
            % group's post rises to its pre value (the must-fail leg).
            ctrl = false(12, 1); ctrl([1:5 7:12]) = true;
            [M, dspec, H, R] = tBudgetKernel.met_synth(32, 1:6);
            rt = macos.budget.project_met(M, dspec, H, R, 'control', ctrl);
            rm = macos.budget.mc_met(M, dspec, H, R, 'control', ctrl, ...
                'ndraw', 20000, 'seed', 4);
            tc.verifyEqual(rm.wfe_pre, rt.wfe_pre, 'RelTol', 0.05);
            tc.verifyEqual(rm.wfe_post, rt.wfe_post, 'RelTol', 0.05);
            tc.verifyEqual(rm.extra(1).wfe, rt.extra(1).wfe, 'RelTol', 0.05);
            tc.verifyEqual(rm.total_post, rt.total_post, 'RelTol', 0.05);
            % unseen: post == pre exactly -- although the WFC stage nulls
            % the same group (all of 7..12 actuated): between WFC updates
            % only the metrology drives the controller
            tc.verifyEqual(rt.alpha_met_post(2), rt.alpha_met(2), 'RelTol', 1e-12);
            rc = macos.budget.cov_budget(M, dspec, 'control', ctrl);
            tc.verifyLessThan(rc.alpha(2), 1e-12 * rt.alpha_met_post(2));
            % seen: metrology helps
            tc.verifyLessThan(rt.wfe_post(1), 0.5 * rt.wfe_pre(1));
            % must-fail: no gauges -> the seen group's post rises to pre
            r0 = macos.budget.project_met(M, dspec, 0 * H, R, 'control', ctrl);
            tc.verifyEqual(r0.wfe_post(1), r0.wfe_pre(1), 'RelTol', 1e-12);
            tc.verifyGreaterThan(r0.wfe_post(1), 2 * rt.wfe_post(1));
        end

        function test_coef_and_rollup_carry_the_habex_arithmetic(tc)
            % Both stages on an mm-deck model: every coefficient slot is a
            % number; per line sigma*coef reproduces each WFE column (the
            % workbook's E = C*D, F = C*E, I = G*G', J = G*H'); line Total =
            % sqrt(PostWFC^2 + PostMET^2); coef.total = the two stages
            % RSS'd; rollup's summary rows follow (sections, Everything
            % Else, TOTAL, Payload with the design residual, Margin).
            rng(15); Nw = 400;
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'}; cn = {}; iE = []; df = [];
            for e = [3 4], for d = 1:6
                cn{end+1} = sprintf('Elt %d %s', e, dofs{d}); iE(end+1)=e; df(end+1)=d-1; %#ok<AGROW>
            end, end
            ox = struct('dwdxall', randn(Nw,12), 'channel_names', {cn}, ...
                'iElt', iE, 'dof_idx', df, 'kind', {repmat({'RigidBody'},12,1)}, ...
                'cbm', 1e-3, 'base_units', 'mm', 'trans_output', 'si');
            M = macos.budget.assemble(ox);
            vn = {'Name','Elements','DOFs','Modes','Param','Channel','Dist', ...
                'Sigma','Units','Min','Max','Cov','Control','Section'};
            T = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid','WFC','Primary'; ...
                'E3 Despace','3','despace','','','dwdx','uniform',NaN,'nm',-200,200,'iid','','Primary'; ...
                'E4 Decenter','4','decenter','','','dwdx','normal',100,'nm',NaN,NaN,'iid','WFC','Secondary'}, ...
                'VariableNames', vn);
            spec = macos.budget.spec_from_table(T, M, 'wfs_sigma', {3, 'nm'}, ...
                'act_sigma', {'E3 Tilt', 0.5, 'nrad'; 'E4 Decenter', 1, 'nm'});
            D = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',5,'nrad',NaN,NaN,'iid','','Primary'; ...
                'E4 Decenter','4','decenter','','','dwdx','normal',20,'nm',NaN,NaN,'iid','','Secondary'; ...
                'E4 Tilt','4','tilt','','','dwdx','normal',10,'nrad',NaN,NaN,'iid','',''}, ...
                'VariableNames', vn);
            dspec = macos.budget.spec_from_table(D, M);
            H = zeros(6, 12);  H(:, 10:11) = randn(6, 2);     % gauges see E4 Tx,Ty
            met = struct('drift', dspec, 'H', H, 'R', (1e-10)^2 * eye(6), ...
                'gauge_units', 'nm', 'gauge_per_eng', 1e-9);
            ctrl = spec.groups(1).cols | spec.groups(3).cols;
            rc = macos.budget.cov_budget(M, spec, 'control', ctrl, 'met', met);
            coef = macos.budget.to_coef(rc, M, spec);
            keys = coef.meta.order;
            kinds = cellfun(@(k) coef.(k).kind, keys, 'UniformOutput', false);
            tc.verifyEqual(sort(unique(kinds)), sort({'act'; 'drift'; 'error'; 'met'; 'wfs'}));
            for i = 1:numel(keys)
                c = coef.(keys{i});
                v = [c.alpha c.alpha_post c.alpha_met c.alpha_met_post ...
                    c.coef_pre c.coef_post c.coef_met c.coef_met_post];
                tc.verifyTrue(all(isfinite(v)), c.name);
                tc.verifyEqual(c.sigma_eng * c.coef_pre, c.wfe_pre, 'AbsTol', 1e-12 * (1 + c.wfe_pre), c.name);
                tc.verifyEqual(c.sigma_eng * c.coef_post, c.wfe, 'AbsTol', 1e-12 * (1 + c.wfe), c.name);
                tc.verifyEqual(c.sigma_drift * c.coef_met_post, c.wfe_met_post, ...
                    'AbsTol', 1e-12 * (1 + c.wfe_met_post), c.name);
                if strcmp(c.kind, 'error') || strcmp(c.kind, 'drift')
                    tc.verifyEqual(c.sigma_drift * c.coef_met, c.wfe_met, ...
                        'AbsTol', 1e-12 * (1 + c.wfe_met), c.name);
                end
                tc.verifyEqual(c.total, hypot(c.wfe, c.wfe_met_post), 'RelTol', 1e-12);
            end
            tonm = M.cbm * 1e9;
            tc.verifyEqual(coef.total, rc.total_all * tonm, 'RelTol', 1e-12);
            tc.verifyEqual(coef.total, hypot(coef.total_post_wfc, coef.total_post_met), 'RelTol', 1e-12);
            % engineering units: 50 nrad / 100 nm in, uniform 200 nm -> 115.47 nm 1-sigma
            tc.verifyEqual(coef.E3_Tilt.sigma_eng, 50, 'RelTol', 1e-12);
            tc.verifyEqual(coef.E3_Despace.sigma_eng, 200 / sqrt(3), 'RelTol', 1e-12);
            tc.verifyEqual(coef.E3_Tilt.sigma_drift, 5, 'RelTol', 1e-12);
            tc.verifyEqual(coef.E3_Tilt_Actuator.sigma_drift, 0.5, 'RelTol', 1e-12); % both stages
            tc.verifyEqual(coef.Metrology_Gauge_Noise.sigma_drift, 0.1, 'RelTol', 1e-12);
            % roll-up summary rows
            R = macos.budget.rollup(rc, 'cbm', M.cbm, 'coef', coef, ...
                'design_wfe', 12, 'requirement', 30);
            row = @(nm) find(strcmp(R.Name, nm) & strcmp(R.Kind, 'summary'));
            lines = ~strcmp(R.Kind, 'summary');
            tc.verifyEqual(R.Total_nm(row('TOTAL')), coef.total, 'RelTol', 1e-12);
            pay = hypot(coef.total, 12);
            tc.verifyEqual(R.Total_nm(row('Payload WFE')), pay, 'RelTol', 1e-12);
            tc.verifyEqual(R.Total_nm(row('WFE Margin')), 1 - pay / 30, 'RelTol', 1e-12);
            inP = lines & strcmp(R.Section, 'Primary');
            tc.verifyEqual(R.Post_WFC_nm(row('Primary WFE')), ...
                sqrt(sum(R.Post_WFC_nm(inP).^2)), 'RelTol', 1e-12);
            tc.verifyEqual(R.Total_nm(row('Everything Else')), ...
                sqrt(sum(R.Total_nm(lines & ~strcmp(R.Section, 'Primary')).^2)), 'RelTol', 1e-12);
            % no control, no metrology: post == pre, met == pre, total == v1
            rc0 = macos.budget.cov_budget(M, spec);
            c0 = macos.budget.to_coef(rc0, M, spec);
            tc.verifyEqual(c0.E3_Tilt.alpha_post, c0.E3_Tilt.alpha);
            tc.verifyEqual(c0.E3_Tilt.alpha_met_post, c0.E3_Tilt.alpha);
            tc.verifyEqual(c0.total, rc0.total * tonm, 'RelTol', 1e-12);
            % the two writers
            wd = tempname; mkdir(wd); cl = onCleanup(@() rmdir(wd, 's'));
            macos.budget.write_xlsx(R, coef, fullfile(wd, 'b.xlsx'));
            macos.budget.write_csv(R, fullfile(wd, 'b.csv'));
            tc.verifyTrue(isfile(fullfile(wd, 'b.xlsx')) && isfile(fullfile(wd, 'b.csv')));
        end

        function test_workbook_formulas_recalculate(tc)
            % The live-formula calculator: LibreOffice (headless) recalculates
            % the written workbook and its Payload WFE equals MATLAB's
            % hypot(coef.total, design) to 1e-9; an INPUT sigma edited in the
            % file re-rolls the payload to the predicted value (a workbook
            % whose formulas do not recompute is not the deliverable).
            [so, ~] = system('command -v soffice');
            tc.assumeEqual(so, 0, 'soffice (LibreOffice) not installed');
            [M, spec, met, ctrl] = tBudgetKernel.two_stage_model();
            rc = macos.budget.cov_budget(M, spec, 'control', ctrl, 'met', met);
            coef = macos.budget.to_coef(rc, M, spec);
            wd = tempname; mkdir(wd); cl = onCleanup(@() rmdir(wd, 's'));
            f = fullfile(wd, 'budget.xlsx');
            info = macos.budget.write_workbook(coef, f, 'design_wfe', 12, ...
                'requirement', 30, 'title', 'synthetic');
            % the formulas are formulas, not text
            raw = tBudgetKernel.sheet_xml(f, 1);
            tc.verifyNotEmpty(regexp(raw, '<f>C4\*Coefficients!D4</f>', 'once'));
            tc.verifyNotEmpty(regexp(raw, '<f>SQRT\(SUMSQ\(K\d+,K\d+\)\)</f>', 'once'));
            V = tBudgetKernel.recalc(f, wd);
            rp = info.rows.Payload_WFE;
            pay = hypot(coef.total, 12);
            tc.verifyEqual(V{rp, 11}, pay, 'RelTol', 1e-9);
            tc.verifyEqual(V{info.rows.TOTAL, 11}, coef.total, 'RelTol', 1e-9);
            tc.verifyEqual(V{info.rows.WFE_Margin, 11}, 1 - pay/30, 'RelTol', 1e-9);
            % displayed precision (Dave 2026-10-16: General showed every
            % digit): the cells AS SHOWN by LibreOffice -- nm to three
            % decimals, the Margin a percent, coefficients scientific, the
            % sigma inputs untouched (General)
            Sw = tBudgetKernel.shown(f, wd, 1);
            tc.verifyMatches(Sw{4, 5}, '^-?\d+\.\d{3}$', 'WFE!E4');
            tc.verifyMatches(Sw{info.rows.TOTAL, 11}, '^\d+\.\d{3}$', 'TOTAL');
            tc.verifyMatches(Sw{info.rows.Payload_WFE, 11}, '^\d+\.\d{3}$', 'Payload');
            tc.verifyMatches(Sw{info.rows.WFE_Margin, 11}, '^-?\d+\.\d%$', 'Margin');
            % the sigma inputs too (Dave 2026-10-16): "0.000", the stored
            % value untouched -- the uniform despace h/sqrt(3) reads 115.470
            tc.verifyMatches(Sw{4, 3}, '^\d+\.\d{3}$', 'WFE!C4 (input)');
            tc.verifyEqual(str2double(Sw{4, 3}), V{4, 3}, 'AbsTol', 5e-4);
            rd = find(strcmp(Sw(:, 2), 'E3 Despace'));
            tc.verifyEqual(Sw{rd, 3}, '115.470');
            tc.verifyEqual(V{rd, 3}, 200 / sqrt(3), 'RelTol', 1e-12);
            % the legend: white cells are inputs, gray cells formulas
            tc.verifyMatches(Sw{1, 13}, '(?i)white.*input.*gr[ae]y.*formula');
            Sc = tBudgetKernel.shown(f, wd, 2);
            tc.verifyMatches(Sc{4, 4}, '^-?\d\.\d{3}E[+-]\d{2}$', 'Coefficients!D4');
            tc.verifyMatches(Sc{4, 11}, '^-?\d\.\d{3}E[+-]\d{2}$', 'Coefficients!K4');
            % every line's Total cell == the tool's line total
            keys = coef.meta.order;
            for r = info.rows.lines(1):info.rows.lines(2)
                nm = V{r, 2};
                k = find(cellfun(@(q) strcmp(coef.(q).name, nm), keys), 1);
                tc.verifyEqual(V{r, 11}, coef.(keys{k}).total, ...
                    'AbsTol', 1e-9 * max(1, coef.total), nm);
            end
            % edit one INPUT sigma in the file -- the line with the largest
            % post-WFC term (a fully actuated line's post coefficient is 0
            % and rightly does not move the payload): the payload re-rolls
            rr = info.rows.lines(1):info.rows.lines(2);
            [~, ib] = max(cellfun(@(x) x, V(rr, 6)));
            r1 = rr(ib);
            k1 = find(cellfun(@(q) strcmp(coef.(q).name, V{r1, 2}), keys), 1);
            e1 = coef.(keys{k1});
            s2 = 3 * e1.sigma_eng;
            tBudgetKernel.set_input(f, wd, sprintf('C%d', r1), s2);
            V2 = tBudgetKernel.recalc(f, wd);
            t1 = hypot(s2 * e1.coef_post, e1.wfe_met_post);
            pred = sqrt(coef.total^2 - e1.total^2 + t1^2 + 12^2);
            tc.verifyEqual(V2{rp, 11}, pred, 'RelTol', 1e-9);
            tc.verifyGreaterThan(abs(V2{rp, 11} - V{rp, 11}), 1e-6 * pay);
        end

        function test_workbook_wfs_habex_reading(tc)
            % Dave 2026-10-16: the WFS sigma is WHITE per-sample noise
            % (Eq. 32) by default, and the HabEx reading (alpha = 1, the
            % sigma passes whole) stays reachable through ONE cell, which
            % the Log names: the WFS line's post-WFC coefficient
            % Coefficients!M<row>.  Typing 1 there makes the line's Post
            % WFC equal its sigma and re-rolls the payload.
            [so, ~] = system('command -v soffice');
            tc.assumeEqual(so, 0, 'soffice (LibreOffice) not installed');
            [M, spec, met, ctrl] = tBudgetKernel.two_stage_model();
            rc = macos.budget.cov_budget(M, spec, 'control', ctrl, 'met', met);
            coef = macos.budget.to_coef(rc, M, spec);
            wd = tempname; mkdir(wd); cl = onCleanup(@() rmdir(wd, 's'));
            f = fullfile(wd, 'budget.xlsx');
            info = macos.budget.write_workbook(coef, f, 'design_wfe', 12, ...
                'requirement', 30, 'title', 'synthetic');
            keys = coef.meta.order;
            kw = keys{cellfun(@(q) strcmp(coef.(q).kind, 'wfs'), keys)};
            w = coef.(kw);
            V = tBudgetKernel.recalc(f, wd);
            rr = info.rows.lines(1):info.rows.lines(2);
            rw = rr(strcmp(V(rr, 2), w.name));
            tc.verifyNumElements(rw, 1);
            % default: the per-sample leak, sqrt(rank/nw) of the sigma
            tc.verifyEqual(V{rw, 6}, w.wfe, 'RelTol', 1e-9);
            tc.verifyLessThan(V{rw, 6}, 0.5 * w.sigma_eng);
            tc.verifyEqual(w.coef_post, sqrt(rank(M.J(:, ctrl)) / M.nw), 'RelTol', 1e-12);
            % the Log names the cell and both readings
            lg = tBudgetKernel.sheet_xml(f, 6);
            cell_ = sprintf('Coefficients!M%d', rw);
            tc.verifyNotEmpty(strfind(lg, cell_), 'Log does not name the WFS cell');
            tc.verifyNotEmpty(regexpi(lg, 'HabEx', 'once'));
            % HabEx reading: type 1 in that cell
            tBudgetKernel.set_input(f, wd, sprintf('M%d', rw), 1, 2);
            V2 = tBudgetKernel.recalc(f, wd);
            tc.verifyEqual(V2{rw, 6}, w.sigma_eng, 'RelTol', 1e-12);
            t1 = hypot(w.sigma_eng, w.wfe_met_post);
            pred = sqrt(coef.total^2 - w.total^2 + t1^2 + 12^2);
            tc.verifyEqual(V2{info.rows.Payload_WFE, 11}, pred, 'RelTol', 1e-9);
        end

        function test_matches_run_met_rmsw(tc)
            % cov_budget on one full-cov group == sqrt(trace(X*G)/nw)
            rng(21); Nw = 400; nz = 24; Jk = randn(Nw, nz);
            G = Jk' * Jk; v = rand(nz, 1) * 1e-12; X = diag(v);
            rmsw = sqrt(trace(X * G) / Nw);
            M = struct('G', G, 'nw', Nw, 'Jk', Jk, 'cbm', 1, ...
                'meta', struct('channel', {repmat({'dwdx'}, nz, 1)}));
            cm = struct('type', 'full', 'Sigma', X, 'class', 'length_si', ...
                'var', NaN, 'sigma_native', sqrt(mean(v)), ...
                'sampler', @() sqrt(v) .* randn(nz, 1));
            spec = struct('groups', struct('name', 'all', 'cols', true(nz, 1), 'cm', cm));
            rc = macos.budget.cov_budget(M, spec);
            tc.verifyEqual(rc.total, rmsw, 'RelTol', 1e-12);
        end

        function test_distributions_cov_eq_mc(tc)
            rng(4); Nw = 400; nz = 10; Jk = randn(Nw, nz);
            M = struct('G', Jk'*Jk, 'nw', Nw, 'Jk', Jk, 'cbm', 1, ...
                'meta', struct('channel', {repmat({'dwdx'}, nz, 1)}));
            h = 2e-6; nc = 5;
            cu = false(nz,1); cu(1:5)=true; cw = false(nz,1); cw(6:10)=true;
            cmU = struct('type','iid','var',h^2/3,'sigma_native',h/sqrt(3), ...
                'class','length_si','Sigma',[],'sampler',@() (2*rand(nc,1)-1).*h);
            cmW = struct('type','iid','var',h^2,'sigma_native',h, ...
                'class','length_si','Sigma',[],'sampler',@() sign(randn(nc,1)).*h);
            spec = struct('groups', [ ...
                struct('name','unif','cols',cu,'cm',cmU), ...
                struct('name','worst','cols',cw,'cm',cmW)]);
            rc = macos.budget.cov_budget(M, spec);
            rm = macos.budget.mc_budget(M, spec, 'ndraw', 20000, 'seed', 2);
            tc.verifyEqual(rc.wfe, rm.wfe, 'RelTol', 0.05);
        end

        function test_assemble_units_and_warnings(tc)
            % nested ox (mm deck) + oz figure; units nrad/nm reconciled.
            rng(5); Nw = 400;
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'}; cn = {}; iE = []; df = [];
            for e = [3 4], for d = 1:6
                cn{end+1} = sprintf('Elt %d %s', e, dofs{d}); iE(end+1)=e; df(end+1)=d-1; %#ok<AGROW>
            end, end
            ox = struct('dwdxall', randn(Nw,12), 'channel_names', {cn}, ...
                'iElt', iE, 'dof_idx', df, 'kind', {repmat({'RigidBody'},12,1)}, ...
                'w0_stacked', randn(Nw,1), 'cbm', 1e-3, 'base_units', 'mm');
            ox.dwdxall(7,:) = NaN;
            zc = [4 5 6 7 8 11]; cnz = {}; for mm = zc, cnz{end+1}=sprintf('Elt 3 Z%d',mm); end %#ok<AGROW>
            oz = struct('dwdxall', randn(Nw,6), 'channel_names', {cnz}, ...
                'iElt', repmat(3,6,1), 'mode', zc(:), ...
                'kind', {repmat({'MonZern'},6,1)}, 'w0_stacked', ox.w0_stacked, 'cbm', 1e-3);
            S = struct('ox', ox, 'oz', oz, 'os', [], 'og', []);
            M = macos.budget.assemble(S, 'trans_per_metre', false);
            tc.verifyEqual(size(M.J), [Nw 18]);
            tc.verifyEqual(M.nw, Nw - 1);                 % NaN row dropped
            T = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid',''; ...
                'E3 LowFig','3','','4-10','','dwdz','normal',10,'nm',NaN,NaN,'iid',''}, ...
                'VariableNames', {'Name','Elements','DOFs','Modes','Param', ...
                'Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
            tc.verifyWarning(@() macos.budget.spec_from_table(T, M), 'budget:spec'); % orphans
            spec = macos.budget.spec_from_table(T, M);
            % 50 nrad -> 5e-8 rad; 10 nm on an mm deck -> 1e-5 mm native
            tc.verifyEqual(spec.groups(1).cm.sigma_native, 5e-8, 'RelTol', 1e-12);
            tc.verifyEqual(spec.groups(2).cm.sigma_native, 1e-5, 'RelTol', 1e-12);
            tc.verifyEqual(spec.groups(1).cm.class, 'angle');
            tc.verifyEqual(spec.groups(2).cm.class, 'length_base');
            rc = macos.budget.cov_budget(M, spec);
            rm = macos.budget.mc_budget(M, spec, 'ndraw', 8000, 'seed', 2);
            tc.verifyEqual(rc.total, rm.total, 'RelTol', 0.08);
        end

        function test_base_harvest_converts_translation_per_metre(tc)
            % A dw_dx harvest with trans_output='base' (the dw_dx default
            % since 2026-10-06: translation columns per BaseUnit) must give
            % the SAME model as its per-SI-metre twin -- the sigma path
            % (covariance_model) converts length sigmas to metres.  Red on
            % a0961dc: assemble gated the conversion on exist('macos.
            % dwdx_trans_per_metre','file'), which is 0 for a package
            % function, so on an mm deck every decenter / despace
            % contribution came out 1/cbm = 1000x small.
            rng(6); Nw = 300;
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'}; cn = {}; iE = []; df = [];
            for e = [3 4], for d = 1:6
                cn{end+1} = sprintf('Elt %d %s', e, dofs{d}); iE(end+1)=e; df(end+1)=d-1; %#ok<AGROW>
            end, end
            cbm = 1e-3;
            si = struct('dwdxall', randn(Nw,12), 'channel_names', {cn}, ...
                'iElt', iE, 'dof_idx', df, 'kind', {repmat({'RigidBody'},12,1)}, ...
                'cbm', cbm, 'base_units', 'mm', 'trans_output', 'si');
            bu = si;  bu.trans_output = 'base';
            t = df >= 3;
            bu.dwdxall(:, t) = bu.dwdxall(:, t) * cbm;     % per BaseUnit
            % 'keep': the unit conversion alone (removing the random
            % columns' piston rounds at ~1e-12 relative)
            Ms = macos.budget.assemble(si, 'piston', 'keep');
            Mb = macos.budget.assemble(bu, 'piston', 'keep');
            tc.verifyEqual(Mb.J, Ms.J, 'RelTol', 1e-12);
            tc.verifyEqual(Mb.trans_output, 'si');
            T = cell2table({'Dec','all','decenter','','','dwdx','normal',100,'nm',NaN,NaN,'iid',''}, ...
                'VariableNames', {'Name','Elements','DOFs','Modes','Param', ...
                'Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
            rs = macos.budget.cov_budget(Ms, macos.budget.spec_from_table(T, Ms));
            rb = macos.budget.cov_budget(Mb, macos.budget.spec_from_table(T, Mb));
            tc.verifyEqual(rb.total, rs.total, 'RelTol', 1e-12);
        end

        function test_zernike_decomp_recovers_modes(tc)
            n = 64; [X, Y] = meshgrid(linspace(-1, 1, n));
            mask = (X.^2 + Y.^2) <= 1;
            Z5 = macos.noll_mode(mask, 5); Z6 = macos.noll_mode(mask, 6);
            w2d = 3 * Z5 - 2 * Z6;
            [c, ~, res] = macos.budget.zernike_decomp(w2d, mask, [4 5 6 7]);
            tc.verifyEqual(c(:)', [0 3 -2 0], 'AbsTol', 1e-8);
            tc.verifyLessThan(max(abs(res)), 1e-8);
        end

        function test_run_budget_end_to_end_synthetic(tc)
            rng(5); Nw = 400;
            dofs = {'Rx','Ry','Rz','Tx','Ty','Tz'}; cn = {}; iE = []; df = [];
            for e = [3 4], for d = 1:6
                cn{end+1} = sprintf('Elt %d %s', e, dofs{d}); iE(end+1)=e; df(end+1)=d-1; %#ok<AGROW>
            end, end
            ox = struct('dwdxall', randn(Nw,12), 'channel_names', {cn}, ...
                'iElt', iE, 'dof_idx', df, 'kind', {repmat({'RigidBody'},12,1)}, ...
                'w0_stacked', randn(Nw,1), 'cbm', 1e-3, 'base_units', 'mm');
            wd = tempname; mkdir(wd); c = onCleanup(@() rmdir(wd, 's'));
            T = cell2table({ ...
                'E3 Tilt','3','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid',''; ...
                'E3 Dec','3','decenter','','','dwdx','uniform',NaN,'nm',-300,300,'iid',''; ...
                'E4 Trans','4','Tx,Ty,Tz','','','dwdx','normal',100,'nm',NaN,NaN,'iid',''}, ...
                'VariableNames', {'Name','Elements','DOFs','Modes','Param', ...
                'Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
            art = run_budget(ox, 'spec', T, 'ndraw', 2000, 'name', 'synth', ...
                'out_dir', wd, 'trans_per_metre', false, 'figures', true, 'visible', false);
            tc.verifyTrue(isfile(art.xlsx));
            tc.verifyTrue(isfile(art.csv));
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(all(cellfun(@isfile, art.figs)));
            tc.verifyGreaterThan(art.total_nm, 0);
            tc.verifyTrue(isfinite(art.total_nm));
            tc.verifyEqual(art.cov.total, art.mc.total, 'RelTol', 0.1);
            % the two sheets exist in the fresh workbook
            tc.verifyTrue(ismember('Coefficients', sheetnames(art.xlsx)));
            tc.verifyTrue(ismember('WFE', sheetnames(art.xlsx)));
        end
    end
end
