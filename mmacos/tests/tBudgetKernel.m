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
            Ms = macos.budget.assemble(si);
            Mb = macos.budget.assemble(bu);
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
