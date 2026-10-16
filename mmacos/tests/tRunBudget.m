classdef tRunBudget < matlab.unittest.TestCase
    % run_budget end-to-end on the public 7-segment hex deck e5hex1.in
    % (model 128): a real dw/dx harvest -> covariance + Monte-Carlo error
    % budget -> artifacts.  The worked-example gate (Redding 2014) on a
    % public prescription.  Pure-algebra kernel checks live in
    % tBudgetKernel (no engine); this one exercises the harvest + loader +
    % runner against the engine.
    %
    % Checks:
    %   (1) assemble ingests a live dw_dx_multi struct; metadata recovered
    %   (2) run_budget writes report + xlsx (WFE+Coefficients) + csv + mat + fig
    %   (3) covariance total == Monte-Carlo total (the Fig.6 agreement)
    %   (4) a finite, positive total WFE; per-group alphas finite

    properties
        ox
    end

    methods (TestClassSetup)
        function harvest(tc)
            here = fileparts(mfilename('fullpath'));
            root = fileparts(fileparts(here));
            addpath(fullfile(root, 'mmacos', 'src'));
            addpath(fullfile(root, 'mmacos', 'design', 'runners'));
            rx = fullfile(root, 'mmacos', 'templates', '50_sensitivities', ...
                'e5hex1', 'e5hex1.in');
            tc.assumeTrue(isfile(rx), 'e5hex1.in fixture missing');
            m = macos.Session(128);
            tc.ox = macos.dw_dx_multi(m, rx, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'delta', 1e-8);
        end
    end

    methods (Test)
        function test_assemble_live_struct(tc)
            M = macos.budget.assemble(tc.ox);
            tc.verifyGreaterThan(size(M.J, 2), 6);           % >= one optic's 6 DOF
            tc.verifyTrue(all(strcmp(M.meta.channel, 'dwdx')));
            tc.verifyTrue(any(~isnan(M.meta.dof)));          % DOF metadata present
            tc.verifyGreaterThan(M.nw, 0);
        end

        function test_run_budget_end_to_end(tc)
            wd = tempname; mkdir(wd);
            cl = onCleanup(@() rmdir(wd, 's'));
            T = cell2table({ ...
                'Optics Tilt','all','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid',''; ...
                'Optics Decenter','all','decenter','','','dwdx','normal',100,'nm',NaN,NaN,'iid',''; ...
                'Optics Despace','all','despace','','','dwdx','uniform',NaN,'nm',-200,200,'iid',''}, ...
                'VariableNames', {'Name','Elements','DOFs','Modes','Param', ...
                'Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
            art = run_budget(tc.ox, 'spec', T, 'name', 'e5hex1', ...
                'out_dir', wd, 'ndraw', 800, 'visible', false);
            tc.verifyTrue(isfile(art.xlsx));
            tc.verifyTrue(isfile(art.csv));
            tc.verifyTrue(isfile(art.mat));
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(all(cellfun(@isfile, art.figs)));
            tc.verifyTrue(ismember('WFE', sheetnames(art.xlsx)));
            tc.verifyTrue(ismember('Coefficients', sheetnames(art.xlsx)));
            tc.verifyGreaterThan(art.total_nm, 0);
            tc.verifyTrue(isfinite(art.total_nm));
            tc.verifyTrue(all(isfinite(art.cov.alpha)));
            % the Fig.6 agreement, on a real harvest
            tc.verifyEqual(art.cov.total, art.mc.total, 'RelTol', 0.12);
        end
    end
end
