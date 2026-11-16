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
            % the report file prints the piston stamp (PLAN 1.2 (www) sweep)
            tc.verifySubstring(fileread(art.report), ...
                'piston removed (harvest opd_ref chief)');
            tc.verifyTrue(all(cellfun(@isfile, art.figs)));
            tc.verifyTrue(ismember('WFE', sheetnames(art.xlsx)));
            tc.verifyTrue(ismember('Coefficients', sheetnames(art.xlsx)));
            tc.verifyGreaterThan(art.total_nm, 0);
            tc.verifyTrue(isfinite(art.total_nm));
            tc.verifyTrue(all(isfinite(art.cov.alpha)));
            % the Fig.6 agreement, on a real harvest
            tc.verifyEqual(art.cov.total, art.mc.total, 'RelTol', 0.12);
        end

        function test_run_budget_without_spec(tc)
            % No spec: suggest_groups proposes the skeleton and run_budget
            % runs it.  Until 2026-10-28 the skeleton gave each optic's
            % Clock group (dwdx DOF 2, an angle) the length unit "nm", and
            % the covariance model refused it with budget:unit.
            wd = tempname; mkdir(wd);
            cl = onCleanup(@() rmdir(wd, 's'));
            art = run_budget(tc.ox, 'name', 'e5hex1_nospec', ...
                'out_dir', wd, 'ndraw', 200, 'visible', false);
            tc.verifyTrue(isfinite(art.total_nm));
            tc.verifyGreaterThan(art.total_nm, 0);
            % every suggested rigid-body group of angle DOFs carries an
            % angle unit, every other one a length unit
            M = macos.budget.assemble(tc.ox);
            T = macos.budget.suggest_groups(M);
            ang = {'rad','mrad','urad','microrad','nrad','arcsec'};
            for k = 1:height(T)
                if ~strcmp(T.Channel{k}, 'dwdx'), continue; end
                is_ang = ismember(T.DOFs{k}, {'tilt', 'clock'});
                tc.verifyEqual(ismember(T.Units{k}, ang), is_ang, sprintf( ...
                    '%s: units "%s"', T.Name{k}, T.Units{k}));
            end
        end

        function test_wfc_stage_cov_equals_mc(tc)
            % The WFC stage (iSeq = 1) on the e5hex1 harvest, Dave's
            % control set (6 DOF per PM segment + the SM's 6), WFS 3 nm
            % per sample, actuators 1 nm (rotations: 1 nm of edge motion).
            % Monte-Carlo with the pseudo-inverse controller in the loop
            % (Eq. 22, sensing + actuator draws) == covariance on every
            % row; the uncontrolled total is the role-free v1 table's.
            here = fileparts(mfilename('fullpath'));
            root = fileparts(fileparts(here));
            addpath(fullfile(root, 'mmacos', 'templates', '50_sensitivities', ...
                'e5hex1_budget'));
            [T, act, wfs] = budget_spec_e5hex1();
            wd = tempname; mkdir(wd);
            cl = onCleanup(@() rmdir(wd, 's'));
            art = run_budget(tc.ox, 'spec', T, 'name', 'e5hex1_wfc', ...
                'out_dir', wd, 'ndraw', 1000, 'control', true, ...
                'wfs_sigma', wfs, 'act_sigma', act, 'figures', false);
            rc = art.cov;  rm = art.mc;
            tc.verifyTrue(rc.controlled);
            tc.verifyEqual(nnz(art.control), 48);           % 8 bodies x 6
            % uncontrolled numbers: the v1 table (DOF type over all optics)
            Tv1 = cell2table({ ...
                'Optics Tilt','all','tilt','','','dwdx','normal',50,'nrad',NaN,NaN,'iid',''; ...
                'Optics Clocking','all','clock','','','dwdx','normal',100,'nrad',NaN,NaN,'iid',''; ...
                'Optics Decenter','all','decenter','','','dwdx','normal',100,'nm',NaN,NaN,'iid',''; ...
                'Optics Despace','all','despace','','','dwdx','uniform',NaN,'nm',-200,200,'iid',''}, ...
                'VariableNames', {'Name','Elements','DOFs','Modes','Param', ...
                'Channel','Dist','Sigma','Units','Min','Max','Cov','Control'});
            M = macos.budget.assemble(tc.ox);
            rc1 = macos.budget.cov_budget(M, macos.budget.spec_from_table(Tv1, M));
            tc.verifyEqual(rc.total_pre, rc1.total, 'RelTol', 1e-12);
            % the actuated groups are null after control (Pu Ju = 0)
            isc = [art.spec.groups.is_control]';
            tc.verifyLessThan(max(rc.wfe(isc)), 1e-9 * max(rc.wfe_pre(isc)));
            % noise rows: one WFS row + one actuator row per control group
            tc.verifyEqual(sum(strcmp({rc.extra.kind}, 'wfs')), 1);
            tc.verifyEqual(sum(strcmp({rc.extra.kind}, 'act')), 8);
            % cov == MC: every non-null group, every noise row, the total
            nn = ~isc;
            tc.verifyEqual(rm.wfe(nn), rc.wfe(nn), 'RelTol', 0.15);
            tc.verifyEqual([rm.extra.wfe]', [rc.extra.wfe]', 'RelTol', 0.15);
            tc.verifyEqual(rm.total, rc.total, 'RelTol', 0.08);
            tc.verifyLessThan(rc.total, rc.total_pre);
            % the coefficient struct and the roll-up carry the same totals
            tonm = art.M.cbm * 1e9;
            tc.verifyEqual(art.coef.total, rc.total * tonm, 'RelTol', 1e-12);
            tc.verifyEqual(art.coef.total_pre_wfc, rc.total_pre * tonm, 'RelTol', 1e-12);
            Tt = art.table;
            r = strcmp(Tt.Name, 'TOTAL') & strcmp(Tt.Kind, 'summary');
            tc.verifyEqual(Tt.Total_nm(r), art.coef.total, 'RelTol', 1e-12);
            tc.verifyEqual(Tt.Pre_WFC_nm(r), rc.total_pre * tonm, 'RelTol', 1e-12);
            tc.verifyEqual(Tt.MC_Post_WFC_nm(r), rm.total * tonm, 'RelTol', 0.1);
            % the live-formula calculator is written beside the plain export
            tc.verifyTrue(isfile(art.workbook));
            tc.verifyEqual(cellstr(sheetnames(art.workbook))', ...
                {'WFE', 'Coefficients', 'WFE BD', 'PSD', 'StrTherm', 'Log'});
        end
    end
end
