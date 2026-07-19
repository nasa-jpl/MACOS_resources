classdef tRunSegmentation < matlab.unittest.TestCase
    % Segmentation stage runner (design/runners/run_segmentation) on
    % the e5 monolithic parent: bare-splice parity, engine-truth
    % footprints, physical apertures, artifact reload verification.

    methods (TestClassSetup)
        function paths(tc) %#ok<MANU>
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            addpath(fullfile(res_root, 'mmacos', 'design', 'runners'));
        end
    end

    methods (Test)
        function test_pie_end_to_end(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            bin = fullfile(res_root, 'segmirmaker', ...
                           'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(bin), 'SegMirMaker not built');
            wd = tempname; mkdir(wd);
            cwd = onCleanup(@() rmdir(wd, 's'));
            art = run_segmentation( ...
                fullfile(res_root, 'segmirmaker', 'test_in', 'e5mono.in'), ...
                'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'out_dir', wd, 'name', 'pie', 'views', false, ...
                'verbose', false);
            % artifacts on disk
            tc.verifyTrue(isfile(art.in));
            tc.verifyTrue(isfile(art.hx));
            tc.verifyTrue(isfile(art.report));
            tc.verifyTrue(isfile(art.footprints_png));
            % standalone reload verification is the runner's own gate
            tc.verifyTrue(art.verified);
            % pie tiling: 7 segments, footprint labels cover all 7
            tc.verifyEqual(art.seg.nseg, 7);
            tc.verifyEqual(numel(unique(art.labels(art.labels > 0))), 7);
            % apertures clip SOME gap rays but keep the vast majority
            tc.verifyGreaterThan(art.pass_ap, 0.9*art.pass_bare);
            tc.verifyLessThanOrEqual(art.pass_ap, art.pass_bare);
            % parity: segmented rms within a few percent of the parent
            tc.verifyLessThan(abs(art.rms_ap - art.rms_parent) ...
                / art.rms_parent, 0.10);
        end
    end
end
