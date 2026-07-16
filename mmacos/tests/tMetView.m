classdef tMetView < matlab.unittest.TestCase
%TMETVIEW  macos.design.met_view geometry-only rendering tests.
%   Pure-MATLAB (no engine): met_view must render a synthetic 2-segment
%   MET setup ('rays',false short-circuits before macos.has_rx), honor
%   the overlay/edge_off options, and save a PNG.  The full engine-ray
%   path is exercised by the e5_seg worked example.

    methods (Test)

        function test_renders_and_saves(testCase)
            [seg, am] = synth_();
            f = [tempname '.png'];
            c = onCleanup(@() delete_if_(f)); %#ok<NASGU>
            fig = macos.design.met_view(seg, am, 'rays', false, ...
                'visible', false, 'save', f);
            cf = onCleanup(@() close(fig)); %#ok<NASGU>
            testCase.verifyClass(fig, 'matlab.ui.Figure');
            ax = findobj(fig, 'Type', 'axes');
            testCase.verifyNumElements(ax, 3, ...
                '3-D scene + face-on panel + M2-M3 inset');
            d = dir(f);
            testCase.assertNotEmpty(d, 'PNG must be written');
            testCase.verifyGreaterThan(d.bytes, 1000, 'PNG must be non-trivial');
        end

        function test_hex_tiles_and_beams_present(testCase)
            [seg, am] = synth_();
            fig = macos.design.met_view(seg, am, 'rays', false, ...
                'visible', false);
            cf = onCleanup(@() close(fig)); %#ok<NASGU>
            % one hex patch per segment + the hub disc
            pt = findobj(fig, 'Type', 'patch');
            testCase.verifyNumElements(pt, seg.nseg + 1);
            % legend carries the annotation vocabulary (launchers are
            % colored per segment, keyed by the truss entry)
            lg = findobj(fig, 'Type', 'legend');
            testCase.assertNotEmpty(lg);
            testCase.verifyTrue(any(contains(lg.String, 'fiducials')));
            testCase.verifyTrue(any(contains(lg.String, 'segment trusses')));
            testCase.verifyTrue(any(contains(lg.String, 'extra-source truss')));
        end

        function test_overlay_and_edge_off_options(testCase)
            [seg, am] = synth_();
            base = macos.design.met_view(seg, am, 'rays', false, ...
                'visible', false);
            c1 = onCleanup(@() close(base)); %#ok<NASGU>
            n0 = numel(findobj(base, 'Type', 'line'));
            fig = macos.design.met_view(seg, am, 'rays', false, ...
                'visible', false, 'overlay_pts', am.src_pts + 5, ...
                'edge_off', 5);
            c2 = onCleanup(@() close(fig)); %#ok<NASGU>
            n1 = numel(findobj(fig, 'Type', 'line'));
            testCase.verifyGreaterThan(n1, n0, ...
                'overlay_pts + edge_off must add line objects');
        end

        function test_title_annotation(testCase)
            [seg, am] = synth_();
            fig = macos.design.met_view(seg, am, 'rays', false, ...
                'visible', false);
            cf = onCleanup(@() close(fig)); %#ok<NASGU>
            % findall cannot reach the sgtitle layout Text (verified
            % R2026a) -- met_view mirrors the title onto fig.Name.
            testCase.verifyTrue(contains(fig.Name, '2 segments'), ...
                'auto title must carry the segment count');
        end
    end
end

% ---------------------------------------------------------------------------
function [seg, am] = synth_()
%SYNTH_  Minimal 2-segment + hub MET geometry (mm), no engine required.
fr = struct('name', {}, 'rpt', {}, 'xhat', {}, 'yhat', {}, 'zhat', {}, ...
            'lmon', {});
fr(1) = struct('name', 's1', 'rpt', [0;0;0], 'xhat', [1;0;0], ...
               'yhat', [0;1;0], 'zhat', [0;0;1], 'lmon', 100);
fr(2) = struct('name', 's2', 'rpt', [210;0;2], 'xhat', [1;0;0], ...
               'yhat', [0;1;0], 'zhat', [0;0;1], 'lmon', 100);
seg = struct('nseg', 2, 'frames', fr);

% hub fiducials: ring of 3 about (0,0,2000)
thf = 2*pi*(0:2)/3;
fid = [50*cos(thf); 50*sin(thf); 2000*ones(1,3)];
pair = [1 2 2 3 3 1];
src = zeros(3, 0); tgt = zeros(3, 0);
for s = 1:2
    th6 = pi/6 + 2*pi*(0:5)/6;
    L = fr(s).rpt + 70*[cos(th6); sin(th6); zeros(1,6)];
    src = [src, L]; tgt = [tgt, fid(:, pair)]; %#ok<AGROW>
end
% extra-source truss ("around M3"): ring at z = 1500
L3 = [120*cos(th6); 120*sin(th6); 1500*ones(1,6)];
src = [src, L3]; tgt = [tgt, fid(:, pair)];
am = struct('in', '', 'n_beams', size(src, 2), 'src_pts', src, ...
            'tgt_pts', tgt);
end

function delete_if_(f)
if exist(f, 'file'), delete(f); end
end
