classdef tViewRx < matlab.unittest.TestCase
%TVIEWRX  ray_hist / get_elt_info / view_rx solid rendering (model 256).
%   Engine substrate for the layout visualizer: the per-trace ray-
%   position history (RayPosHist via macos.ray_hist), element metadata
%   (macos.get_elt_info), and the solid-body view_rx default.  Fixture:
%   the manual's CassWithExitPupil.in (no GridFile, self-contained).

    properties
        man
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));            % mmacos/tests
            res_root = fileparts(fileparts(here));              % MACOS_resources
            tc.man = fullfile(fileparts(res_root), 'macos', ...
                              'docs', 'macos-manual', 'examples');
            tc.assumeTrue(isfile(fullfile(tc.man, 'CassWithExitPupil.in')), ...
                'manual examples not present');
            macos.init(256);
            macos.load_rx(fullfile(tc.man, 'CassWithExitPupil.in'));
        end
    end

    methods (Test)

        function test_ray_hist_capture(tc)
            macos.ray_hist('on');
            t = macos.trace();
            h = macos.ray_hist(t.nRays);
            macos.ray_hist('off');
            nE = macos.num_elt();
            tc.verifySize(h.P, [3, t.nRays, nE+1]);
            tc.verifySize(h.ok, [t.nRays, nE+1]);
            tc.verifyGreaterThan(mean(h.ok(:)), 0.9, ...
                'most rays must reach most elements on the Cass');
            % every DRAW-fan crossing must sit within one source-grid
            % spacing of a history ray at the same element (the fan
            % resamples its own rays, so identity is to the grid pitch)
            b = macos.draw_rays3d('YZ', 0, nE);
            P0 = squeeze(h.P(:, :, 1));
            dgrid = 2*max(vecnorm(P0 - mean(P0, 2))) / sqrt(t.nRays);
            maxd = 0;
            for r = 1:b.nray
                for c = 1:b.nper(r)
                    k = b.elt(c, r);
                    if k < 1 || k > nE, continue; end
                    d = squeeze(h.P(:, :, k+1)) - b.P(:, c, r);
                    maxd = max(maxd, min(vecnorm(d)));
                end
            end
            tc.verifyLessThan(maxd, 1.5*dgrid, ...
                'fan crossings must be reproduced by the history rays');
        end

        function test_ray_hist_toggle_dirties_trace(tc)
            % enabling AFTER a trace must still capture on the NEXT
            % trace (the engine skips the retrace unless dirtied --
            % same class as the grid-setter retrace rule)
            macos.ray_hist('off');
            t = macos.trace();
            macos.ray_hist('on');
            t = macos.trace();
            h = macos.ray_hist(t.nRays);
            macos.ray_hist('off');
            tc.verifyGreaterThan(mean(h.ok(:)), 0.9);
        end

        function test_elt_info(tc)
            i2 = macos.get_elt_info(2);
            tc.verifyEqual(i2.type, 'Reflector');
            tc.verifyEqual(i2.ap_type, 1);                 % Circular
            tc.verifyEqual(i2.ap_vec(1), 2.1, 'AbsTol', 1e-12);
            tc.verifyEmpty(i2.poly);
            i1 = macos.get_elt_info(1);
            tc.verifyEqual(i1.type, 'Obscuring');
            i6 = macos.get_elt_info(macos.num_elt());
            tc.verifyEqual(i6.type, 'FocalPlane');
        end

        function test_view_rx_solids(tc)
            f = [tempname '.png'];
            c = onCleanup(@() delete_if_(f)); %#ok<NASGU>
            fig = macos.view_rx('visible', false, 'save', f);
            cf = onCleanup(@() close(fig)); %#ok<NASGU>
            tc.verifyClass(fig, 'matlab.ui.Figure');
            % solid default: lit patches for the two mirrors (+ the
            % Obscuring fill), a camlight, and bundle polylines
            pt = findobj(fig, 'Type', 'patch');
            tc.verifyGreaterThanOrEqual(numel(pt), 2);
            tc.assertNotEmpty(findobj(fig, 'Type', 'light'));
            tc.verifyGreaterThan(numel(findobj(fig, 'Type', 'line')), 10);
            d = dir(f);
            tc.assertNotEmpty(d, 'PNG must be written');
            tc.verifyGreaterThan(d.bytes, 1000);
        end

        function test_view_rx_legacy_modes(tc)
            fig = macos.view_rx('visible', false, ...
                'bodies', 'outline', 'bundle', 'rim', 'nrays', 10);
            c1 = onCleanup(@() close(fig)); %#ok<NASGU>
            tc.verifyClass(fig, 'matlab.ui.Figure');
            fig2 = macos.view_rx('visible', false, ...
                'bodies', 'patch', 'bundle', 'fans');
            c2 = onCleanup(@() close(fig2)); %#ok<NASGU>
            tc.verifyClass(fig2, 'matlab.ui.Figure');
        end
    end
end

% ---------------------------------------------------------------------------
function delete_if_(f)
if exist(f, 'file'), delete(f); end
end
