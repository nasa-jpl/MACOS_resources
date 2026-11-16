classdef tRefRays < matlab.unittest.TestCase
%TREFRAYS  macos.ref_rays -- the reference rays of the source grid (the CLI's RRAY / PRAY through
%   the api's ref_rays; BRIEF_to_bindings B.2, 2026-10-17); twin of pymacos test_ref_rays.py.
%   Deck: Rx_Cass_FarField (circular aperture 4 m, collimated along +z, xGrid = -x, yGrid = -y),
%   model 128.  The CLI numbers are what `rray` / `pray` printed (both compilers, the fixed build).
%   The geometric leg is the commands' definition, from the deck's own lines: RRAY's rays at the
%   aperture edge (radius Aperture/2) on the grid frame's +x, +y, -x, -y axes, PRAY's on a ring at
%   0.9 of it every 45 degrees from +x -- each the nearest grid ray that passes, so within two
%   grid spacings (4 m / 126 = 0.032 m).  PRAY's first point read 9.0 before the fix (0.17 m off
%   the ring).  Each offset is tied to its index through macos.ray: the crossing at element 1
%   less the chief ray's, projected on xGrid / yGrid.
    properties (Constant)
        Model = 128
        RrayId = [1 6416 12443 6292 15]
        RrayOff = [0 1.9683 0.0317 -1.9683 0.0317; 0 0.0317 1.9683 0.0317 -1.9683]
        PrayId = [1 6411 10941 12241 10861 6297 1515 217 1595]
        PrayOff = [0 1.8095 1.2698 0.0317 -1.2698 -1.8095 -1.2698 0.0317 1.2698; ...
                   0 0.0317 1.2698 1.8095 1.2698 0.0317 -1.2698 -1.8095 -1.2698]
        TwoSpacings = 0.065
    end
    methods (Static, Access = private)
        function [ne, deck] = cass()
            deck = rx_fixture_path('Rx_Cass_FarField.in');
            macos.init(tRefRays.Model);
            macos.load_rx(deck);
            ne = macos.num_elt();
        end
        function [rad, xg, yg] = deck_lines(deck)
            txt = fileread(deck);
            rad = str2double(regexp(txt, 'Aperture=\s*(\S+)', 'tokens', 'once')) / 2;
            t = regexp(txt, 'xGrid=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');  xg = str2double(t(:));
            t = regexp(txt, 'yGrid=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');  yg = str2double(t(:));
        end
    end
    methods (Test)
        function test_ref_rays_reproduce_the_cli(tc)
            tRefRays.cass();
            r = macos.ref_rays('rray');
            tc.verifyEqual(r.id, tc.RrayId);
            tc.verifyEqual(r.offset, tc.RrayOff, 'AbsTol', 5e-5);
            r = macos.ref_rays('pray');
            tc.verifyEqual(r.id, tc.PrayId);
            tc.verifyEqual(r.offset, tc.PrayOff, 'AbsTol', 5e-5);
        end
        function test_ref_rays_sit_on_their_points(tc)
            [~, deck] = tRefRays.cass();
            rad = tRefRays.deck_lines(deck);
            t = deg2rad(45 * (0:7));
            pts = {rad * [1 0 -1 0; 0 1 0 -1], 0.9 * rad * [cos(t); sin(t)]};
            kinds = {'rray', 'pray'};
            for k = 1:2
                r = macos.ref_rays(kinds{k});
                tc.verifyTrue(all(r.id > 0));
                miss = vecnorm(r.offset(:, 2:end) - pts{k});
                tc.verifyLessThan(miss, tc.TwoSpacings);
            end
        end
        function test_pray_ring_is_symmetric(tc)
            tRefRays.cass();
            o = macos.ref_rays('pray').offset(:, 2:end);
            % +x / -x mirror in x, +y / -y in y, the opposite diagonals in both
            tc.verifyEqual(o(1, 1), -o(1, 5), 'AbsTol', 1e-12);
            tc.verifyEqual(o(2, 3), -o(2, 7), 'AbsTol', 1e-12);
            tc.verifyEqual(o(:, 2), -o(:, 6), 'AbsTol', 1e-12);
            tc.verifyEqual(o(:, 4), -o(:, 8), 'AbsTol', 1e-12);
        end
        function test_ref_rays_offset_is_the_rays_own(tc)
            [~, deck] = tRefRays.cass();
            [~, xg, yg] = tRefRays.deck_lines(deck);
            kinds = {'rray', 'pray'};
            for k = 1:2
                r = macos.ref_rays(kinds{k});
                p0 = macos.ray(1).pos(:, 1);
                for j = 2:numel(r.id)
                    q = macos.ray(r.id(j));
                    tc.verifyTrue(q.ok);
                    d = q.pos(:, 1) - p0;
                    tc.verifyEqual([xg' * d; yg' * d], r.offset(:, j), 'AbsTol', 1e-9);
                end
            end
        end
        function test_rray_clears_the_slots_it_does_not_fill(tc)
            tRefRays.cass();
            macos.ref_rays('pray');
            [id, off] = mmacos('ref_rays', 1, zeros(9, 1), zeros(2, 9), 9);
            tc.verifyEqual(id(1:5)', tc.RrayId);
            tc.verifyEqual(id(6:9)', zeros(1, 4));
            tc.verifyEqual(off(:, 6:9), zeros(2, 4));
        end
        function test_ref_rays_refuses_a_bad_kind(tc)
            tRefRays.cass();
            tc.verifyEqual(macos.ref_rays('rray').id, tc.RrayId);
            tc.verifyError(@() macos.ref_rays('tpr'), ?MException);
            for bad = [0 3]
                tc.verifyError(@() mmacos('ref_rays', bad, zeros(9, 1), zeros(2, 9), 9), ?MException);
            end
        end
        function test_ref_rays_leave_the_full_trace_unchanged(tc)
            ne = tRefRays.cass();
            macos.trace(ne);  W0 = macos.opd();  K0 = macos.opd_mask();
            macos.ref_rays('pray');  macos.ref_rays('rray');
            % read WITHOUT a re-trace: the source grid RRAY / PRAY rebuild left nGridPts one
            % short, so opd / opd_mask returned a crop of the last map
            tc.verifyEqual(macos.opd_mask(), K0);
            tc.verifyEqual(macos.opd() .* K0, W0 .* K0, 'AbsTol', 0);
            macos.trace(ne);
            tc.verifyEqual(macos.opd_mask(), K0);
            tc.verifyEqual(macos.opd() .* K0, W0 .* K0, 'AbsTol', 0);
        end
    end
end
