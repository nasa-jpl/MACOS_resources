classdef tEdgeSensors < matlab.unittest.TestCase
    % Sprint 2D S2 (+ 2026-07-19 rework): edge-sensor ingestion (dedx)
    % and independent validation of the parsed model against the
    % segment frames.
    %
    % Measurement model under test (SegMirMaker, Dave's spec): per
    % SHARED EDGE, 2 sensor locations at +/-SensorOff from the edge
    % midpoint along the edge direction, 3 axes per location (1 =
    % surface normal, 2/3 = in-plane pair) -- 6 differential rows per
    % edge, NO absolute-piston anchor row.  The validation needs no
    % surface re-evaluation: the generator's algebra implies
    % identities the ingested rows must satisfy --
    %   (1) each row's per-segment translation triplet is the axis
    %       expressed in that segment's triad -> unit 2-norm, and
    %       T_i*del_i' == -T_j*del_j' recovers the SAME world axis
    %       from both segments (validates columns AND frames);
    %   (2) rotation triplets are per-segment moment-arm cross terms:
    %       w_s = T_s*th_s' equals +/- rho_s x axis with rho_s = the
    %       sensor point offset from THAT segment's center -> the
    %       sensor point recovered from segment i and from segment j
    %       must coincide (perpendicular to the axis);
    %   (3) recovered sensor points sit ON the shared edge at
    %       +/-SensorOff along the edge direction, opposite signs for
    %       the two locations.

    properties
        seg    % shared segment_rx output (hex 2-ring, config 2)
        es     % ingested edge sensors for seg.hx
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));
            res_root = fileparts(fileparts(here));
            tin = fullfile(res_root, 'segmirmaker', 'test_in');
            bin = fullfile(res_root, 'segmirmaker', ...
                           'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(bin), ...
                'SegMirMaker binary not built (source ./makesegmirmaker.sh)');
            tc.seg = macos.design.segment_rx( ...
                fullfile(tin, 'e5mono.in'), 'elt', 1, ...
                'rings', 2, 'grid', 'Hex', 'gap', 50, 'dofs', 6, ...
                'meas_config', 2);
            tc.es = macos.design.edge_sensors(tc.seg.hx);
        end
    end

    methods (Test)
        function test_shape_no_anchor(tc)
            e = tc.es;
            tc.verifyEqual(e.nstate, 114);         % 6 * 19
            tc.verifyEqual(e.dof, 6);
            tc.verifyEqual(e.nseg, 19);
            tc.verifyFalse(e.has_anchor, ...
                'no absolute-piston anchor row (not a measurement)');
            tc.verifyTrue(all(ismember(e.axis, 1:3)));
            tc.verifyTrue(all(ismember(e.loc, 1:2)));
            % 6 rows per unordered shared edge (3 axes x 2 locations)
            pairs = sort(e.meas_to_seg, 1).';
            [up, ~, ic] = unique(pairs, 'rows');
            tc.verifyEqual(e.nmeas, 6*size(up, 1));
            tc.verifyTrue(all(accumarray(ic, 1) == 6));
            % every pair distinct segments
            tc.verifyTrue(all(up(:,1) ~= up(:,2)));
        end

        function test_axis_recovery_and_unit_norms(tc)
            e = tc.es; f = tc.seg.frames;
            z1 = f(1).zhat;
            for m = 1:e.nmeas
                i = e.meas_to_seg(1,m); j = e.meas_to_seg(2,m);
                ri = e.dedx(m, (i-1)*6+(1:6));
                rj = e.dedx(m, (j-1)*6+(1:6));
                Ti = [f(i).xhat f(i).yhat f(i).zhat];
                Tj = [f(j).xhat f(j).yhat f(j).zhat];
                ai =  Ti * ri(4:6)';    % axis from segment i's row
                aj = -Tj * rj(4:6)';    % ... and from segment j's row
                tc.verifyEqual(norm(ai), 1, 'AbsTol', 1e-8);
                tc.verifyEqual(norm(ai - aj), 0, 'AbsTol', 1e-8, ...
                    sprintf('row %d: axes from segs %d/%d disagree', m, i, j));
                if e.axis(m) == 1
                    % surface normal: near the parent axis
                    tc.verifyGreaterThan(dot(ai, z1), 0.99);
                else
                    % in-plane pair: nearly perpendicular to it
                    tc.verifyLessThan(abs(dot(ai, z1)), 0.2, ...
                        sprintf('row %d: in-plane axis leans into the normal', m));
                end
            end
        end

        function test_sensor_points_on_edge(tc)
            % The rows determine each sensor point only PERPENDICULAR
            % to the measurement axis (rho x a annihilates the
            % a-component), so every geometric check below projects
            % out the axis direction first.
            e = tc.es; f = tc.seg.frames;
            width = 1600;  soff = 0.25*width;      % SMM default SensorOff
            z1 = f(1).zhat;
            P = nan(3, e.nmeas);  A = nan(3, e.nmeas);
            for m = 1:e.nmeas
                i = e.meas_to_seg(1,m); j = e.meas_to_seg(2,m);
                ri = e.dedx(m, (i-1)*6+(1:6));
                rj = e.dedx(m, (j-1)*6+(1:6));
                Ti = [f(i).xhat f(i).yhat f(i).zhat];
                Tj = [f(j).xhat f(j).yhat f(j).zhat];
                a  = Ti * ri(4:6)';  a = a / norm(a);
                wi = Ti * ri(1:3)';                 % = rho_i x a
                wj = Tj * rj(1:3)';                 % = -(rho_j x a)
                pi_ = f(i).rpt + cross(a, wi);      % perp-to-a position
                pj_ = f(j).rpt + cross(a, -wj);
                d = pi_ - pj_;  d = d - dot(d, a)*a;
                tc.verifyLessThan(norm(d), 1e-3*width, sprintf( ...
                    'row %d: sensor points from segs %d/%d disagree', m, i, j));
                P(:, m) = pi_;  A(:, m) = a;
            end
            % per row: the (axis-perpendicular part of the) sensor
            % point must sit ON the shared-edge line {mid + t*ehat},
            % at t = +/-SensorOff matching the row's location tag.
            % The in-plane axes are point-local (radhat/tanhat rotate
            % ~18 deg over the 800-mm separation), so each row is
            % validated independently against its own axis.
            for m = 1:e.nmeas
                i = e.meas_to_seg(1,m); j = e.meas_to_seg(2,m);
                a = A(:, m);
                Pp = @(v) v - a*dot(a, v);
                cij  = f(j).rpt - f(i).rpt;
                ehat = cross(z1, cij);  ehat = ehat / norm(ehat);
                mid = 0.5*(f(i).rpt + f(j).rpt);
                pe = Pp(ehat);
                d  = Pp(P(:, m) - mid);
                if norm(pe) < 0.5
                    % axis nearly along the edge: the along-edge
                    % position is unobservable from this row -- only
                    % require the point to be near the edge PLANE
                    tc.verifyLessThan(norm(d - pe*dot(pe, d)/max(pe'*pe, eps)), ...
                        0.08*width, sprintf( ...
                        'row %d (edge %d-%d axis %d): off the edge line', ...
                        m, i, j, e.axis(m)));
                    continue
                end
                t = dot(pe, d) / (pe'*pe);
                res = d - t*pe;
                tc.verifyLessThan(norm(res), 0.05*width, sprintf( ...
                    'row %d (edge %d-%d axis %d): off the edge line', ...
                    m, i, j, e.axis(m)));
                tc.verifyEqual(abs(t), soff, 'RelTol', 0.08, sprintf( ...
                    'row %d (edge %d-%d axis %d): not at +/-SensorOff', ...
                    m, i, j, e.axis(m)));
                tc.verifyEqual(sign(t), 2*e.loc(m) - 3, sprintf( ...
                    'row %d: location tag / offset sign mismatch', m));
            end
        end
    end
end
