classdef tEdgeSensors < matlab.unittest.TestCase
    % Sprint 2D S2: edge-sensor ingestion (dedx) + independent
    % validation of the parsed model against the segment frames.
    %
    % The validation needs no surface re-evaluation: the generator's
    % algebra implies identities the ingested rows must satisfy —
    %   (1) each adjacency row's per-segment translation triplet is
    %       normhat expressed in that segment's triad -> unit 2-norm,
    %       and T_i*del_i' == -T_j*del_j' recovers the SAME world
    %       normal from both segments (validates column mapping AND
    %       the segment_rx frames at once);
    %   (2) rotation triplets are moment-arm cross terms about ONE
    %       shared sensor-point offset rho (the generator uses rhoi
    %       for both segments): th' = cross(rho, del') -> rho is the
    %       least-squares solution of 6 equations and the residual is
    %       ~0 only if columns, frames, and packing are all right;
    %   (3) the recovered sensor point pSeg_i + rho sits laterally
    %       between the two segment centers.

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
        function test_shape_and_master_row(tc)
            e = tc.es;
            tc.verifyEqual(e.nmeas, 79);           % 1 + 2*39 adjacencies
            tc.verifyEqual(e.nstate, 114);         % 6 * 19
            tc.verifyEqual(e.dof, 6);
            tc.verifyEqual(e.nseg, 19);
            tc.verifySize(e.dedx, [79 114]);
            tc.verifyEqual(e.meas_to_seg(:,1), [1;1]);
            % Row 1 = master-segment absolute piston: exactly one
            % nonzero, +1 on Seg1's trans_z column (col 6).
            r1 = e.dedx(1,:);
            tc.verifyEqual(r1(6), 1);
            tc.verifyEqual(nnz(r1), 1);
        end

        function test_normal_recovery_and_unit_norms(tc)
            e = tc.es; f = tc.seg.frames;
            for m = 2:e.nmeas
                i = e.meas_to_seg(1,m); j = e.meas_to_seg(2,m);
                ri = e.dedx(m, (i-1)*6+(1:6));
                rj = e.dedx(m, (j-1)*6+(1:6));
                Ti = [f(i).xhat f(i).yhat f(i).zhat];
                Tj = [f(j).xhat f(j).yhat f(j).zhat];
                ni =  Ti * ri(4:6)';    % normhat from segment i's row
                nj = -Tj * rj(4:6)';    % ... and from segment j's row
                tc.verifyEqual(norm(ni), 1, 'AbsTol', 1e-8);   % Hx text precision
                tc.verifyEqual(norm(ni - nj), 0, 'AbsTol', 1e-8, ...
                    sprintf('row %d: normals from segs %d/%d disagree', m, i, j));
                % Sensor normal is near the parent axis (Seg1 zhat),
                % never wildly off (< ~6 deg at the outer ring).
                tc.verifyGreaterThan(dot(ni, f(1).zhat), 0.99);
            end
        end

        function test_sensor_point_recovery(tc)
            e = tc.es; f = tc.seg.frames;
            width = 1600;
            z1 = f(1).zhat;
            for m = 2:e.nmeas
                i = e.meas_to_seg(1,m); j = e.meas_to_seg(2,m);
                ri = e.dedx(m, (i-1)*6+(1:6));
                rj = e.dedx(m, (j-1)*6+(1:6));
                % th' = cross(rho, del') for BOTH segments, same rho:
                % stack -[del]_x * rho = -th'  ->  A*rho = b.
                cm = @(v) [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
                A = [-cm(ri(4:6)'); -cm(rj(4:6)')];
                b = -[ri(1:3)'; rj(1:3)'];
                rho = A \ b;
                res = norm(A*rho - b) / max(norm(b), 1);
                tc.verifyLessThan(res, 1e-6, ...   % Hx text precision
                    sprintf('row %d: rotation rows inconsistent with a shared sensor point', m));
                % Solving [del]_x * rho = th' recovers rho = pSeg_i - pr
                % (verified empirically: pr = rpt_i - rho lands on the
                % chord midpoint).  The sensor point must sit laterally
                % at the shared-edge midpoint; the residual lateral
                % offset is the surface-sag projection (~10 mm at the
                % outer ring on the 25.8-m-focal parent), bounded well
                % under 0.05 width.
                pr = f(i).rpt - rho;
                mid = 0.5*(f(i).rpt + f(j).rpt);
                d = pr - mid;
                dlat = d - dot(d, z1)*z1;
                tc.verifyLessThan(norm(dlat), 0.05*width, ...
                    sprintf('row %d: sensor point off the shared edge', m));
            end
        end
    end
end
