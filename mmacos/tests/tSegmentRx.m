classdef tSegmentRx < matlab.unittest.TestCase
    % Sprint 2D S1: segment_rx splice — SegMirMaker blocks replace one
    % parent element; the merged .in must be structurally right AND
    % load/trace clean in the engine.
    %
    % Parent fixture: segmirmaker/test_in/e5mono.in (elt 1 = the 8-m
    % FreeForm primary).  The 2-ring hex / gap 50 / 6-DOF / config-2
    % case mirrors the committed hand-assembled reference e5hex2.in,
    % whose engine behavior pins the expectations here (25 elements by
    % read order, all rays land, rmsWFE within sampling of the parent).
    %
    % Engine tests run at model_size 512 (nGridMat=256 grids).
    % Skips cleanly if the SegMirMaker binary is not built.

    properties
        tin
        bin
        seg    % shared segment_rx output (hex2-equivalent config)
    end

    methods (TestClassSetup)
        function make(tc)
            here = fileparts(mfilename('fullpath'));            % mmacos/tests
            res_root = fileparts(fileparts(here));              % MACOS_resources
            tc.tin = fullfile(res_root, 'segmirmaker', 'test_in');
            tc.bin = fullfile(res_root, 'segmirmaker', ...
                              'build_release_ifx', 'SegMirMaker');
            tc.assumeTrue(isfile(tc.bin), ...
                'SegMirMaker binary not built (source ./makesegmirmaker.sh)');
            tc.seg = macos.design.segment_rx( ...
                fullfile(tc.tin, 'e5mono.in'), 'elt', 1, ...
                'rings', 2, 'grid', 'Hex', 'gap', 50, 'dofs', 6, ...
                'meas_config', 2);
        end
    end

    methods (Test)
        function test_structure(tc)
            s = tc.seg;
            tc.verifyEqual(s.nseg, 19);
            tc.verifyEqual(s.seg_elts, 1:19);
            tc.verifyEqual(s.n_elt, 25);
            txt = fileread(s.in);
            tc.verifyEqual(count(txt, 'Element=  Segment'), 19);
            tc.verifySubstring(txt, 'nElt=  25');
            % Segment tiling grid replaces the source ray grid and the
            % nSeg block is present.
            tc.verifySubstring(txt, 'GridType=  Hex');
            tc.verifySubstring(txt, 'nSeg=  19');
            tc.verifySubstring(txt, 'SegCoord=');
            % Downstream train renumbered by +18, order preserved.
            lines = readlines(s.in);
            starts = find(startsWith(strtrim(lines), "iElt="));
            tc.verifyNumElements(starts, 25);
            getname = @(k) strtrim(extractAfter( ...
                lines(find(startsWith(strtrim(lines(starts(k):end)), ...
                "EltName=") , 1) + starts(k) - 1), "EltName="));
            tc.verifyEqual(getname(20), "m2");
            tc.verifyEqual(getname(21), "lens_s1");
            tc.verifyEqual(getname(25), "");   % FocalPlane has empty name
            iElt20 = sscanf(char(lines(starts(20))), ' iElt= %d');
            tc.verifyEqual(iElt20, 20);
        end

        function test_loads_and_traces_like_parent(tc)
            s = tc.seg;
            old = cd(s.run.workdir);           % GridFile= resolves from cwd
            restore = onCleanup(@() cd(old));
            macos.init(512);
            % Parent baseline (copy already staged in the workdir).
            macos.load_rx(fullfile(s.run.workdir, 'e5mono.in'));
            p = macos.trace();
            % Segmented system.
            macos.load_rx(s.in);
            tc.verifyEqual(macos.num_elt(), 25);
            t = macos.trace();
            tc.verifyGreaterThan(t.nRays, 5000);
            rs = macos.get_ray_status(t.nRays);
            tc.verifyEqual(sum(rs.status ~= 0), 0, ...
                'segmented trace lost rays the parent did not');
            % Same wavefront, different sampling (hex segment tiling vs
            % circular): rms parity within 15% (measured 3.564e-5 vs
            % 3.920e-5 mm on the reference).
            tc.verifyLessThan(abs(t.rmsWFE - p.rmsWFE)/p.rmsWFE, 0.15);
        end

        function test_frames(tc)
            f = tc.seg.frames;
            tc.verifyNumElements(f, 19);
            tc.verifyEqual(f(1).name, "Seg1");
            % Seg1 sits at the parent vertex (to the segment-center
            % surface-solve tolerance; measured 1.9e-6 mm = 1.9 nm on
            % the 8-m parent).
            tc.verifyLessThan(norm(f(1).rpt), 1e-4);
            for k = 1:19
                T = [f(k).xhat f(k).yhat f(k).zhat];
                tc.verifyLessThan(max(abs(T'*T - eye(3)), [], 'all'), 1e-12, ...
                    sprintf('Seg%d triad not orthonormal', k));
                tc.verifyGreaterThan(dot(cross(f(k).xhat, f(k).yhat), ...
                    f(k).zhat), 0.999999, ...
                    sprintf('Seg%d triad not right-handed', k));
                tc.verifyGreaterThan(f(k).lmon, 0);
            end
            % Ring-1 centers are equidistant IN THE TILING PLANE (the
            % 3-D norms differ at ~2e-6 relative because the e5 parent
            % is a FreeForm with mm-scale astigmatism — azimuth-
            % dependent sag is real figure, not solve noise).  Project
            % out the vertex normal (Seg1's zhat = parent psi).
            z1 = f(1).zhat;
            rho = @(k) norm(f(k).rpt - dot(f(k).rpt, z1)*z1);
            r = arrayfun(rho, 2:7);
            tc.verifyLessThan(max(abs(r - mean(r)))/mean(r), 1e-7);
        end

        function test_emit_apertures_and_rxpoly(tc)
            % segment_rx emit_apertures declares each segment's PHYSICAL
            % boundary as a polygonal aperture (pie: center HEXAGON +
            % chorded sectors with inner-sector obscurations), and
            % seg_boundary auto-switches to its 'rxpoly' source so
            % launcher placement uses the Rx-declared edges.
            s = macos.design.segment_rx(fullfile(tc.tin, 'e5mono.in'), ...
                'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'dofs', 6, 'meas_config', 1, 'emit_apertures', true);
            tc.verifyEqual(s.nseg, 7);
            txt = fileread(s.in);
            tc.verifyEqual(count(txt, 'ApType=  Polygonal'), 7);
            tc.verifyEqual(count(txt, 'PolyApVec='), 7);
            tc.verifyEqual(count(txt, 'PolyObsVec='), 6);   % wedges only
            % center = hexagon; wedges = apex + chorded arc; the ring-1
            % obscuration is the apex TRIANGLE to the chord facing the
            % center hexagon's flat (straight physical edge, not an arc)
            tc.verifyEqual(size(s.apertures.poly{1}, 2), 6);
            tc.verifyEqual(size(s.apertures.poly{2}, 2), 14);
            tc.verifyEqual(size(s.apertures.obs{2}, 2), 3);
            % rxpoly reader: auto-detected; the center hexagon must
            % round-trip vertex-for-vertex (polyshape may re-order)
            B = macos.design.seg_boundary(s);
            tc.verifyEqual(B.kind, 'rxpoly');
            P = B.poly{1}(:, 1:end-1);
            Q = s.apertures.poly{1};
            tc.verifyEqual(size(P, 2), 6);
            for q = 1:6
                tc.verifyLessThan(min(vecnorm(P - Q(:,q))), 1e-6, ...
                    sprintf('center hex vertex %d does not round-trip', q));
            end
            % the wedge boundary is the aperture MINUS its obscuration:
            % no boundary point may sit inside the chord (perpendicular
            % distance (width+gap)/2 from the tiling center), and EVERY
            % segment tile must survive as a full-size region (the
            % subtract used to leave numeric slivers as boundary #1)
            W2 = [B.u.'; B.v.'] * (B.poly{2} - B.c0);
            d_chord = (s.width + s.gap)/2;
            tc.verifyGreaterThan(min(vecnorm(W2)), 0.95*d_chord, ...
                'wedge rxpoly boundary must exclude the obscured apex region');
            for s2 = 1:s.nseg
                tc.verifyGreaterThan(size(B.poly{s2}, 2), 5, ...
                    sprintf('segment %d rxpoly boundary collapsed', s2));
                span = max(B.poly{s2}, [], 2) - min(B.poly{s2}, [], 2);
                tc.verifyGreaterThan(max(span), s.width, ...
                    sprintf('segment %d rxpoly tile is a sliver', s2));
            end
            % source can be forced; rxpoly on an aperture-less Rx refuses
            Bt = macos.design.seg_boundary(s, 0, 'source', 'tiling');
            tc.verifyEqual(Bt.kind, 'pie');
            tc.verifyError(@() macos.design.seg_boundary(tc.seg, 0, ...
                'source', 'rxpoly'), 'macos:design:seg_boundary:rxpoly');
            % engine parity: the variant loads + traces; ONLY the
            % source-tiling gap rays clip (physical honesty), the
            % vast majority of rays survive
            old = cd(s.run.workdir);
            restore = onCleanup(@() cd(old));
            macos.init(512);
            macos.load_rx(s.in);
            t = macos.trace();
            ri_ = macos.get_ray_info(t.nRays);
            frac = nnz(ri_.ok_pass) / t.nRays;
            tc.verifyGreaterThan(frac, 0.90, 'apertures clipped far more than the gaps');
            tc.verifyLessThan(frac, 1.0, 'physical apertures must clip the gap rays');
            % elt_info polygon read-back: the engine's projected
            % PolyApVtx must reproduce the emitted center hexagon IN
            % the element's aperture plane (the out-of-plane component
            % is projected away by SetCvxPolyApVtx)
            i1 = macos.get_elt_info(1);
            tc.verifyEqual(i1.ap_type, 7);
            ps = mmacos('elt_psi', 1, zeros(3,1), 0, 1);  ps = ps/norm(ps);
            vp = mmacos('elt_vpt', 1, zeros(3,1), 0, 1);
            xa = i1.x_obs - dot(i1.x_obs, ps)*ps;  xa = xa/norm(xa);
            ya = cross(ps, xa);
            G2 = i1.ap_vec(1:2) + i1.poly;
            T2 = [xa.'; ya.'] * (s.apertures.poly{1} - vp);
            for q = 1:6
                tc.verifyLessThan(min(vecnorm(G2 - T2(:,q))), 1e-6, ...
                    sprintf('elt_info hexagon vertex %d off in-plane', q));
            end
        end

        function test_seg_apertures_hex(tc)
            % hex tiles: exact hex corners, no obscurations (pure
            % geometry -- no engine, runs on the shared 2-ring seg)
            ap = macos.design.seg_apertures(tc.seg);
            tc.verifyEqual(ap.kind, 'hex');
            tc.verifyNumElements(ap.blocks, 19);
            for k = [1 7 19]
                tc.verifyEqual(size(ap.poly{k}, 2), 6);
                tc.verifyEmpty(ap.obs{k});
                tc.verifyTrue(any(contains(ap.blocks{k}, "ApType=  Polygonal")));
                tc.verifyTrue(any(contains(ap.blocks{k}, "xObs=")));
            end
            % apothem = width/2: every corner sits at width/2/cos(30)
            % from its segment center, in the tiling plane
            T = macos.design.hex_tile(tc.seg);
            prj = @(P) [T.u.'; T.v.'] * (P - T.c0);
            for k = [1 7 19]
                r = vecnorm(prj(ap.poly{k}) - prj(tc.seg.frames(k).rpt));
                tc.verifyLessThan(max(abs(r - (tc.seg.width/2)/cos(pi/6))), ...
                    1e-6, sprintf('hex corner radius off on segment %d', k));
            end
        end

        function test_apstop_handling_and_refusals(tc)
            % ApStop's element form is a 2-vector offset carried INSIDE
            % the stop element's own block (StopElt = that element by
            % read order; header form is a 3-vector position).  Neither
            % renumbers.  Synthetic parent: element-form ApStop on m1
            % (the segmented element -> dropped with the block, warn)
            % AND on m2 (downstream -> must survive in m2's block).
            wd = tempname; mkdir(wd);
            copyfile(fullfile(tc.tin, 'flat.txt'), fullfile(wd, 'flat.txt'));
            copyfile(fullfile(tc.tin, 'macos_param.txt'), ...
                     fullfile(wd, 'macos_param.txt'));
            lines = readlines(fullfile(tc.tin, 'e5mono.in'));
            apline = "           ApStop=  0.0E+00  0.0E+00";
            for nm = ["m1" "m2"]
                i = find(strtrim(lines) == "EltName=  " + nm, 1);
                if isempty(i)
                    i = find(contains(lines, "EltName=  " + nm), 1);
                end
                lines = [lines(1:i); apline; lines(i+1:end)]; %#ok<AGROW>
            end
            pin = fullfile(wd, 'e5mono_apstop.in');
            writelines(lines, pin);
            s = tc.verifyWarning(@() macos.design.segment_rx(pin, ...
                'elt', 1, 'rings', 1, 'grid', 'Pie', 'gap', 50, ...
                'dofs', 6), 'macos:design:segment_rx:apstop');
            tc.verifyEqual(s.nseg, 7);
            tc.verifyTrue(s.dropped_apstop);
            % Exactly one ApStop line survives -- m2's, untouched.
            merged = readlines(s.in);
            ia = find(strtrim(merged) == strtrim(apline));
            tc.verifyNumElements(ia, 1);
            im2 = find(contains(merged, "EltName=  m2"), 1);
            tc.verifyEqual(ia, im2 + 1);
            % Index keywords we cannot renumber yet must refuse loudly
            % (before paying for the SegMirMaker run).
            iap = find(startsWith(strtrim(lines), "Obscratn="), 1);
            lines2 = [lines(1:iap); "        OptTgtElt=  3"; lines(iap+1:end)];
            pin2 = fullfile(wd, 'e5mono_opttgt.in');
            writelines(lines2, pin2);
            tc.verifyError(@() macos.design.segment_rx(pin2, 'elt', 1), ...
                'macos:design:segment_rx:renumber');
        end

        function test_state_consistency_rays_on_frames(tc)
            % THE segmentation invariant (Dave 2026-07-18): moving
            % segment k's DOFs must move segment k's outputs -- so the
            % rays assigned to element k must physically lie ON the
            % emitted frame k (which drives apertures, launchers, edge
            % sensors).  Gate = engine-truth ray positions (ray_hist)
            % vs frames, BOTH grids, on the e5 back-facing parent --
            % the case where SegMirMaker's 180-deg in-plane basis flip
            % used to desynchronize the emitted SegXgrid from the
            % frames (every back-facing fixture was point-reflected).
            for cfg = {{'Pie', 1}, {'Hex', 2}}
                s = macos.design.segment_rx( ...
                    fullfile(tc.tin, 'e5mono.in'), 'elt', 1, ...
                    'rings', cfg{1}{2}, 'grid', cfg{1}{1}, ...
                    'gap', 50, 'dofs', 6);
                old = cd(s.run.workdir); c_ = onCleanup(@() cd(old));
                macos.init(512);
                macos.load_rx(s.in);
                macos.ray_hist('on');
                t = macos.trace();
                h = macos.ray_hist(t.nRays);
                for k = 2:s.nseg          % off-center segments = the gate
                    m = squeeze(h.ok(:, k+1));
                    tc.assertGreaterThan(nnz(m), 0, ...
                        sprintf('%s Seg%d received no rays', cfg{1}{1}, k));
                    P = squeeze(h.P(:, m, k+1));
                    ctr = mean(P(1:2,:), 2);
                    tc.verifyLessThan( ...
                        norm(ctr - s.frames(k).rpt(1:2)), 0.35*s.width, ...
                        sprintf(['%s Seg%d rays centered at (%.0f,%.0f), ' ...
                                 'frame at (%.0f,%.0f) -- tiling/frame ' ...
                                 'mismatch'], cfg{1}{1}, k, ctr, ...
                                 s.frames(k).rpt(1:2)));
                end
                macos.ray_hist('off');
                clear c_
            end
        end

        function test_carry_parent_zernike_figure(tc)
            % A design-layer parent (Surface=Zernike, solved figure)
            % must hand its figure to EVERY segment via the FF channel
            % -- SegMirMaker replicates FF-channel figures (e5) but
            % silently drops the Zern channel (e2e s3, 2026-07-18).
            % Gate: figure text on all segments + trace parity of the
            % figured wavefront.
            t = macos.design.Telescope('family','Cassegrain', ...
                'aperture_diameter_m',1.0, 'system_fnum',8.0, ...
                'primary_fnum',2.0, 'BFD_m',0.25, ...
                'model_size',512, 'grid_npts',21);
            t.build();
            t.set_freeform(1, 5, 2e-7, 'type','BornWolf', 'lmon', 0.5);
            t.build('', 'init', false);
            s0 = macos.trace();
            rms0 = s0.rmsWFE;
            tc.verifyGreaterThan(rms0, 1e-8, 'figure not expressed on parent');
            wd = tempname; mkdir(wd);          % short basename: SegMirMaker
            f = fullfile(wd, 'zfig.in');       % truncates long file names
            t.save(f);
            s = macos.design.segment_rx(f, 'elt', 1, 'rings', 1, ...
                'grid', 'Hex', 'gap', 0.005, 'dofs', 6);
            merged = strtrim(readlines(s.in));
            % each segment: SegMirMaker's zeroed FF line + the carried
            % parent figure appended LAST (parser last-wins)
            tc.verifyEqual(nnz(startsWith(merged, "FFZernCoef=") & ...
                               ~contains(merged, "0d0")), s.nseg, ...
                'parent figure not carried onto every segment');
            tc.verifyEqual(nnz(startsWith(merged, "lFF=") & ...
                               ~contains(merged, "0.0E+00")), s.nseg);
            macos.init(512);
            macos.load_rx(s.in);
            s1 = macos.trace();
            tc.verifyEqual(s1.rmsWFE, rms0, 'RelTol', 0.25, ...
                'segmented wavefront does not reproduce the figured parent');
        end

        function test_carry_obs_center_segment(tc)
            % A central-hole obscuration on the segmented element (the
            % Telescope set_hole emission) must ride onto the CENTER
            % segment, where it physically lives: appended LAST in
            % Seg1's block (parser last-wins) and clipping rays after
            % the splice (a bare segmentation otherwise loses none).
            wd = tempname; mkdir(wd);
            copyfile(fullfile(tc.tin, 'flat.txt'), fullfile(wd, 'flat.txt'));
            copyfile(fullfile(tc.tin, 'macos_param.txt'), ...
                     fullfile(wd, 'macos_param.txt'));
            lines = readlines(fullfile(tc.tin, 'e5mono.in'));
            i1 = find(contains(lines, "EltName=  m1"), 1);
            grp = ["             nObs=  1"; ...
                   "          ObsType=  Circle"; ...
                   "           ObsVec=  5.0E+02  0.0E+00  0.0E+00"];
            lines = [lines(1:i1); grp; lines(i1+1:end)];
            pin = fullfile(wd, 'e5mono_hole.in');
            writelines(lines, pin);
            s = macos.design.segment_rx(pin, 'elt', 1, 'rings', 1, ...
                'grid', 'Hex', 'gap', 50, 'dofs', 6);
            tc.verifyEqual(s.carried_obs, 1);
            merged = strtrim(readlines(s.in));
            ic = find(merged == "ObsType=  Circle");
            tc.verifyNumElements(ic, 1);
            ie = find(startsWith(merged, "iElt="), 2);
            tc.verifyTrue(ic > ie(1) && ic < ie(2), ...
                'carried obscuration not inside Seg1''s block');
            old = cd(s.run.workdir); c_ = onCleanup(@() cd(old));
            macos.init(512);
            macos.load_rx(s.in);
            st = macos.trace();
            rs = macos.get_ray_status(st.nRays);
            tc.verifyGreaterThan(nnz(rs.status ~= 0), 0, ...
                'carried hole did not clip any center-segment rays');
        end
    end
end
