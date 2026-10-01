classdef tGratingOpl < matlab.unittest.TestCase
%TGRATINGOPL  A grating's optical-path jump must agree with its ray directions.
%   Eikonal consistency: the wavefront is the surface orthogonal to the rays,
%   so wherever the engine's rays converge to a point, the OPD it reports on
%   a reference sphere centred on that point must be flat.  The spectrometer
%   Offner seed (spectrometer_geom, concave R used twice + convex grating at
%   the stop, slit on the ring) is that case: at order -1 the engine's rays
%   land within 0.1 um of the chief at the FPA (tSpectrometerRx pins them to
%   the exact chain at 1e-9 m), so the pupil OPD on a reference sphere about
%   the chief's focus must be < lambda/50 rms.
%
%   Found 2026-10-01 building the propagation twin: at order 0 the OPD is
%   8e-11 m (the terminal is right); at order -1 it is 4.05 WAVES rms with a
%   tilt + coma-like residual, while the rays are perfect.  Mechanism, read
%   in elemsub.F Snells_Law_Grating: the OPL jump is
%   dL = (nb r - na i) . rho_prj with rho_prj the hit vector projected into
%   the LOCAL tangent plane, = (m lambda/d) (s0 . rho_prj); the groove-count
%   phase of equidistant groove PLANES is (m lambda/d) (s0 . rho) with rho
%   the hit vector from the vertex along the FIXED ruling direction s0.  The
%   two differ by (m lambda/d)(rho.N)(s0.N) ~ (m lambda/d) rho^3/(2 R^2): a
%   cubic, 12 waves at this grating's 45 mm footprint and R = 250 mm, zero
%   on a FLAT grating (which is why the air fixtures never saw it).
%   Must-PASS leg: order 0.  Must-fail-today leg: order -1 (goes green with
%   the engine fix; no test change).  Size 128 -> SUITE_FAST.

    properties (Constant)
        Model = 128
        P = struct('Fno',2.8,'pixel_m',18e-6,'npix',[3000 500],'band_m',[380e-9 2500e-9], ...
                   'lambda_ref_m',1e-6,'order',1,'y_slit',6e-3,'offner_R',0.5, ...
                   'block_r',0.22,'glass','Silica','face_offset',0.5e-3,'Rg_factor',1)
        Lam = 1e-6
        Lref = 0.1
    end

    properties
        tmpdir
        G
    end

    methods (TestClassSetup)
        function setup(tc)
            run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'mmacos_setup.m'));
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);
            macos.init(tc.Model);
            tc.G = spectrometer_geom('offner', tc.P);
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmpdir, 'dir'), rmdir(tc.tmpdir, 's'); end
        end
    end

    methods
        function [opd_rms, ray_spread] = pupil_opd(tc, m)
            G = tc.G;
            if m == 0, G.grating.m = 0;  G.grating.d = 1; end
            M = spectrometer_rx(G, fullfile(tc.tmpdir, sprintf('offner_m%d.in', m)), ...
                                'ngridpts', 41, 'terminal', 'farfield', 'L_ref', tc.Lref);
            macos.load_rx(M.file);
            slit = tc.G.slit;  da = tc.G.aim(slit, tc.Lam);
            macos.stop(M.iG);
            macos.set_src_fov('src_pos', slit, 'src_dir', da, 'zSrc', -tc.G.src.zsrc_gap);
            macos.set_src_wvl(tc.Lam);  macos.modify();
            s = macos.trace(M.iFPA);  ri = macos.get_ray_info(s.nRays);
            ok = ri.ok_trace & ri.ok_pass;
            pc = ri.pos(:,1);  din = ri.dir(:,1);  din = din/norm(din);
            ray_spread = max(vecnorm(ri.pos(:, ok) - pc));       % transverse convergence, m
            % reference sphere centred on the chief's focus, L upstream
            macos.set_xp(pc - tc.Lref*din, din, -tc.Lref);
            macos.set_elt_vpt(M.iFPr, pc);  macos.set_elt_vpt(M.iFPA, pc);
            macos.modify();  macos.trace(M.iEP);
            W = macos.opd();  w = W(isfinite(W) & W ~= 0);
            opd_rms = std(w);
        end
    end

    methods (Test)
        function test_order_zero_relay_has_a_flat_pupil(tc)
            [opd_rms, spread] = tc.pupil_opd(0);
            tc.verifyLessThan(spread, 1e-7, 'order-0 rays converge');
            tc.verifyLessThan(opd_rms, tc.Lam/200, 'order-0 pupil OPD flat (the terminal is right)');
        end

        function test_order_minus_one_pupil_is_as_flat_as_its_rays(tc)
            [opd_rms, spread] = tc.pupil_opd(-1);
            tc.verifyLessThan(spread, 1e-7, 'order -1 rays converge (gated elsewhere at 1e-9 m vs the chain)');
            tc.verifyLessThan(opd_rms, tc.Lam/50, sprintf( ...
                ['order -1 pupil OPD must be flat where the rays converge: %.3f waves rms -- ' ...
                 'the grating OPL jump disagrees with its ray directions (groove count along the chord)'], opd_rms/tc.Lam));
        end
    end
end
