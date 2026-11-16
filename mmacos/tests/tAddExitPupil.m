classdef tAddExitPupil < matlab.unittest.TestCase
%TADDEXITPUPIL  macos.design.add_exit_pupil -- the exit-pupil pair on an imported deck.
%
%   The fixture is the zmx2macos conversion of the synthetic f/6 Cassegrain
%   (rx_converter/zemax/examples/cassegrain_example.zmx): primary R -1000 mm K -1
%   (the stop), secondary 400 mm in front with R -240 mm K -1.96, image 600 mm
%   behind the secondary.  Its exit pupil is the stop's image in the secondary,
%   a textbook number: 1/s' = 2/R2 - 1/s with s = 400 -> s' = -92.3077 mm,
%   so the pupil vertex sits at z = -492.3077 mm and the reference sphere
%   radius is 692.3077 mm.  The gate is written from that, never from the
%   engine.  Then: the pair costs no ray, the on-axis OPD at the pupil is
%   null, dw_dx reads at it (the chain entry), and the three refusals.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_Zemax.in'
        R1 = -1000;  D12 = 400;  R2 = -240;  ZFP = 200;   % mm, the deck
    end
    properties
        rx_in
        tmpdir
    end

    methods (TestClassSetup)
        function setup(tc)
            tc.rx_in  = rx_fixture_path(tc.RxName);
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);
            macos.init(tc.ModelSize);
        end
    end
    methods (TestClassTeardown)
        function clean(tc)
            if ~isempty(tc.tmpdir) && isfolder(tc.tmpdir), rmdir(tc.tmpdir, 's'); end
        end
    end

    methods
        function [z_ep, radius] = paraxial_(tc)
            s  = tc.D12;                              % the stop (primary) seen from the secondary
            sp = 1 / (2/tc.R2 - 1/s);                 % mirror equation, its image
            z_ep   = -tc.D12 + sp;                    % behind the secondary (virtual)
            radius = tc.ZFP - z_ep;
        end
        function info = run_(tc, name)
            info = macos.design.add_exit_pupil(tc.rx_in, fullfile(tc.tmpdir, name), ...
                'model_size', tc.ModelSize, 'init', false, 'verbose', false);
        end
    end

    methods (Test)
        function test_pair_inserted_before_the_focal_plane(tc)
            info = tc.run_('xp.in');
            tc.verifyEqual(info.n_in, 3);
            tc.verifyEqual(info.n_out, 5);
            tc.verifyEqual([info.img_elt info.ep_elt info.fp_elt], [3 4 5]);
            macos.load_rx(info.rx_out);
            tc.verifyEqual(macos.num_elt(), 5);
            i3 = macos.get_elt_info(3);  i4 = macos.get_elt_info(4);  i5 = macos.get_elt_info(5);
            tc.verifyTrue(any(i3.elt_id == [3 8]), 'FP_return is not a Return');
            tc.verifyTrue(any(i4.elt_id == [3 8]), 'ExitPupil is not a Return');
            tc.verifyEqual(i5.type, 'FocalPlane');
            tc.verifyGreaterThan(abs(macos.get_elt_kr(3)), 1e21, 'FP_return must be flat');
            tc.verifyLessThan(abs(macos.get_elt_kr(4)), 1e21, 'ExitPupil must be curved (dw_dx read criterion)');
            tc.verifyEqual(macos.get_elt_kr(4), -info.ep_radius, 'RelTol', 1e-12);
        end

        function test_pupil_is_the_stop_image_in_the_secondary(tc)
            % Textbook: the exit pupil of a Cassegrain with the stop at the primary is
            % the primary's image formed by the convex secondary.
            [z_ep, radius] = tc.paraxial_();
            info = tc.run_('xp2.in');
            % FEX on the deck's own chief ray: exact to 1e-9 mm (measured 2026-11-04;
            % a 1-arcmin probe field moves the radius by 5e-4 mm and tilts psi)
            tc.verifyEqual(info.ep_vpt(3), z_ep,     'AbsTol', 1e-6);
            tc.verifyEqual(info.ep_radius, radius,   'AbsTol', 1e-6);
            tc.verifyLessThan(hypot(info.ep_vpt(1), info.ep_vpt(2)), 1e-9, 'EP off the axis');
            tc.verifyEqual(info.ep_psi(:), [0;0;1], 'AbsTol', 1e-9, 'psi must point at the image');
            fprintf('  EP z %.4f (paraxial %.4f, diff %.2e); radius %.4f (paraxial %.4f, diff %.2e)\n', ...
                info.ep_vpt(3), z_ep, info.ep_vpt(3)-z_ep, info.ep_radius, radius, info.ep_radius-radius);
            % must-fail: the seed radius (secondary vertex to image, 600 mm) is 13 % off
            tc.verifyGreaterThan(abs(info.ep_radius - (tc.ZFP + tc.D12)) / radius, 0.1);
        end

        function test_no_ray_lost_and_null_opd_at_the_pupil(tc)
            info = tc.run_('xp3.in');
            tc.verifyEqual(info.nrays_out, info.nrays_in);
            tc.verifyGreaterThan(info.nrays_in, 3000);
            tc.verifyLessThan(info.rms_ep, 1e-6, 'on-axis Cassegrain OPD at its pupil (mm)');
        end

        function test_dw_dx_reads_at_the_new_pupil(tc)
            % the chain entry: dw_dx refuses the bare deck and accepts the output
            m = macos.Session(tc.ModelSize);
            tc.verifyError(@() macos.dw_dx(m, tc.rx_in, 'elts', 2, 'dofs', 0, 'verbose', false), ...
                'macos:dw_dx:noPupil');
            info = tc.run_('xp4.in');
            out = macos.dw_dx(m, info.rx_out, 'elts', 2, 'dofs', 0, 'verbose', false);
            J = out.dwdx;
            tc.verifyEqual(size(J, 2), 1);
            tc.verifyTrue(all(isfinite(J(:))));
            tc.verifyGreaterThan(norm(J), 0, 'a secondary tilt must move the pupil OPD');
        end

        function test_refusals(tc)
            txt = fileread(tc.rx_in);
            % (a) terminal element not a FocalPlane
            f = fullfile(tc.tmpdir, 'notfp.in');
            t = regexprep(txt, '(?m)^(\s*Element=\s*)FocalPlane', '$1Reference', 'once');
            fid = fopen(f, 'w'); fprintf(fid, '%s', t); fclose(fid);
            tc.verifyError(@() macos.design.add_exit_pupil(f, fullfile(tc.tmpdir,'o1.in'), ...
                'init', false, 'verbose', false), 'macos:design:add_exit_pupil:notFP');
            % (b) already carries a pupil
            info = tc.run_('xp5.in');
            tc.verifyError(@() macos.design.add_exit_pupil(info.rx_out, fullfile(tc.tmpdir,'o2.in'), ...
                'init', false, 'verbose', false), 'macos:design:add_exit_pupil:hasPupil');
            % (c) no stop anywhere
            f = fullfile(tc.tmpdir, 'nostop.in');
            t = regexprep(txt, '(?m)^\s*ApStop=[^\n]*\n', '', 'once');
            tc.assertTrue(isempty(regexp(t, '(?m)^\s*ApStop=', 'once')));   % the head comment still SAYS ApStop=
            fid = fopen(f, 'w'); fprintf(fid, '%s', t); fclose(fid);
            tc.verifyError(@() macos.design.add_exit_pupil(f, fullfile(tc.tmpdir,'o3.in'), ...
                'init', false, 'verbose', false), 'macos:design:add_exit_pupil:noStop');
            % ... and 'stop_elt' rescues it
            info = macos.design.add_exit_pupil(f, fullfile(tc.tmpdir,'o4.in'), ...
                'stop_elt', 1, 'init', false, 'verbose', false);
            tc.verifyEqual(info.n_out, 5);
        end
    end
end
