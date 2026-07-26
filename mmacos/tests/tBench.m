classdef tBench < matlab.unittest.TestCase
%TBENCH  macos.design.Bench -- sequential add-optic bench builder.
%   Builds a miniature folded bench exercising every add_* primitive
%   (baffle, collimating + focusing singlets, BS reflect, retro mirror,
%   BS transmit with Snell walk-off, fold, reference, detector), emits
%   the .in, and verifies against the engine: element count, zero
%   vignetting, and the builder's analytic chief ray matching the
%   engine's traced chief at EVERY element (this is what validates the
%   mirror-turn and plate-refraction geometry, including walk-off).

    properties (Constant)
        ModelSize = 128
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
        end
    end

    methods (Test)
        function test_refract_statics(testCase)
            % plate refraction: exit parallel to entry, bent inside
            d  = macos.design.Bench.unit([1; 0.2; 0]);
            nh = macos.design.Bench.unit([-1; -1; 0]);
            d1 = macos.design.Bench.refract(d, nh, 1.0, 1.5);
            d2 = macos.design.Bench.refract(d1, nh, 1.5, 1.0);
            testCase.verifyLessThan(dot(d1, d), 1 - 1e-6);      % bent
            testCase.verifyEqual(dot(d2, d), 1, 'AbsTol', 1e-12); % parallel out
        end

        function test_bench_end_to_end(testCase)
            b = macos.design.Bench('tbench', ...
                'aperture', 2*atan(10/200)*0.9, 'ngridpts', 21);
            b.add_baffle(200, 10);
            L1 = b.add_lens(200, 400, 30, 'mode','collimate', 'name','L1');
            [~, bs] = b.add_bs_reflect(100, [0;-1;0], 'thickness', 6);
            iDM = b.add_mirror(150, 'name','DM', 'aprad', 20);
            iTx = b.add_bs_transmit(bs);
            b.add_fold(100, [1;0;0], 'name','Fold1');
            L2 = b.add_lens(100, 200, 30, 'mode','focus', 'name','L2');
            b.add_reference(200 - L2.thickness, 'FocalMask');
            iDet = b.add_detector(50, 'Detector');

            % builder-side: plate walk-off is real (back-face vertex off
            % the unrefracted straight line through the front face)
            straight = b.E(iTx(1)).vpt + [0;1;0]*(b.E(iTx(2)).s - b.E(iTx(1)).s);
            testCase.verifyGreaterThan(norm(b.E(iTx(2)).vpt - straight), 0.1);

            rx = fullfile(tempname); mkdir(rx);
            rxf = fullfile(rx, 'tbench.in');
            b.emit(rxf);
            macos.load_rx(rxf);
            nE = macos.num_elt();
            testCase.verifyEqual(nE, numel(b.E));

            % no vignetting: the 0.9-fill cone clears the baffle
            s1 = macos.trace(1);
            sN = macos.trace(nE);
            testCase.verifyGreaterThan(s1.nRays, 100);
            testCase.verifyEqual(sN.nRays, s1.nRays);

            % engine chief ray must cross every emitted vertex
            for k = 1:nE
                sk = macos.trace(k);
                info = macos.get_ray_info(sk.nRays);
                testCase.verifyLessThan(norm(info.pos(:,1) - b.E(k).vpt), 1e-6, ...
                    sprintf('chief mismatch at elt %d (%s)', k, b.E(k).name));
            end

            % sanity on the optics: collimator/imager conic signs, and the
            % focusing leg actually converges (spot at mask << beam at DM)
            testCase.verifyGreaterThan(L1.Kr, 0);
            testCase.verifyLessThan(L2.Kr, 0);
            macos.trace(iDM);
            iDM_info = macos.get_ray_info(macos.trace(iDM).nRays);
            rDM = max(vecnorm(iDM_info.pos - iDM_info.pos(:,1)));
            sM = macos.trace(nE - 1);   % FocalMask
            m_info = macos.get_ray_info(sM.nRays);
            rM = max(vecnorm(m_info.pos - m_info.pos(:,1)));
            testCase.verifyLessThan(rM, 0.2*rDM);
            macos.trace(iDet);   % leave state clean
        end

        function test_oap_relay(testCase)
            % source -> collimating OAP -> fold -> focusing OAP -> detector.
            % The chief-agreement loop is what validates the off-axis-
            % section emission (RptElt=pole on the parent figure): a wrong
            % Kr/psi pairing moves the parent surface off the pole and the
            % chief misses it.  Collimation/spot tolerances are LOOSE
            % because zSource offsets the effective point source ~25 mm
            % from ChfRayPos (absorbed by optimization in real use).
            b = macos.design.Bench('toap', ...
                'aperture', 2*atan(8/200)*0.9, 'ngridpts', 21);
            b.add_baffle(200, 8);
            b.add_oap(100, [0;1;0], 'mode','collimate', 'focus_dist',300, ...
                      'name','OAP1');
            iF = b.add_fold(200, [1;0;0], 'name','Fold1');
            b.add_oap(200, [0;-1;0], 'mode','focus', 'focus_dist',250, ...
                      'name','OAP2');
            iDet = b.add_detector(250, 'Detector');

            rx = fullfile(tempname); mkdir(rx);
            rxf = fullfile(rx, 'toap.in');
            b.emit(rxf);
            macos.load_rx(rxf);
            nE = macos.num_elt();
            testCase.verifyEqual(nE, numel(b.E));
            s1 = macos.trace(1);
            sN = macos.trace(nE);
            testCase.verifyEqual(sN.nRays, s1.nRays);
            for k = 1:nE
                % chief crosses the POLE (rpt) -- equals vpt except for
                % off-axis sections, where vpt is the parent vertex
                sk = macos.trace(k);
                info = macos.get_ray_info(sk.nRays);
                testCase.verifyLessThan(norm(info.pos(:,1) - b.E(k).rpt), 1e-6, ...
                    sprintf('chief mismatch at elt %d (%s)', k, b.E(k).name));
            end
            % beam roughly collimated after OAP1 (sign-flip failure ~100x this)
            sF = macos.trace(iF);
            info = macos.get_ray_info(sF.nRays);
            ok = info.ok_trace(:) & info.ok_pass(:);
            D = info.dir(:,ok); D = D ./ vecnorm(D);
            dch = info.dir(:,1)/norm(info.dir(:,1));
            spread = sqrt(mean(acos(max(min(dch.'*D,1),-1)).^2));
            testCase.verifyLessThan(spread, 0.02);
            % beam converges to a compact spot at the OAP2 focus
            sD = macos.trace(iDet);
            info = macos.get_ray_info(sD.nRays);
            ok = info.ok_trace(:) & info.ok_pass(:);
            d = info.pos(:,ok) - info.pos(:,1);
            testCase.verifyLessThan(max(sqrt(sum(d.^2,1))), 3.0);
        end

        function test_offner_relay(testCase)
            % source -> baffle -> Offner 3-mirror concentric relay ->
            % image Reference -> detector.  Chief agreement validates the
            % exact-sphere construction and the Kr<0 / psi-concave
            % emission on all three mirrors (M2 worked on its convex
            % side).  Spot tolerance is loose (zSource offsets the true
            % source ~25 mm; the 1:1 relay images the true point, so the
            % nominal image plane sees a small defocus blur).
            b = macos.design.Bench('trelay', ...
                'aperture', 2*atan(5/300)*0.9, 'ngridpts', 21);
            b.add_baffle(300, 5);
            Rl = b.add_relay(100, 'focus_dist', 400, 'R', 400, ...
                             'ring_offset', 40, 'side', 1, 'name','Off');
            iImg = b.add_reference(Rl.image_dist, 'Image');
            iDet = b.add_detector(50, 'Detector');

            rx = fullfile(tempname); mkdir(rx);
            rxf = fullfile(rx, 'trelay.in');
            b.emit(rxf);
            macos.load_rx(rxf);
            nE = macos.num_elt();
            testCase.verifyEqual(nE, numel(b.E));
            s1 = macos.trace(1);
            sN = macos.trace(nE);
            testCase.verifyEqual(sN.nRays, s1.nRays);
            for k = 1:nE
                sk = macos.trace(k);
                info = macos.get_ray_info(sk.nRays);
                testCase.verifyLessThan(norm(info.pos(:,1) - b.E(k).rpt), 1e-6, ...
                    sprintf('chief mismatch at elt %d (%s)', k, b.E(k).name));
            end
            % relay actually reconverges: spot at the image plane is small
            % vs the M1 beam footprint
            sM = macos.trace(Rl.i(1));
            info = macos.get_ray_info(sM.nRays);
            ok = info.ok_trace(:) & info.ok_pass(:);
            dM = info.pos(:,ok) - info.pos(:,1);
            rM1 = max(sqrt(sum(dM.^2,1)));
            sI = macos.trace(iImg);
            info = macos.get_ray_info(sI.nRays);
            ok = info.ok_trace(:) & info.ok_pass(:);
            dI = info.pos(:,ok) - info.pos(:,1);
            rImg = max(sqrt(sum(dI.^2,1)));
            testCase.verifyLessThan(rImg, 0.2*rM1);
            macos.trace(iDet);
        end

        function test_tail_arches(testCase)
            % l2_trade detector-leg architectures (twyman_green
            % 'tail_arch'): each builds, emits, traces with zero ray loss,
            % and the engine chief crosses every emitted vertex.  Params
            % are the optimized values from l2_trade/TRADE_NOTE.md.
            archs = { ...
                {'tail_arch','fieldlens', 'FL_F',25.02100857, ...
                 'FL_Kc',-2.11278288, 'D_MASK_FL',6.277463741, ...
                 'DET_TRIM',1.085330067}, ...
                {'tail_arch','doublet', 'MASK_TRIM',1.614619633, ...
                 'L2A_Kc',-3.575374653, 'L2B_Kc',2.328903027, ...
                 'DET_TRIM',2.97066401}};
            for a = 1:numel(archs)
                G = macos.design.twyman_green('ngridpts',21, archs{a}{:});
                rx = fullfile(tempname); mkdir(rx);
                rxf = fullfile(rx, 'tg_arm.in');
                G.bt.emit(rxf);
                macos.load_rx(rxf);
                nE = macos.num_elt();
                testCase.verifyEqual(nE, numel(G.bt.E));
                s1 = macos.trace(1);
                sN = macos.trace(nE);
                testCase.verifyEqual(sN.nRays, s1.nRays, ...
                    sprintf('%s: ray loss through the tail', archs{a}{2}));
                for k = 1:nE
                    sk = macos.trace(k);
                    info = macos.get_ray_info(sk.nRays);
                    testCase.verifyLessThan( ...
                        norm(info.pos(:,1) - G.bt.E(k).vpt), 1e-6, ...
                        sprintf('%s: chief mismatch at elt %d (%s)', ...
                        archs{a}{2}, k, G.bt.E(k).name));
                end
            end
        end
    end
end
