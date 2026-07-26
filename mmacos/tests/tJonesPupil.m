classdef tJonesPupil < matlab.unittest.TestCase
%TJONESPUPIL  Phase-2 Jones-pupil physics gates (PLAN_POLARIZATION 2a/2b).
%   Exercises macos.jones_pupil + macos.pol_maps against closed-form truth:
%
%   * unitarity gate     -- stock Cass mirrors carry the perfect-conductor
%                           idiom (IndRef=1, Extinc=1e22): RP=RS=-1, so the
%                           Jones pupil must be unitary-to-a-scalar at every
%                           point (D, ret, T-nonuniformity all ~ 0).
%   * basis invariance   -- D is a singular-value invariant: identical
%                           between 'double-pole' and 'local-sp'; retardance
%                           is NOT (the s/p coordinate singularity is the
%                           artifact the double-pole basis exists to kill).
%   * Fresnel gate       -- Bench 45-deg flat fold + optically-thick Al
%                           layer: per-ray RS/RP (ratio form, convention-
%                           free) and per-ray D against the analytic
%                           Fresnel coefficients, at round-off.  This gate
%                           pins BOTH coated-branch engine fixes (incident
%                           medium; |cos| in the recursion) -- it fails by
%                           ~(RP/RS)^2 if the signed-cosine bug returns.
%   * 2-theta symmetry   -- Al on both Cass mirrors: on-axis rotationally
%                           symmetric system; diattenuation orientation
%                           locks to the pupil azimuth (s/p geometry), no
%                           circular component, D grows with radius.
%   * pol_maps identity  -- synthetic diattenuator*retarder recovered
%                           exactly; ambiguity flag fires near delta=pi.
%
%   All engine-facing tests run at ModelSize 128 on Rx_Cass_FarField (via
%   rx_fixture_path) or on a Bench-emitted fold rig in a temp folder.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6      % Cass focal plane
        Prim      = 2      % Cass primary
        Sec       = 3      % Cass secondary
        nAl       = 1.45   % Al at 632.8 nm
        kAl       = 7.54
        thkAl     = 2.0e-4 % 200 nm in mm -- optically thick (skin ~13 nm)
    end

    properties
        foldRx      % Bench-emitted 45-deg fold rig (temp dir)
        foldElt
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            macos.init(testCase.ModelSize);
            % emit the fold rig once: point source, fold +x -> +z, detector
            b = macos.design.Bench('foldrig', 'aperture', 0.06, 'ngridpts', 41);
            testCase.foldElt = b.add_fold(50, [0;0;1]);
            b.add_detector(60);
            testCase.foldRx = [tempname '_dir'];
            mkdir(testCase.foldRx);
            b.emit(fullfile(testCase.foldRx, 'foldrig.in'));
        end
    end

    methods (TestMethodSetup)
        function loadRx(testCase)
            % load_rx clears coating state (verified), so tests are independent
            macos.load_rx(rx_fixture_path(testCase.RxName));
        end
    end

    methods (Test)
        % ---- unitarity gate (stock = perfect conductors) --------------
        function test_unitarity_gate(testCase)
            jp = macos.jones_pupil(testCase.Det);
            pm = macos.pol_maps(jp);
            m = jp.mask;
            testCase.verifyGreaterThan(nnz(m), 1000);
            testCase.verifyLessThan(max(pm.D(m)),   1e-12, 'diattenuation');
            testCase.verifyLessThan(max(pm.ret(m)), 1e-12, 'retardance');
            testCase.verifyLessThan(std(pm.T(m))/mean(pm.T(m)), 1e-12, ...
                'transmission uniformity');
            testCase.verifyLessThan(jp.leak, 1e-12, 'longitudinal leak');
        end

        % ---- D basis-invariant; local-sp retardance artifact ----------
        function test_basis_invariance_and_sp_artifact(testCase)
            macos.coating(testCase.Sec, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            pm_dp = macos.pol_maps(macos.jones_pupil(testCase.Det, ...
                'basis', 'double-pole'));
            pm_sp = macos.pol_maps(macos.jones_pupil(testCase.Det, ...
                'basis', 'local-sp'));
            m = pm_dp.mask & pm_sp.mask;
            % singular-value invariants agree at round-off
            testCase.verifyLessThan(max(abs(pm_dp.D(m) - pm_sp.D(m))), 1e-12);
            testCase.verifyLessThan(max(abs(pm_dp.T(m) - pm_sp.T(m))) ...
                / mean(pm_dp.T(m)), 1e-12);
            % the coordinate singularity inflates s/p retardance variation
            testCase.verifyGreaterThan(pm_sp.var_rms.ret, ...
                10*pm_dp.var_rms.ret, ...
                'local-sp retardance artifact should dominate');
        end

        % ---- Fresnel-analytic gate (fold rig) -------------------------
        function test_fold_fresnel_analytic(testCase)
            macos.load_rx(fullfile(testCase.foldRx, 'foldrig.in'));
            fold = testCase.foldElt;
            macos.coating(fold, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);

            % single trace with both s and p lit
            macos.polarization('on', 'Ex', [1/sqrt(2) 0], 'Ey', [1/sqrt(2) 0]);
            macos.trace(fold);
            rf = macos.ray_field(fold);
            m = rf.status == 0;
            testCase.verifyGreaterThan(nnz(m), 500);

            % geometry: reflect exit direction back through the flat
            kox=rf.kx(m); koy=rf.ky(m); koz=rf.kz(m);
            nx=rf.nx(m);  ny=rf.ny(m);  nz=rf.nz(m);
            kd = kox.*nx + koy.*ny + koz.*nz;
            kix = kox-2*kd.*nx;  kiy = koy-2*kd.*ny;  kiz = koz-2*kd.*nz;
            % engine s/p frames: s = ki x n; pi = s x ki; pr = s x ko
            sx=kiy.*nz-kiz.*ny; sy=kiz.*nx-kix.*nz; sz=kix.*ny-kiy.*nx;
            sm=sqrt(sx.^2+sy.^2+sz.^2); sx=sx./sm; sy=sy./sm; sz=sz./sm;
            pix=sy.*kiz-sz.*kiy; piy=sz.*kix-sx.*kiz; piz=sx.*kiy-sy.*kix;
            prx=sy.*koz-sz.*koy; pry=sz.*kox-sx.*koz; prz=sx.*koy-sy.*kox;
            % engine per-ray launch frame (ssrcray point-source seeding):
            %   yray = unitize(RayDir x xGrid); xray = yray x RayDir
            xg = [0;1;0];   % Bench-emitted xGrid (survives re-orthog.)
            yrx=kiy*xg(3)-kiz*xg(2); yry=kiz*xg(1)-kix*xg(3); yrz=kix*xg(2)-kiy*xg(1);
            ym=sqrt(yrx.^2+yry.^2+yrz.^2); yrx=yrx./ym; yry=yry./ym; yrz=yrz./ym;
            xrx=yry.*kiz-yrz.*kiy; xry=yrz.*kix-yrx.*kiz; xrz=yrx.*kiy-yry.*kix;
            einx=(xrx+yrx)/sqrt(2); einy=(xry+yry)/sqrt(2); einz=(xrz+yrz)/sqrt(2);

            Es = rf.Ex(m).*sx  + rf.Ey(m).*sy  + rf.Ez(m).*sz;
            Ep = rf.Ex(m).*prx + rf.Ey(m).*pry + rf.Ez(m).*prz;
            qs = einx.*sx  + einy.*sy  + einz.*sz;
            qp = einx.*pix + einy.*piy + einz.*piz;
            ratio_meas = (Es./Ep).*(qp./qs);

            % analytic thick-layer (= bare-interface) Fresnel, engine forms
            N1 = 1.0;  N2 = complex(testCase.nAl, -testCase.kAl);
            cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
            ctht = sqrt(1 - (N1/N2)^2*(1 - cthi.^2));
            RPa = (N1*ctht - N2*cthi)./(N2*cthi + N1*ctht);
            RSa = (N1*cthi - N2*ctht)./(N1*cthi + N2*ctht);

            testCase.verifyLessThan( ...
                max(abs(abs(ratio_meas) - abs(RSa./RPa))), 1e-12, ...
                'RS/RP magnitude vs Fresnel');
            testCase.verifyLessThan( ...
                max(abs(angle(ratio_meas./(RSa./RPa)))), 1e-12, ...
                'RS/RP phase vs Fresnel');

            % jones_pupil D per ray vs analytic
            jp = macos.jones_pupil(fold);
            pm = macos.pol_maps(jp);
            ko2=[jp.kx(jp.mask), jp.ky(jp.mask), jp.kz(jp.mask)];
            n2 =[rf.nx(jp.mask), rf.ny(jp.mask), rf.nz(jp.mask)];
            ki2 = ko2 - 2*sum(ko2.*n2,2).*n2;
            c2  = abs(sum(ki2.*n2,2));
            ct2 = sqrt(1 - (N1/N2)^2*(1 - c2.^2));
            RP2 = (N1*ct2 - N2*c2)./(N2*c2 + N1*ct2);
            RS2 = (N1*c2 - N2*ct2)./(N1*c2 + N2*ct2);
            Da  = abs(abs(RS2).^2 - abs(RP2).^2)./(abs(RS2).^2 + abs(RP2).^2);
            testCase.verifyLessThan(max(abs(pm.D(jp.mask) - Da)), 1e-12, ...
                'per-ray diattenuation vs Fresnel');
        end

        % ---- 2-theta rotational symmetry ------------------------------
        function test_2theta_symmetry(testCase)
            macos.coating(testCase.Prim, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            macos.coating(testCase.Sec,  'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            pm = macos.pol_maps(macos.jones_pupil(testCase.Det));
            m = pm.mask;
            [ii, jj] = find(m);
            N = size(m,1);
            [JJ, II] = meshgrid(1:N, 1:N);
            R  = sqrt((II-mean(ii)).^2 + (JJ-mean(jj)).^2);
            TH = atan2(JJ-mean(jj), II-mean(ii));
            rmax = max(R(m));
            ring  = m & (R > 0.60*rmax) & (R < 0.75*rmax);
            inner = m & (R > 0.25*rmax) & (R < 0.40*rmax);

            % diattenuation axis locks to azimuth (2-theta pattern):
            % 0.5*atan2(D2,D1) - theta = const (mod pi) at round-off
            D1 = pm.Dvec(:,:,1); D2 = pm.Dvec(:,:,2); D3 = pm.Dvec(:,:,3);
            ang = 0.5*atan2(D2(ring), D1(ring));
            off = 0.5*angle(mean(exp(2i*(ang - TH(ring)))));
            resid = mod(ang - TH(ring) - off + pi/2, pi) - pi/2;
            testCase.verifyLessThan(max(abs(resid)), 1e-10, ...
                'diattenuation orientation must track pupil azimuth');
            % no circular diattenuation from mirrors
            testCase.verifyLessThan(max(abs(D3(m))), 1e-10*max(pm.D(m)), ...
                'no circular component');
            % radial growth (AOI grows off-axis)
            testCase.verifyGreaterThan(mean(pm.D(ring)), 2*mean(pm.D(inner)));
        end

        % ---- pol_maps pure-math identity ------------------------------
        function test_pol_maps_synthetic_identity(testCase)
            % diattenuator (axis s1) * retarder (axis s2): exact recovery
            D0 = 0.3;  del = 0.8;
            t1 = sqrt(1+D0);  t2 = sqrt(1-D0);
            Jd = [t1 0; 0 t2];
            Jr = [cos(del/2), -1i*sin(del/2); -1i*sin(del/2), cos(del/2)];
            Jpt = Jd*Jr;
            J = nan(2,2,2,2);  J = complex(J,J);
            for a=1:2, for b=1:2, J(:,:,a,b) = Jpt(a,b); end, end
            pm = macos.pol_maps(struct('J', J, 'mask', true(2)));
            testCase.verifyEqual(pm.D(1,1),   D0,  'AbsTol', 1e-12);
            testCase.verifyEqual(pm.ret(1,1), del, 'AbsTol', 1e-12);
            % retarder axis is s2: retvec = (0, del, 0)
            testCase.verifyEqual(squeeze(pm.retvec(1,1,:)), [0;del;0], ...
                'AbsTol', 1e-12);
            testCase.verifyFalse(pm.ambiguous(1,1));

            % near-pi retardance trips the ambiguity flag
            Jr2 = [cos(1.6), -1i*sin(1.6); -1i*sin(1.6), cos(1.6)]; % del=3.2>pi-0.2
            for a=1:2, for b=1:2, J(:,:,a,b) = Jr2(a,b); end, end
            pm2 = macos.pol_maps(struct('J', J, 'mask', true(2)));
            testCase.verifyTrue(pm2.ambiguous(1,1));
        end
    end
end
