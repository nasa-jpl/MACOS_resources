classdef tJonesPupil < matlab.unittest.TestCase
%TJONESPUPIL  Phase-2 Jones-pupil physics gates (PLAN_POLARIZATION 2a/2b).
%   Exercises macos.jones_pupil + macos.pol_maps against closed-form truth:
%
%   * unitarity gate     -- stock Cass mirrors carry the perfect-conductor
%                           idiom (IndRef=1, Extinc=1e22): RS=-1, RP=+1 in
%                           the engine's ray-following basis (E_out=-E_in
%                           at normal incidence), so the Jones pupil must
%                           be unitary-to-a-scalar at every point (D, ret,
%                           T-nonuniformity all ~ 0).
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
%   * pol_zernike (2b)   -- the low-order expansion.  Recovers synthetic
%                           Zernike input exactly on an ANNULUS (where the
%                           basis is non-orthogonal, so it pins that the
%                           fit is least-squares); reproduces the published
%                           two-mirror form (POLARIZATION ASTIGMATISM:
%                           astig0 in s1, astig45 in s2, equal magnitude,
%                           no circular part, no defocus, rho^2 radial law
%                           whose on-axis extrapolation vanishes); and
%                           inherits pol_maps' basis behaviour in mode
%                           space (D invariant, retardance not).
%
%   All engine-facing tests run at ModelSize 128 on Rx_Cass_FarField (via
%   rx_fixture_path) or on a Bench-emitted fold rig in a temp folder.
%   Mind the BaseUnits difference between the two -- see thkAl below.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Det       = 6      % Cass focal plane
        Prim      = 2      % Cass primary
        Sec       = 3      % Cass secondary
        nAl       = 1.45   % Al at 632.8 nm
        kAl       = 7.54
        % Al layer thickness: 200 nm, well beyond the ~13 nm skin depth, so
        % the stack reduces to a bare interface and the analytic gate
        % applies.  TWO constants because this class uses TWO prescriptions
        % with DIFFERENT BaseUnits, and macos.coating takes thickness in
        % ELEMENT BaseUnits (a documented exception to the SI-metres veneer
        % convention):
        %   Rx_Cass_FarField  BaseUnits=m   -> 2.0e-7
        %   Bench fold rig    BaseUnits=mm  -> 2.0e-4
        % A single 2.0e-4 used to serve both, which on the Cassegrain
        % silently meant 200 um.  Harmless for the gates (any optically
        % thick layer satisfies them) but it made the mmacos and pymacos
        % Jones coefficients differ in the 8th digit for no stated reason.
        % With this split they agree to 11 digits.
        thkAl     = 2.0e-7   % Rx_Cass_FarField (BaseUnits = m)
        thkAlBench= 2.0e-4   % Bench fold rig   (BaseUnits = mm)
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
            % NOTE: this std/mean statistic has a SUMMATION floor of
            % ~5e-14 (mean over ~1e4 points; the map's true spread is
            % ~6e-15 p-v) -- the 1e-12 gate is a valid regression
            % tripwire two orders above that floor.  Do NOT "tighten"
            % it toward the measured value; you would be asserting the
            % accumulator, not the physics (see polval section 2).
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
            % the coordinate singularity inflates s/p retardance variation.
            % RE-VALIDATED on the r_p-sign-corrected engine (2026-07-27):
            % measured 247x (sp 0.891 vs dp 3.61e-3, sp mean ret = pi/2 --
            % the artifact in person).  A HALF-patched engine (uncoated
            % branch only) makes the two bases spuriously agree at ~3.4e-3
            % -- if this assertion starts failing, suspect a partial
            % convention change in elemsub.F before doubting the doctrine
            % (REVIEW_POL_SP_SIGN_2026-07-27.md, Fable decision section).
            testCase.verifyGreaterThan(pm_sp.var_rms.ret, ...
                10*pm_dp.var_rms.ret, ...
                'local-sp retardance artifact should dominate');
        end

        % ---- Fresnel-analytic gate (fold rig) -------------------------
        function test_fold_fresnel_analytic(testCase)
            macos.load_rx(fullfile(testCase.foldRx, 'foldrig.in'));
            fold = testCase.foldElt;
            macos.coating(fold, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAlBench);

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

            % analytic thick-layer (= bare-interface) Fresnel, TEXTBOOK
            % forms (Born & Wolf, ray-following p-hat).  RPa was
            % originally transcribed from the engine's own expression,
            % which made the phase comparison CIRCULAR in the r_p sign --
            % exactly the sign the 2022 import had flipped (see
            % REVIEW_POL_SP_SIGN_2026-07-27.md).  Written from the
            % textbook now, this gate pins the sign non-circularly:
            % r_p = (N2 c_i - N1 c_t)/(N2 c_i + N1 c_t),
            % r_s = (N1 c_i - N2 c_t)/(N1 c_i + N2 c_t).
            N1 = 1.0;  N2 = complex(testCase.nAl, -testCase.kAl);
            cthi = abs(kix.*nx + kiy.*ny + kiz.*nz);
            ctht = sqrt(1 - (N1/N2)^2*(1 - cthi.^2));
            RPa = (N2*cthi - N1*ctht)./(N2*cthi + N1*ctht);
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
            RP2 = (N2*c2 - N1*ct2)./(N2*c2 + N1*ct2);   % textbook r_p (sign-
            RS2 = (N1*c2 - N2*ct2)./(N1*c2 + N2*ct2);   % invariant in Da below)
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

        % ---- 2b: Zernike expansion recovers synthetic input exactly ----
        function test_pol_zernike_synthetic_recovery(testCase)
            % Pure-math gate: build Dvec/retvec maps FROM known Zernike
            % coefficients, then confirm macos.pol_zernike returns them.
            % Uses an annular mask on purpose -- circular Zernikes are not
            % orthogonal there, so this also proves the least-squares fit
            % (rather than a naive projection) is doing the right thing.
            N = 64;
            [II, JJ] = ndgrid(1:N, 1:N);
            c0 = (N+1)/2;
            R = hypot(II-c0, JJ-c0);
            rad = 0.45*N;
            mask = (R <= rad) & (R >= 0.25*rad);     % annulus
            modes = [1 4 5 6 9 13];
            rho = R/rad;  th = atan2(JJ-c0, II-c0);
            B = zeros(N, N, numel(modes));
            for k = 1:numel(modes)
                B(:,:,k) = tJonesPupil.zern_(modes(k), rho, th);
            end
            ctrue = [ 1.0 -0.3  0.7
                     -2.0  0.5  0.0
                      0.4  1.1 -0.6
                      3.0 -0.2  0.9
                     -0.7  0.8  0.1
                      0.2 -1.4  0.3];
            Dv = zeros(N, N, 3);  Rv = zeros(N, N, 3);
            for c = 1:3
                acc = zeros(N);
                for k = 1:numel(modes)
                    acc = acc + ctrue(k,c)*B(:,:,k);
                end
                acc(~mask) = NaN;
                Dv(:,:,c) = acc;  Rv(:,:,c) = 2*acc;
            end
            pmS = struct('Dvec', Dv, 'retvec', Rv, ...
                         'D', sqrt(sum(Dv.^2,3)), 'ret', sqrt(sum(Rv.^2,3)), ...
                         'mask', mask);
            pz = macos.pol_zernike(pmS, 'modes', modes, ...
                                   'center', [c0 c0], 'radius', rad);
            testCase.verifyEqual(pz.D,   ctrue,   'AbsTol', 1e-10, ...
                'Dvec Zernike coefficients must be recovered exactly');
            testCase.verifyEqual(pz.ret, 2*ctrue, 'AbsTol', 1e-10, ...
                'retvec Zernike coefficients must be recovered exactly');
            testCase.verifyLessThan(max(pz.resid_rms.D), 1e-10);
            % names/orders are the documented ANSI mapping
            testCase.verifyEqual(pz.names{ modes == 6 }, 'astig0');
            testCase.verifyEqual(pz.nm(modes == 6, :), [2 2]);
            testCase.verifyEqual(pz.nm(modes == 4, :), [2 -2]);
        end

        % ---- 2b: two-mirror system reproduces the literature form ------
        function test_pol_zernike_two_mirror_form(testCase)
            % Standard polarization-aberration theory for an ON-AXIS
            % rotationally symmetric two-mirror system: diattenuation and
            % retardance grow as rho^2 with a 2*theta azimuthal
            % dependence.  In the Pauli representation that is EXACTLY
            % astigmatism -- astig0 in s1, astig45 in s2, equal magnitude,
            % nothing else, and no circular (s3) component.  This is the
            % pattern the literature calls "polarization astigmatism", and
            % it is what makes MACOS results comparable term-by-term.
            macos.coating(testCase.Prim, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            macos.coating(testCase.Sec,  'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            pm = macos.pol_maps(macos.jones_pupil(testCase.Det));
            pz = macos.pol_zernike(pm);

            iA0 = find(pz.modes == 6);      % astig0   (rho^2 cos2th)
            iA45= find(pz.modes == 4);      % astig45  (rho^2 sin2th)
            i2A0= find(pz.modes == 14);     % rho^4 cos2th companion
            i2A45=find(pz.modes == 12);

            for f = {'D', 'ret'}
                C = pz.(f{1});
                a0 = abs(C(iA0,1));  a45 = abs(C(iA45,2));
                testCase.verifyGreaterThan(a0, 0, ...
                    sprintf('%s: astigmatism must be present', f{1}));
                % The astigmatic pair is equal in magnitude (the map is a
                % pure 2-theta rotation of one shape).  The residual
                % asymmetry is a PUPIL-DISCRETIZATION effect, not physics:
                % measured 1.9e-7 at model 128 and 5.8e-8 at 256 -- it
                % shrinks with sampling, and it is the same value for D
                % and for retardance, which physics would not arrange.
                % Tolerance set for the size this class runs at.
                testCase.verifyEqual(a0, a45, 'RelTol', 1e-6, sprintf( ...
                    '%s: astig0(s1) and astig45(s2) must be equal', f{1}));
                % EVERYTHING else in s1/s2 is round-off, except the rho^4
                % companion which theory also allows (measured ~0.26%)
                keep = true(numel(pz.modes), 1);
                keep([iA0 iA45 i2A0 i2A45]) = false;
                other = max(max(abs(C(keep, 1:2))));
                testCase.verifyLessThan(other/a0, 1e-10, sprintf( ...
                    ['%s: only astigmatism (and its rho^4 companion) may ' ...
                     'appear -- no piston/tilt/defocus/coma/trefoil'], f{1}));
                % the rho^4 companion is present but strictly sub-dominant
                testCase.verifyLessThan(abs(C(i2A0,1))/a0, 1e-2, ...
                    sprintf('%s: rho^4 companion must be sub-dominant', f{1}));
                % no circular diattenuation/retardance from mirrors
                testCase.verifyLessThan(max(abs(C(:,3)))/a0, 1e-10, ...
                    sprintf('%s: no circular (s3) component', f{1}));
            end

            % Radial law: the MAGNITUDE map of a rho^2 aberration is
            % piston + defocus only.  Its on-axis extrapolation must
            % vanish -- at normal incidence there is no diattenuation --
            % which the fit is never told and has no way to arrange.
            cm = pz.Dmag;
            keep = true(numel(pz.modes),1);
            keep([find(pz.modes==1) find(pz.modes==5) find(pz.modes==13)]) = false;
            % Tolerance 1e-6, not round-off, and the reason is measured:
            % the largest non-symmetric term is quadrafoil-X (cos4th) at
            % 1.0e-7 of piston at model 128, falling to 2.8e-8 at 256,
            % while quadrafoil-Y (sin4th) stays at 1e-17.  A square pixel
            % grid is 4-fold symmetric about its OWN axes, so it imprints
            % cos4th and not sin4th; physics on a rotationally symmetric
            % system has no way to prefer the grid's axes.  Discretization,
            % and it shrinks with sampling.
            testCase.verifyLessThan(max(abs(cm(keep)))/abs(cm(pz.modes==1)), ...
                1e-6, 'diattenuation magnitude must be rotationally symmetric');
            D0 = tJonesPupil.zern_eval_at_(pz.modes, cm, 0, 0);   % rho = 0
            D1 = tJonesPupil.zern_eval_at_(pz.modes, cm, 1, 0);   % rho = 1
            testCase.verifyLessThan(abs(D0)/abs(D1), 1e-3, ...
                'extrapolated on-axis diattenuation must vanish');
        end

        % ---- 2b: the expansion inherits pol_maps' basis behaviour ------
        function test_pol_zernike_basis_dependence(testCase)
            % D is a singular-value invariant, so its EXPANSION must be
            % basis-independent too; retardance is not, and the local-sp
            % coordinate singularity shows up in mode space as a large
            % spurious low-order term.
            macos.coating(testCase.Sec, 'index', testCase.nAl, ...
                'extinc', testCase.kAl, 'thickness', testCase.thkAl);
            pzd = macos.pol_zernike(macos.pol_maps( ...
                macos.jones_pupil(testCase.Det, 'basis', 'double-pole')));
            pzs = macos.pol_zernike(macos.pol_maps( ...
                macos.jones_pupil(testCase.Det, 'basis', 'local-sp')));
            scale = max(abs(pzd.D(:)));
            testCase.verifyLessThan(max(abs(pzd.D(:) - pzs.D(:)))/scale, ...
                1e-9, 'diattenuation expansion must be basis-invariant');
            testCase.verifyGreaterThan(max(abs(pzs.ret(:))), ...
                10*max(abs(pzd.ret(:))), ...
                'local-sp must inflate the retardance expansion');
        end
    end

    methods (Static)
        function Z = zern_(j, rho, th)
            % ANSI mode on caller-supplied polar coordinates.  Duplicated
            % here (small) rather than reaching into +macos/private/, which
            % is not visible from the tests folder.
            jj = j - 1;
            n  = ceil((-3 + sqrt(9 + 8*jj)) / 2);
            m  = 2*jj - n*(n + 2);
            am = abs(m);
            R  = zeros(size(rho));
            for s = 0:((n - am)/2)
                c = (-1)^s * factorial(n - s) / ...
                    (factorial(s) * factorial((n+am)/2 - s) * factorial((n-am)/2 - s));
                R = R + c * rho.^(n - 2*s);
            end
            if m >= 0, ang = cos(m*th); else, ang = sin(am*th); end
            P = [1, 2, 2, sqrt(6), sqrt(3), sqrt(6), sqrt(8), sqrt(8), ...
                 sqrt(8), sqrt(8), sqrt(10), sqrt(10), sqrt(5), sqrt(10), sqrt(10)];
            Z = P(j) .* R .* ang;
        end

        function v = zern_eval_at_(modes, coefs, rho, th)
            v = 0;
            for k = 1:numel(modes)
                v = v + coefs(k)*tJonesPupil.zern_(modes(k), rho, th);
            end
        end
    end
end
