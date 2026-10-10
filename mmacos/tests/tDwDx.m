classdef tDwDx < matlab.unittest.TestCase
%TDWDX  Regression tests for macos.dw_dx + multi-field.

    properties (Constant)
        ModelSize       = 128
        RxName          = 'e5hex1.in'
        DOFsForTest     = (3:5).'  % Tx,Ty,Tz only -- keep tests fast
        ExpectedActOpts = 11       % 13 elements - 2 Reference/Return
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (Test)
        function test_actual_optic_count(testCase)
            % Parse the Rx text -- 13 elements minus 2 Reference/Return
            % should leave 11 actual optics.
            macos.load_rx(testCase.rx_path);
            chs = macos.channels.rigid_body_channels( ...
                macos.Session(testCase.ModelSize), testCase.rx_path, ...
                'dofs', [3]);
            testCase.verifyEqual(numel(chs), testCase.ExpectedActOpts, ...
                'rigid_body_channels actual-optic count mismatch');
        end

        function test_single_field_shape(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8);
            n_dof = numel(testCase.DOFsForTest);
            expected = testCase.ExpectedActOpts * n_dof;
            testCase.verifyEqual(numel(out.channel_names), expected);
            testCase.verifyEqual(size(out.dwdx, 2), expected);
            testCase.verifyEqual(size(out.dwdx, 1), numel(out.w_nom_vec));
            testCase.verifyGreaterThan(max(abs(out.dwdx(:))), 0);
        end

        function test_element_major_channel_order(testCase)
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest);
            % Per-element block: Elt 1 (Tx,Ty,Tz), Elt 2 ...
            for k = 1:numel(out.channel_names)
                expected_elt = ceil(k / numel(testCase.DOFsForTest));
                actual = sscanf(out.channel_names{k}, 'Elt %d');
                testCase.verifyEqual(actual, ...
                    out.iElt(find(out.iElt > 0, 1) + expected_elt - 1), ...
                    'Channel order not element-major');
                break;   % single-element check is sufficient evidence
            end
        end

        function test_multi_field_5fp_shapes(testCase)
            m = macos.Session(testCase.ModelSize);
            % Shape check only (EP-convention independent).  Pinned to
            % reset_xp=false so it stays a pure per-field-tiling test and
            % skips the FEX resets.
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8, ...
                'reset_xp', false);
            n_dof = numel(testCase.DOFsForTest);
            expected = testCase.ExpectedActOpts * n_dof;
            testCase.verifyEqual(numel(out.field_names), 5);
            testCase.verifyEqual(size(out.field_table, 1), 5);
            testCase.verifyEqual(size(out.field_table, 2), 4);
            testCase.verifyEqual(size(out.dwdxall, 2), expected);
        end

        function test_ngridpts_override(testCase)
            % 'ngridpts' overrides the .in ray-grid sampling (Luis's
            % request): the OPD canvas follows the override, not the
            % .in value / model clamp.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'ngridpts', 31);
            testCase.verifyEqual(size(out.w_nom_2d), [31 31]);
            testCase.verifyEqual(double(m.get_src_sampling()), 31);
        end

        function test_ngridpts_clamp_warns(testCase)
            % Oversized request clamps to the model limit and warns.
            m = macos.Session(testCase.ModelSize);
            testCase.verifyWarning(@() macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'ngridpts', 99999), 'macos:dw_dx:ngridpts');
            testCase.verifyLessThanOrEqual( ...
                double(m.get_src_sampling()), testCase.ModelSize);
        end

        function test_multi_ngridpts_override(testCase)
            % Supervisor applies the override once after load_rx; it
            % persists across the per-field calls (reload_rx=false),
            % so every tile comes out at the override size.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', [3], 'ngridpts', 31, 'reset_xp', false);  % shape only
            testCase.verifyEqual(size(out.per_field_w_nom_2d{1}), [31 31]);
            testCase.verifyEqual(size(out.OPDall), [3*31 3*31]);
        end

        function test_multi_field_center_tile_bitwise(testCase)
            % Bitwise scatter/tiling check, EP-convention independent.
            % Pinned to reset_xp=false to isolate the tiling invariant.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest, 'reset_xp', false);
            cidx = find(out.field_table(:,1) == 0 ...
                      & out.field_table(:,2) == 0, 1);
            testCase.verifyNotEmpty(cidx);
            tr = out.field_table(cidx, 3);
            tc = out.field_table(cidx, 4);
            indx = out.indxall;
            in_ctr = (indx.i > tr*128) & (indx.i <= (tr+1)*128) ...
                   & (indx.j > tc*128) & (indx.j <= (tc+1)*128);
            dwdxall_ctr = out.dwdxall(in_ctr, :);
            dwdx_C = out.per_field_dwdx{cidx};
            testCase.verifyEqual( ...
                max(abs(dwdxall_ctr(:) - dwdx_C(:))), 0, ...
                'Center-tile rows of dwdxall must bitwise-match per_field_dwdx[center]');
        end

        % ---- PR #11 additions: elts / src_samp / per-DOF delta / LOS ----

        function test_default_delta_unchanged(testCase)
            % The scalar-default call must produce the SAME Jacobian as
            % the historical explicit delta=1e-8 -- guards against the
            % default silently drifting (the 1e-5 regression).
            m = macos.Session(testCase.ModelSize);
            out_def = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest);
            out_1e8 = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8);
            testCase.verifyEqual(out_def.delta, 1e-8, ...
                'default delta must be 1e-8');
            testCase.verifyEqual(out_def.dwdx, out_1e8.dwdx, ...
                'default-delta Jacobian must match explicit delta=1e-8');
        end

        function test_elts_subset(testCase)
            % 'elts' restricts the perturbed set to the intersection with
            % the discovered actual optics.  Pick two element ids that are
            % actual optics in the fixture.
            m = macos.Session(testCase.ModelSize);
            full = macos.dw_dx(m, testCase.rx_path, 'dofs', [3]);
            opt_elts = unique(full.iElt(full.iElt > 0));
            testCase.assumeGreaterThanOrEqual(numel(opt_elts), 2);
            keep = opt_elts(1:2).';
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'elts', keep);
            n_dof = numel(testCase.DOFsForTest);
            testCase.verifyEqual(size(out.dwdx, 2), numel(keep) * n_dof, ...
                'elts must restrict the channel count to the kept optics');
            testCase.verifyEqual(unique(out.iElt(out.iElt > 0)).', keep, ...
                'only the kept element ids may appear as channels');
        end

        function test_src_samp_override(testCase)
            % 'src_samp' resamples the source ray grid before the sweep,
            % same effect as 'ngridpts' but via set_src_sampling.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'src_samp', 31);
            testCase.verifyEqual(size(out.w_nom_2d), [31 31]);
            testCase.verifyEqual(double(m.get_src_sampling()), 31);
        end

        function test_per_dof_delta_matches_scalar(testCase)
            % A (1,6) delta whose Tx,Ty,Tz entries all equal the scalar
            % must reproduce the scalar-delta Jacobian exactly.
            m = macos.Session(testCase.ModelSize);
            d = 1e-8;
            out_s = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', d);
            out_v = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', repmat(d, 1, 6));
            testCase.verifyEqual(out_v.dwdx, out_s.dwdx, ...
                'uniform (1,6) delta must match the scalar delta');
        end

        function test_delta_units_base_matches_si_default(testCase)
            % e5hex1.in is an mm prescription (CBM=1e-3).  A BaseUnits
            % delta of 1e-5 (mm) is the same 10 nm translation poke as the
            % 1e-8 SI-metres default -> identical translation Jacobian.
            m = macos.Session(testCase.ModelSize);
            out_si = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'delta', 1e-8);   % SI, Tx/Ty/Tz
            out_bu = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, ...
                'delta', 1e-5, 'delta_units', 'base');
            testCase.verifyEqual(out_bu.cbm, 1e-3, 'AbsTol', 1e-12, ...
                'fixture must be an mm Rx for this equivalence');
            testCase.verifyEqual(out_bu.dwdx, out_si.dwdx, 'RelTol', 1e-9, ...
                'base-units 1e-5 mm must match SI 1e-8 m for translations');
        end

        function test_per_dof_delta_bad_size_errors(testCase)
            m = macos.Session(testCase.ModelSize);
            testCase.verifyError(@() macos.dw_dx(m, testCase.rx_path, ...
                'dofs', [3], 'delta', [1e-8 1e-8 1e-8]), ...
                'macos:dw_dx:deltaSize');
        end

        function test_compute_los_shapes(testCase)
            % LOS/centroid sensitivities: dcdx is Nz x 2; each row is the
            % [dc_x/dX, dc_y/dX] centroid shift at the focal plane.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, ...
                'dofs', testCase.DOFsForTest, 'compute_los', true);
            Nz = size(out.dwdx, 2);
            testCase.verifyEqual(size(out.dcdx), [Nz 2], ...
                'dcdx must be Nz x 2');
            testCase.verifyEqual(out.spot_elt, macos.num_elt(), ...
                'default spot_elt is the last (focal-plane) element');
            testCase.verifyGreaterThan(max(abs(out.dcdx(:))), 0, ...
                'rigid-body perturbations must move the centroid');
        end

        function test_dcdx_of_a_rigid_tilt_is_the_chief_displacement(testCase)
            % Luis's OPTIIX FSM test (2026-10-06): dw_dx's centroid channel
            % took macos.spot(...,'at','chief'), the spot CENTRED ON THE
            % CHIEF RAY, so a rigid displacement of the spot (a mirror
            % tilt) was subtracted out and dcdx read ~0 for it -- only the
            % aberration change survived.  The centroid for a line-of-sight
            % sensitivity is about the ELEMENT.  Here the Cassegrain's
            % secondary (elt 3) is tilted about x: dcdx_y must equal the
            % chief ray's own displacement per radian at the focal plane,
            % measured independently from the traced chief (ray 1).
            rx = rx_fixture_path('Rx_Cass_FarField.in');
            m = macos.Session(testCase.ModelSize);
            d = 2e-7;
            out = macos.dw_dx(m, rx, 'elts', 3, 'dofs', 0, 'delta', d, 'compute_los', true, 'method', 'central');
            m.load_rx(rx);  nE = m.num_elt();
            p0 = chief_(m, nE);
            m.load_rx(rx);  macos.perturb(3, 'rotation', [d 0 0], 'translation', [0 0 0], 'frame', 'global');
            p1 = chief_(m, nE);
            dch = (p1 - p0)/d;                      % the chief's displacement per rad, global frame
            % the spot is in the TOUT frame; on this coaxial deck Tout's x,y are the global x,y
            testCase.verifyGreaterThan(norm(dch(1:2)), 1, 'the tilt must move the chief (m per rad)');
            testCase.verifyEqual(out.dcdx(1, :), dch(1:2).', 'AbsTol', 0.05*norm(dch(1:2)), ...
                sprintf('dcdx %s vs the chief''s %s m/rad (pre-fix: dcdx ~ 0, the chief-centred spot)', mat2str(out.dcdx(1, :), 4), mat2str(dch(1:2).', 4)));
            function p = chief_(m, nE)
                s = m.trace(nE);  ri = macos.get_ray_info(s.nRays);  p = ri.pos(:, 1);
            end
        end

        function test_no_los_by_default(testCase)
            % Without compute_los the struct carries no LOS fields.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx(m, testCase.rx_path, 'dofs', [3]);
            testCase.verifyFalse(isfield(out, 'dcdx'), ...
                'dcdx must be absent unless compute_los is set');
        end

        function test_multi_compute_los(testCase)
            % Regression for the dw_dx_multi LOS crash: before the fix,
            % dw_dx_multi forwarded only 'spot_elt' (never compute_los),
            % so dw_dx never populated out.dcdx and the supervisor threw
            % "Unrecognized field name 'dcdx'".  compute_los must now
            % populate dcdx_per_field, one Nz x 2 cell per field.
            %
            % Uses a single ON-AXIS field ('grid','1x1'): macos.spot with
            % 'at','chief' can fail at off-axis fields when a rigid-body
            % perturbation vignettes the chief ray (see report -- an
            % engine-side spot fragility, orthogonal to this crash fix).
            % reset_xp=false keeps this focused on the LOS crash regression.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                'dofs', testCase.DOFsForTest, 'compute_los', true, ...
                'reset_xp', false);
            testCase.verifyTrue(isfield(out, 'dcdx_per_field'));
            testCase.verifyEqual(numel(out.dcdx_per_field), ...
                numel(out.field_names));
            Nz = size(out.dwdxall, 2);
            testCase.verifyEqual(size(out.dcdx_per_field{1}), [Nz 2]);
            testCase.verifyEqual(out.spot_elt, macos.num_elt());
        end

        % ---- trans_output: translation columns per BaseUnit (2026-10-06) ----
        % Dave's ruling: the default 'base' emits OPD-BaseUnits per BaseUnit
        % of translation (GMI's dwdx, what Luis compared against); 'si' is
        % per SI metre, the 2026-08-25..10-06 default.  Every gate runs on
        % e5hex1 (mm, cbm = 1e-3) -- on a metre deck the two conventions
        % are identical and these gates would be vacuous.

        function test_trans_output_default_is_base_x_cbm(testCase)
            % (a) default 'base' translation column == 'si' column x cbm.
            % MUST-FAIL leg: the pre-change dw_dx has no 'trans_output'
            % and its default column is the 'si' one, 1000x this.
            m = macos.Session(testCase.ModelSize);
            a = {'elts', [1 8], 'dofs', (0:5).', 'compute_los', true};
            ob = macos.dw_dx(m, testCase.rx_path, a{:});
            os = macos.dw_dx(m, testCase.rx_path, a{:}, 'trans_output', 'si');
            testCase.verifyEqual(ob.cbm, 1e-3, 'AbsTol', 1e-15, ...
                'fixture must be an mm deck (factor != 1)');
            testCase.verifyEqual(ob.trans_output, 'base');
            testCase.verifyEqual(os.trans_output, 'si');
            t = ob.dof_idx >= 3;
            testCase.verifyEqual(ob.dwdx(:, t), os.dwdx(:, t) * os.cbm, ...
                'RelTol', 1e-12, 'base translation columns = si x cbm');
            testCase.verifyEqual(ob.dcdx(t, :), os.dcdx(t, :) * os.cbm, ...
                'RelTol', 1e-12, 'dcdx translation rows follow the same rule');
            % (d) rotations untouched -- bit for bit, columns and dcdx rows
            testCase.verifyEqual(ob.dwdx(:, ~t), os.dwdx(:, ~t));
            testCase.verifyEqual(ob.dcdx(~t, :), os.dcdx(~t, :));
        end

        function test_trans_output_si_reproduces_the_pre_change_numbers(testCase)
            % (c) 'si' == the dw_dx of HEAD 622ee52 (before trans_output).
            % Measured BIT-FOR-BIT on 2026-10-06: the full 66-column
            % harvest (all optics x 6 DOFs + dcdx) and this 12-column
            % subset, old code vs new code, isequal == true.  Pinned
            % here as column rms + dcdx at 1e-12 so an ulp-level engine
            % rebuild does not break the gate; a units slip is 1e3.
            % RE-PINNED 2026-10-09 (PLAN_CONSOLIDATION 6a, macos): the first
            % trace after a load now KEEPS the deck's source frame (zGrid set
            % at load, so OrthoSrcFrame's dead band engages) instead of
            % rebuilding it from cross products -- a round-off-level change of
            % the launch frame.  dw_dx is a finite difference, so that moves
            % every column by ~1e-6 ABSOLUTE (rel 1e-9 on the 500-class
            % columns, 7.8e-6 on the 0.09 one): its noise floor, the same
            % scale as the ~0 dcdx entries (1e-6 .. 3e-5).  Legacy pins, for
            % the record: rms 521.05555419928555 516.8101056700857
            % 0.091711835834967406 10.065414993980431 10.068175165126112
            % 694.73520319921772 546.86500461825199 540.99818020533451
            % 0.67919771920798078 61.513472236407203 60.877583531768714
            % 12.672957482576745.  New values reproduce bit-for-bit across
            % sessions; a units slip is still 1e3.
            % RE-PINNED 2026-10-10 (BRIEF_to_opd_reference, CC's a71b43f): the
            % drivers' opd_ref default is now 'chief' (was 'mean'), matching
            % the engine and the manual.  MEASURED on this harvest: the
            % 'mean' run still reproduces the 2026-10-09 pins EXACTLY (the
            % pre-change numbers below); chief - mean is ONE CONSTANT per
            % column (std/|shift| <= 4e-8) -- the piston the aperture mean
            % attributed to every ray -- largest on the two Tz columns,
            % which poke the mean directly (col 6 +1700.13, col 12 +5.334;
            % e5hex1 is segmented); dcdx bit-identical; rows 10245 both.
            % 2026-10-09 pins: rms 521.0555532558767 516.81010609177349
            % 0.091711117735078299 10.065415064433068 10.068174730560619
            % 694.73520354741947 546.86500408916811 540.99817854416744
            % 0.67919640915289214 61.513473279007904 60.877582423073655
            % 12.672958789441102.
            m = macos.Session(testCase.ModelSize);
            o = macos.dw_dx(m, testCase.rx_path, 'elts', [1 8], ...
                'dofs', (0:5).', 'compute_los', true, 'trans_output', 'si');
            pin_rms = [521.05555326098965 516.81010609246698 0.091711117738404013 ...
                10.065417648308953 10.068182632608565 1836.6010453356064 ...
                546.86500907037771 540.99817854416756 0.67919640919305313 ...
                61.513473279007883 60.882242286992089 13.749900519007273];
            pin_dcdx = [25355.059831627135 44281.553562797169; ...
                43916.251852744805 -25565.96689586854; ...
                2.2763355555969107 -2.4868995751603507e-06; ...
                858.64096847030919 -492.01795704334472; ...
                -495.73659506123869 -852.20002112862403; ...
                1.7532178802387457e-08 124.48102069129163; ...
                -1.2277311948656669e-07 52061.786263379872; ...
                51495.652155054711 3.1263880373444408e-05; ...
                64.17829468561888 1.0302869668521453e-05; ...
                5848.1317754687343 9.2370555648813024e-06; ...
                2.9247685017880134e-07 -5788.8080483792237; ...
                -4.2223153319503865e-07 1163.5166210055559];
            testCase.verifyEqual(rms(o.dwdx, 1), pin_rms, 'RelTol', 1e-12);
            % dcdx: relative on the live entries, absolute on the ~0 ones
            testCase.verifyEqual(o.dcdx, pin_dcdx, 'AbsTol', 1e-6, ...
                'RelTol', 1e-12);
        end

        function test_segment_piston_is_two_cos_aoi_per_baseunit(testCase)
            % (b) physical magnitude.  A Tz (along the segment normal) of
            % a near-normal segment by d changes the reflected path by
            % 2*d*cos(AOI) on that segment's footprint and by nothing
            % elsewhere -- read under the CHIEF reference so the other
            % segments are exactly 0 (under 'mean' they piston, PLAN 0.x).
            % Per BaseUnit that is ~2 (mm per mm); the 'si' column is
            % ~2000 and fails the band.  AOI on e5hex1 < 8 deg.
            m = macos.Session(testCase.ModelSize);
            o = macos.dw_dx(m, testCase.rx_path, 'elts', 2, 'dofs', 5, ...
                'opd_ref', 'chief');
            c = o.dwdx(:, 1);
            on = abs(c) > 1e-3;
            testCase.verifyEqual(nnz(on)/numel(c), 1/7, 'AbsTol', 0.02, ...
                'one hex segment of seven carries the piston');
            testCase.verifyLessThan(max(abs(c(~on))), 1e-6, ...
                'the unpoked segments read 0 under the chief reference');
            testCase.verifyLessThanOrEqual(max(abs(c(on))), 2 + 1e-9, ...
                '|dOPD/dTz| <= 2 BaseUnits per BaseUnit');
            testCase.verifyGreaterThan(min(abs(c(on))), 2*cosd(8), ...
                '|dOPD/dTz| >= 2 cos(8 deg) BaseUnits per BaseUnit');
        end

        function test_trans_per_metre_helper(testCase)
            % macos.dwdx_trans_per_metre, the runners' one-line adapter:
            % IDENTITY on a pre-change harvest (no trans_output field --
            % bit for bit), and on a 'base' harvest the translation
            % columns / dcdx rows come back x 1/cbm, rotations untouched,
            % on single-field AND multi-field (dwdxall, per_field_dwdx,
            % dcdx_per_field) outputs.  mm deck: factor 1e3.
            m = macos.Session(testCase.ModelSize);
            a = {'elts', [1 8], 'dofs', (0:5).', 'compute_los', true};
            os = macos.dw_dx(m, testCase.rx_path, a{:}, 'trans_output', 'si');
            ob = macos.dw_dx(m, testCase.rx_path, a{:});
            legacy = rmfield(os, 'trans_output');   % == a HEAD-622ee52 harvest
            testCase.verifyEqual(macos.dwdx_trans_per_metre(legacy), legacy, ...
                'a pre-change harvest must pass through bit for bit');
            testCase.verifyEqual(macos.dwdx_trans_per_metre(os), os);
            cv = macos.dwdx_trans_per_metre(ob);
            t = ob.dof_idx >= 3;
            testCase.verifyEqual(cv.trans_output, 'si');
            testCase.verifyEqual(cv.dwdx(:, t), ob.dwdx(:, t) / ob.cbm, ...
                'RelTol', 1e-12, 'translation columns x 1/cbm');
            testCase.verifyEqual(cv.dwdx, os.dwdx, 'RelTol', 1e-12);
            testCase.verifyEqual(cv.dcdx, os.dcdx, 'RelTol', 1e-12, ...
                'AbsTol', 1e-9);
            testCase.verifyEqual(cv.dwdx(:, ~t), ob.dwdx(:, ~t));
            testCase.verifyEqual(macos.dwdx_trans_per_metre(cv), cv, ...
                'second call is a no-op');
            % multi-field
            b = {'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                'elts', 8, 'dofs', [1 5].', 'compute_los', true, ...
                'reset_xp', false};
            ms = macos.dw_dx_multi(m, testCase.rx_path, b{:}, 'trans_output', 'si');
            mb = macos.dw_dx_multi(m, testCase.rx_path, b{:});
            testCase.verifyEqual(mb.trans_output, 'base');
            mc = macos.dwdx_trans_per_metre(mb);
            testCase.verifyEqual(mc.dwdxall, ms.dwdxall, 'RelTol', 1e-12);
            testCase.verifyEqual(mc.per_field_dwdx{1}, ms.per_field_dwdx{1}, ...
                'RelTol', 1e-12);
            testCase.verifyEqual(mc.dcdx_per_field{1}, ms.dcdx_per_field{1}, ...
                'RelTol', 1e-12, 'AbsTol', 1e-9);
            testCase.verifyEqual(mb.dwdxall(:, 2), ms.dwdxall(:, 2) * mb.cbm, ...
                'RelTol', 1e-12, 'multi: Tz column per BaseUnit by default');
        end

        % ---- per-field exit-pupil reset (reset_xp) ----------------------

        function test_reset_xp_default_and_stamp(testCase)
            % Default is true (family alignment) and the convention is
            % stamped in the output for run_compare's match check.  The
            % harness fixture declares ApStop= 0 0 0, so FEX resolves with
            % no explicit stop argument.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest);
            testCase.verifyTrue(isfield(out, 'reset_xp'), ...
                'out must stamp the reset_xp convention');
            testCase.verifyTrue(out.reset_xp, ...
                'reset_xp must default true (align dwdz/dwdsurf/dwdgrid)');
        end

        % NOTE: the reset_xp=true no-stop guard (macos:dw_dx_multi:noStop)
        % is not unit-tested here.  The harness fixture
        % (pymacos/tests/Rx/e5hex1.in via rx_fixture_path) declares
        % "ApStop= 0 0 0" in its header, so load_rx sets a stop and FEX
        % always succeeds -- the genuine no-stop path cannot be provoked on
        % it.  (The stop-less copy under templates/60_visualization/view_rx_demo/e5hex1.in
        % DOES raise macos:fex:noStop, which the guard rethrows -- verified
        % during development.)  The guard is a pure defensive rethrow, so
        % every reset_xp=true test below passes through it; run_sensitivities
        % carries the text-level ApStop preflight for the batch path.

        function test_reset_xp_continuity_arcminute(testCase)
            % Continuity: at arcminute fields the per-field EP reset and
            % the frozen EP must agree closely -- the removed term is only
            % the first-order tilt-sensitivity residual, negligible here.
            % (The fixture nominal ChfRayDir is itself ~1.2 arcmin off
            % axis, so no field is exactly on-axis; the claim is global
            % closeness, not per-field identity.)
            m = macos.Session(testCase.ModelSize);
            fx = 1e-4;   % ~0.34 arcmin half-field
            out_reset  = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', fx, 'field_y_rad', fx, ...
                'dofs', testCase.DOFsForTest, 'reset_xp', true);
            out_frozen = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', fx, 'field_y_rad', fx, ...
                'dofs', testCase.DOFsForTest, 'reset_xp', false);
            % Off-axis blocks agree to a loose tolerance at arcminute
            % fields (the removed residual is small, not zero).  The reset
            % strips a per-field piston/tilt reference, so compare on the
            % piston-removed columns to isolate the sensitivity residual.
            rel = norm(out_reset.dwdxall - out_frozen.dwdxall, 'fro') ...
                / max(norm(out_frozen.dwdxall, 'fro'), realmin);
            testCase.verifyLessThan(rel, 0.15, ...
                'arcminute-field reset vs frozen must be close (continuity)');
        end

        function test_reset_xp_restore_discipline(testCase)
            % The per-field FEX mutates elt nElt-1 geometry; the supervisor
            % must restore the as-loaded EP after the field loop so the
            % session is left exactly as the prescription loaded it.
            % dw_dx_multi always load_rx's internally, so a fresh load here
            % reproduces the identical as-loaded EP for the comparison.
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);   % fixture declares ApStop= 0 0 0
            xp_before = macos.get_xp();
            macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest, 'reset_xp', true);
            xp_after = macos.get_xp();
            testCase.verifyEqual(xp_after.vpt, xp_before.vpt, 'AbsTol', 1e-9, ...
                'EP vertex must be restored after the field loop');
            testCase.verifyEqual(xp_after.psi, xp_before.psi, 'AbsTol', 1e-12, ...
                'EP normal must be restored after the field loop');
            testCase.verifyEqual(xp_after.rad, xp_before.rad, 'RelTol', 1e-9, ...
                'EP radius must be restored after the field loop');
        end

        function test_reset_xp_composes_with_fp_track(testCase)
            % fp_mode='track' saves/restores EP vpt/psi/rpt around its FP
            % pokes; with reset_xp the per-field EP is written BEFORE the
            % channels build, so track must run cleanly (no error) and
            % produce a non-zero FP-DOF Jacobian on top of the reset EP.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', (0:5).', 'fp_mode', 'track', ...
                'include_non_optics', true, 'reset_xp', true);
            testCase.verifyGreaterThan(max(abs(out.dwdxall(:))), 0, ...
                'track + reset_xp must yield a non-zero Jacobian');
        end

        % ---- empty-OPD guard + single-field identity --------------------

        function test_emptyOPD_guard_on_clipped_read_surface(testCase)
            % A deck whose read surface (nElt-1, the exit-pupil Return)
            % clips the whole beam yields an empty per-field OPD.  Under
            % Dave's 2026-09-07 ruling the supervisor WARNS once
            % (macos:dw_dx_multi:emptyOPD) and completes with 0 rows from
            % that block -- never errors.  Fixture: the e5hex1 pupil deck
            % with a 0.1 mm circular aperture on its ExitPupil Return, centred
            % 100 m off axis, so every ray is clipped exactly at the read
            % surface (committed fixture; the former rodgers1_stage4 deck is
            % not in the repo).  The aperture used to be ON axis: that let
            % exactly ONE ray through (the central one, whose OPD is exactly
            % 0 under either reference), which the old W ~= 0 test called
            % empty; the valid-ray mask (macos.opd_mask, 2026-10-10) counts
            % it, so the fixture now clips for real (0 rays, measured).
            txt = fileread(testCase.rx_path);
            k = strfind(txt, 'EltName=  exitpupil');
            testCase.assumeTrue(~isempty(k), 'e5hex1 exitpupil block not found');
            blk = txt(k(1):end);
            blk = regexprep(blk, '(ApType=\s*)None', ...
                ['$1Circular' newline '            ApVec=  1.0E-04  1.0E+05  0.0E+00'], 'once');
            tmp = [tempname '_clipxp.in'];
            fid = fopen(tmp, 'w');  fwrite(fid, [txt(1:k(1)-1) blk]);  fclose(fid);
            c = onCleanup(@() delete(tmp));
            m = macos.Session(testCase.ModelSize);
            f = @() macos.dw_dx_multi(m, tmp, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', false);
            out = testCase.verifyWarning(f, 'macos:dw_dx_multi:emptyOPD');
            testCase.verifyEqual(nnz(out.per_field_w_nom_2d{1}), 0, ...
                'the clipped read surface must yield an empty nominal OPD');
        end

        function test_reset_xp_single_field_identity(testCase)
            % reset_xp acts ONLY through the pupil placement: on a single
            % field, a reset_xp harvest of the pupil deck must equal a
            % FROZEN harvest of the same deck whose ExitPupil was FEX'd at
            % that field beforehand (same stop-enforced chief, same FEX).
            % Non-vacuous: the committed e5hex1 carries a legacy
            % single-probe pupil (rad 2548.00) and FEX now places the
            % medial one (2523.74), so frozen-on-the-committed-deck would
            % NOT match.
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            macos.stop_obj(0, 0, 0);           % the deck's own ApStop, re-enforced
            macos.trace(macos.num_elt());
            macos.fex(1);
            tmp = [tempname '_fexed.in'];
            macos.save_rx(tmp);
            c = onCleanup(@() delete(tmp));
            oR = macos.dw_dx_multi(m, testCase.rx_path, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', true);
            oF = macos.dw_dx_multi(m, tmp, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', false);
            testCase.verifyEqual(oR.per_field_dwdx{1}, oF.per_field_dwdx{1}, ...
                'RelTol', 1e-8, 'AbsTol', 1e-15, ...
                'reset_xp at the nominal field must equal frozen-on-the-FEXed deck');
        end

        function test_no_pupil_element_refuses_before_the_loop(testCase)
            % Dave 2026-09-08: the wavefront is read at the PUPIL by default.
            % A bare-focal deck (e2e6m s3_imager_full: nElt-1 = OAPim, a
            % powered Reflector, no pupil element) must be REFUSED up front
            % with macos:dw_dx_multi:noPupil, reset_xp or not -- not read at
            % the powered optic (the one-signed dome), not warned-and-
            % stamped 'no-effect', and never redirected to the FocalPlane
            % (blind to tilt).  Supersedes test_reset_xp_no_pupil_warns_and_
            % stamps.  An explicit exit_pupil_elt at the deck's collimated
            % SharedPupil Reference (elt 23) is the sanctioned override.
            here = fileparts(mfilename('fullpath'));
            rx = fullfile(fileparts(here), 'templates', '80_end_to_end', ...
                          'e2e6m', 's3_imager_full.in');
            testCase.assumeTrue(isfile(rx), 's3_imager_full.in not present');
            m = macos.Session(testCase.ModelSize);
            for rst = [true false]
                testCase.verifyError(@() macos.dw_dx_multi(m, rx, ...
                    'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                    'dofs', (0:5).', 'reset_xp', rst), 'macos:dw_dx_multi:noPupil');
            end
            out = macos.dw_dx_multi(m, rx, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', false, 'exit_pupil_elt', 23);
            testCase.verifyEqual(out.wf_elt, 23, ...
                'an explicit collimated-pupil Reference must be honoured');
        end

        function test_reset_xp_stamps_true_on_pupiled_deck(testCase)
            % The positive: on a deck WITH an exit-pupil element at nElt-1
            % (the e5hex1 fixture's nElt-1 is a Return), FEX writes, the EP
            % moves per field, and out.reset_xp stamps logical true.
            m = macos.Session(testCase.ModelSize);
            out = macos.dw_dx_multi(m, testCase.rx_path, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, ...
                'dofs', testCase.DOFsForTest, 'reset_xp', true);
            testCase.verifyTrue(islogical(out.reset_xp) && out.reset_xp, ...
                'a pupiled deck must stamp reset_xp = true (FEX wrote)');
        end

        % NOTE (wide-field benefit gate -- DEFERRED): the intended gate --
        % reset_xp removes a per-field frame tip/tilt at a wide field,
        % matching a strict-kernel FD prediction -- could NOT be built on
        % rodgers1_stage4.  Measured empirically: reset_xp is a BIT-
        % IDENTICAL no-op on that deck at EVERY field (dwdx AND nominal-OPD
        % reldiff = 0, on-axis and at 2e-3 rad corners).
        %
        % MECHANISM (verified 2026-08-04 by probing the engine, read-only):
        % the SMACOS FEX command (macos_cmd_loop.inc ~L2618) writes the
        % pupil reference into nElt-1 ONLY when that element is a Return
        % (EltID 8) or Reference (EltID 3) surface; for any other type it
        % ABORTS without writing.  On the 4-element rodgers1 TMA nElt-1 is
        % M3, a powered Reflector (EltID 1), so FEX declines to write and
        % macos.fex just reads M3's own Vpt/Kr back (probe: elt vpt/kr
        % byte-unchanged across fex(1); xp.rad == KrElt(M3) == -2.688).
        % NOTE the latent engine gap: xp_fnd returns OK=PASS even when its
        % inner FEX aborted -- which is why the no-op is silent.  The reset
        % is therefore behaving as FROZEN here (hence bit-identical), and
        % the resetNoEffect guard below stamps that truthfully.
        %
        % The effect IS real where nElt-1 is a dedicated Return/Reference
        % surface: the e2e_pie segmented deck (nElt-1 is a Return) shifts
        % ~1.6% under reset_xp (commit 5a704fb).  The wide-field benefit
        % gate belongs on such a deck whose exit-pupil reference genuinely
        % moves per field -- the rodgers2 afocal fixtures (a dedicated
        % Reference coldstop at nElt-1, 0.5 deg box, 30x exit angles) with
        % the afocal-plane kernel (design/src/afocal_*) as the FD
        % comparator, once that stack is in-tree.
    end
end


% =====================================================================
% Helpers: the rodgers1 wide-field TMA deck (a solved design fixture
% under challenges/rodgers1) + an aperture-stripped copy for harvesting.
function p = rodgers1_deck_()
here = fileparts(mfilename('fullpath'));
p = fullfile(here, '..', 'challenges', 'rodgers1', 'rodgers1_stage4.in');
end

function sd = rodgers1_stripped_deck_()
% Persistent per-session stripped copy (ApType= -> None), mirroring
% strict_ladder_deck's strip_ap: the committed deck carries tight
% realize_apertures clips that vignette the read surface at wide field.
persistent cached
if ~isempty(cached) && isfile(cached), sd = cached; return; end
rx = rodgers1_deck_();
if ~isfile(rx), sd = ''; return; end
txt = fileread(rx);
txt = regexprep(txt, '(ApType=\s*)\S+', '$1None');
sd  = [tempname '.in'];
fid = fopen(sd, 'w');  fwrite(fid, txt);  fclose(fid);
cached = sd;
end
