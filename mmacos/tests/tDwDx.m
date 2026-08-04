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
        % it.  (The stop-less copy under examples/view_rx_demo/e5hex1.in
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
            % A deck whose read surface (nElt-1) clips the whole beam at a
            % field yields an empty per-field OPD.  dw_dx_multi must fail
            % LOUDLY (macos:dw_dx_multi:emptyOPD), not trip the opaque
            % center-tile scalar-logical assert.  rodgers1_stage4 is a
            % solved TMA carrying post-realize_apertures clip apertures
            % (M3 ApVec r=0.17 vs M1 r=1.04); the full source grid
            % overflows M3 -> 0 rays at the read surface.
            rx = rodgers1_deck_();
            testCase.assumeTrue(isfile(rx), 'rodgers1_stage4.in not reachable');
            m = macos.Session(256);
            testCase.verifyError(@() macos.dw_dx_multi(m, rx, ...
                'field_x_rad', 1e-4, 'field_y_rad', 1e-4, 'grid', '1x1', ...
                'dofs', (0:5).', 'reset_xp', false), ...
                'macos:dw_dx_multi:emptyOPD');
        end

        function test_reset_xp_single_field_identity(testCase)
            % Cheapest invariant that reset_xp acts ONLY through the field
            % loop: on a SINGLE on-axis field, FEX re-references to the same
            % chief ray the deck already points at, so reset==frozen bit-
            % identical.  (Apertures stripped so the wide-bias deck traces.)
            rx = rodgers1_stripped_deck_();
            testCase.assumeTrue(isfile(rx), 'stripped rodgers1 unavailable');
            m = macos.Session(256);
            oR = macos.dw_dx_multi(m, rx, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', true);
            oF = macos.dw_dx_multi(m, rx, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', false);
            testCase.verifyEqual(oR.per_field_dwdx{1}, oF.per_field_dwdx{1}, ...
                'single on-axis field: reset_xp must be a no-op (identity)');
        end

        function test_reset_xp_no_pupil_warns_and_stamps(testCase)
            % reset_xp=true on a bare focal deck (no exit-pupil element at
            % nElt-1 -- rodgers1's M3 is a powered Reflector) must WARN
            % (macos:dw_dx_multi:noPupil, FEX found nothing to write) and
            % stamp out.reset_xp = 'no-effect' so run_compare sees the truth.
            rx = rodgers1_stripped_deck_();
            testCase.assumeTrue(isfile(rx), 'stripped rodgers1 unavailable');
            m = macos.Session(256);
            f = @() macos.dw_dx_multi(m, rx, 'field_x_rad', 1e-4, ...
                'field_y_rad', 1e-4, 'grid', '1x1', 'dofs', (0:5).', ...
                'reset_xp', true);
            out = testCase.verifyWarning(f, 'macos:dw_dx_multi:noPupil');
            testCase.verifyEqual(out.reset_xp, 'no-effect', ...
                'no-pupil deck must stamp reset_xp = ''no-effect''');
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
% under design/rodgers1) + an aperture-stripped copy for harvesting.
function p = rodgers1_deck_()
here = fileparts(mfilename('fullpath'));
p = fullfile(here, '..', 'design', 'rodgers1', 'rodgers1_stage4.in');
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
