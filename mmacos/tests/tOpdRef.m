classdef tOpdRef < matlab.unittest.TestCase
%TOPDREF  The OPD map's reference: chief ray vs whole-aperture mean.
%   Luis Marchen (2026-08-19) traced a spurious cross-segment piston in
%   rigid-body sensitivity columns to tracesub.F's SUBROUTINE OPD.  It
%   fills OPDMat with the chief ray's OPL as the reference only when
%   LUseChfRayIfOK is set AND LRayOK(1); otherwise it subtracts DAvgl, the
%   mean over EVERY valid ray in the aperture.  On a segmented pupil that
%   mean is one scalar shared by all segments, so perturbing ONE segment
%   moves it and pistons every OTHER segment.
%
%   LRayOK is the GEOMETRIC flag, not LRayPass: an OBSCURED chief ray
%   still serves as the reference (gated below on CassWithExitPupil,
%   whose chief is obscured at every element).  Only a geometric failure
%   -- surface miss, solver bracket -- drops the trace to the mean.
%
%   The flag was UNREACHABLE: ray_mod_init_vars inits it .FALSE.; the
%   LOAD handler (macos_cmd_loop.inc, #ifdef DESIGN_OPTIM) sets it .TRUE.
%   but MBFile6 -- in BOTH macosio.F (CLI) and smacosio.F (SMACOS, i.e.
%   every binding) -- opens with reinitialise_variables(), which puts it
%   straight back to .FALSE.; and the Rx parser had a branch only for
%   `UseChfRay4OPD= N`.  The macos-side fix adds the missing 'Y' branch
%   plus opd_ref_set/opd_ray_get, surfaced here as macos.opd_ref.
%
%   These tests assert on the OPD RESPONSE to a single-segment poke, not
%   on the flag's value: the flag is a means, the piston is the claim.
%   Fixture e5hex1.in (7 hex segments), size 128 -> SUITE_FAST.

    properties (Constant)
        RxName = 'e5hex1.in'
        Model  = 128
        NGrid  = 63       % matches templates/50_sensitivities/run_dwdx_multi
        Eval   = 12       % nElt-1, the 'exitpupil' Return
        Poke   = 3        % Seg2
        Dz     = 1e-8     % SI metres
    end

    properties
        m
        rx_path
    end

    methods (TestClassSetup)
        function setup(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            testCase.m = macos.Session(testCase.Model);
        end
    end

    methods
        function [W0, dW, v] = poke_(testCase, ref)
            % Nominal + single-segment Tz response at the exit pupil.
            testCase.m.load_rx(testCase.rx_path);
            testCase.m.set_src_sampling(testCase.NGrid);
            testCase.m.modify();
            if ~isempty(ref), testCase.m.opd_ref(ref); end
            testCase.m.trace(testCase.Eval);
            W0 = macos.opd();
            macos.perturb(testCase.Poke, 'translation', [0;0;testCase.Dz]);
            testCase.m.modify();                 % pokes need modify()
            testCase.m.trace(testCase.Eval);
            W1 = macos.opd();
            macos.perturb(testCase.Poke, 'translation', [0;0;-testCase.Dz]);
            testCase.m.modify();
            dW = W1 - W0;
            v  = (W0 ~= 0);                      % rays that reached the map
        end
    end

    methods (Test)

        function test_default_is_the_aperture_mean_reference(testCase)
        % A freshly loaded Rx references the aperture MEAN -- the map's
        % mean over valid rays is zero by construction.  (Documents the
        % engine default; the LOAD handler's .TRUE. is overwritten by
        % MBFile6's reinitialise_variables.)
            [W0, ~, v] = testCase.poke_('');
            testCase.verifyEqual(testCase.m.opd_ref(), 'mean');
            testCase.verifyLessThan(abs(mean(W0(v))), 1e-12 * rms(W0(v)), ...
                'default map must be mean-referenced');
        end

        function test_single_segment_poke_pistons_the_others(testCase)
        % THE DEFECT.  Under the mean reference, poking ONE of 7 segments
        % offsets the map everywhere.  The delta is two-valued (response
        % on the poked segment, constant elsewhere) and the unpoked
        % segments are the majority, so the MEDIAN is that constant --
        % no segment mask needed.
            [~, dW, v] = testCase.poke_('mean');
            d = dW(v);
            testCase.verifyGreaterThan(abs(median(d)), 0.05 * max(abs(d)), ...
                ['expected the documented cross-segment piston under the ' ...
                 'mean reference (>5%% of peak); if this fails the engine ' ...
                 'default changed -- see the class help']);
        end

        function test_chief_reference_removes_the_cross_segment_piston(testCase)
        % THE FIX.  Same poke, chief-ray reference: the unpoked segments
        % return EXACTLY zero.
            [~, dW, v] = testCase.poke_('chief');
            d = dW(v);
            testCase.verifyEqual(median(d), 0, 'AbsTol', 1e-18, ...
                'unpoked segments must show no piston under the chief reference');
            testCase.verifyGreaterThan(max(abs(d)), 1e-9, ...
                'the poke must still produce a response (non-vacuity)');
        end

        function test_chief_reference_restores_the_poked_segment_amplitude(testCase)
        % The contamination biases the POKED segment by the same constant,
        % so removing it must grow the peak response by exactly that much.
            [~, dM, v] = testCase.poke_('mean');
            [~, dC, ~] = testCase.poke_('chief');
            offset = median(dM(v));
            testCase.verifyEqual(max(abs(dC(v))) - max(abs(dM(v))), offset, ...
                'RelTol', 1e-6, ...
                'peak response must recover exactly the piston that was subtracted');
        end

        function test_rx_keyword_matches_the_api(testCase)
        % `UseChfRay4OPD= Y` in the Rx and macos.opd_ref('chief') must be
        % the same thing, bit for bit.  Generated here so the variant deck
        % cannot drift from its parent fixture.
            wd = tempname; mkdir(wd); cwd0 = cd(wd);
            cR = onCleanup(@() cleanup_(cwd0, wd)); %#ok<NASGU>
            t = strsplit(fileread(testCase.rx_path), newline);
            k = find(~cellfun('isempty', regexp(t, '^\s*nElt\s*=', 'once')), 1);
            testCase.assertNotEmpty(k, 'fixture must declare nElt=');
            t = [t(1:k-1), {'   UseChfRay4OPD=  Y'}, t(k:end)];
            if isempty(strtrim(t{end})), t(end) = []; end
            fid = fopen('chf.in','w'); fprintf(fid,'%s\n',t{:}); fclose(fid);

            testCase.m.load_rx('chf.in');
            testCase.verifyEqual(testCase.m.opd_ref(), 'chief', ...
                'UseChfRay4OPD= Y must select the chief-ray reference');
            testCase.m.set_src_sampling(testCase.NGrid); testCase.m.modify();
            testCase.m.trace(testCase.Eval);
            W_rx = macos.opd();

            [W_api, ~, ~] = testCase.poke_('chief');
            testCase.verifyEqual(W_api, W_rx, 'AbsTol', 0, ...
                'the Rx keyword and macos.opd_ref must agree exactly');
        end

        function test_load_rx_resets_the_setting(testCase)
        % Session state, cleared by a load: MBFile6's reinitialise_variables
        % runs after the LOAD handler.  Call opd_ref AFTER load_rx.
            testCase.m.load_rx(testCase.rx_path);
            testCase.m.opd_ref('chief');
            testCase.verifyEqual(testCase.m.opd_ref(), 'chief');
            testCase.m.load_rx(testCase.rx_path);
            testCase.verifyEqual(testCase.m.opd_ref(), 'mean', ...
                'a prescription load must reset the OPD reference');
        end

        function test_the_two_maps_differ_by_a_constant(testCase)
        % Only absolute piston moves: the two references differ by that
        % trace's mean OPD, so every mean-removed statistic is untouched.
            [Wm, ~, v] = testCase.poke_('mean');
            [Wc, ~, ~] = testCase.poke_('chief');
            d = Wc(v) - Wm(v);
            testCase.verifyLessThan(std(d), 1e-9 * rms(Wm(v)), ...
                'the maps must differ by a constant');
            testCase.verifyEqual(rms(Wc(v) - mean(Wc(v))), ...
                                 rms(Wm(v) - mean(Wm(v))), 'RelTol', 1e-12, ...
                'mean-removed RMS must be reference-independent');
        end

        function test_an_obscured_chief_ray_still_serves(testCase)
        % The branch gates on LRayOK(1) -- geometric -- NOT on LRayPass(1).
        % CassWithExitPupil's chief ray lands in the central obscuration at
        % every element, so it is flagged Obscured and excluded from the
        % map, yet its path length is defined and IS used as the reference.
        %
        % This is the case a 2026-08-07 note got wrong (it read the
        % structural nPassRays = nRay - 1 -- SUBROUTINE OPD loops from
        % iRay=2, so the chief is never written into OPDMat -- as the chief
        % being lost).  Gated here so the reading cannot drift back.
        %
        % Rx_Cass_FarField (obscured Cassegrain, model 128 like the rest of
        % this class) rather than a new fixture: its chief is obscured at
        % the exit pupil, and it is already in the shared corpus.
            m2 = testCase.m;
            m2.load_rx(rx_fixture_path('Rx_Cass_FarField.in'));
            iE = m2.num_elt() - 1;
            tr = m2.trace(iE);
            ri = macos.get_ray_info(tr.nRays);
            rs = macos.get_ray_status(tr.nRays);
            testCase.assertTrue(ri.ok_trace(1), ...
                'fixture chief must trace geometrically');
            testCase.assertFalse(ri.ok_pass(1), ...
                'fixture chief must be OBSCURED, or this test is vacuous');
            testCase.verifyEqual(double(rs.status(1)), 1, ...   % RayStat_Obscured
                'chief status must be Obscured, not a geometric failure');

            % and the reference is genuinely available: the two maps differ
            % by a non-zero CONSTANT
            m2.opd_ref('mean');  m2.trace(iE); Wm = macos.opd();
            m2.opd_ref('chief'); m2.trace(iE); Wc = macos.opd();
            v = (Wm ~= 0);
            d = Wc(v) - Wm(v);
            testCase.verifyGreaterThan(max(abs(d)), 0, ...
                'chief reference must change the map of an obscured-chief deck');
            testCase.verifyLessThan(std(d), 1e-9 * max(abs(d)), ...
                'the two maps must differ by a constant');
        end

        function test_driver_single_segment_poke_is_local_under_chief(testCase)
            % Luis 2026-09-09: one segment poked (Kr, Kc) through the
            % sensitivity DRIVER, orient xy, no PTT removal -- the OTHER
            % segments must read 0.  Under the engine-default 'mean'
            % reference they read a CONSTANT, -(N_k/N)*mean(poked column)
            % (12-15% of the poked segment's rms on e5hex1); under 'chief'
            % exactly 0.  The drivers/fronts/runner now take 'opd_ref'
            % (re-applied after every reload).  The one case 'chief' does
            % not localise is the chief ray's OWN segment (its reference
            % moves with the poke) -- recorded as a diagnostic, not gated.
            args = {'params', {'Kr','Kc'}, 'ngridpts', testCase.NGrid, ...
                    'orient', 'xy', 'remove_ptt', false};
            oc = macos.dw_dsurf(testCase.m, testCase.rx_path, 'elts', testCase.Poke, ...
                                args{:}, 'opd_ref', 'chief');
            om = macos.dw_dsurf(testCase.m, testCase.rx_path, 'elts', testCase.Poke, ...
                                args{:}, 'opd_ref', 'mean');
            testCase.verifyEqual(string(macos.opd_ref()), "mean");   % last call wins
            for c = 1:size(oc.dwds, 2)
                vc = oc.dwds(:, c);  vm = om.dwds(:, c);
                ok = isfinite(vc) & isfinite(vm);
                supp = ok & abs(vc) > 1e-9*max(abs(vc(ok)));   % the poked segment's rays
                oth  = ok & ~supp;
                % chief: the other six segments read EXACTLY zero, and the
                % support is one segment of seven
                testCase.verifyEqual(max(abs(vc(oth))), 0);
                testCase.verifyEqual(nnz(supp)/nnz(ok), 1/7, 'AbsTol', 0.02);
                % mean: the others carry one constant = -(N_k/N)*mean(poked, chief)
                cst = -nnz(supp)/nnz(ok) * mean(vc(supp));
                testCase.verifyLessThan(std(vm(oth)), 1e-10*abs(cst));
                testCase.verifyEqual(mean(vm(oth)), cst, 'RelTol', 1e-9);
                % and the two maps differ by that same constant everywhere
                % (finite-difference round-off: measured 9e-12 relative)
                testCase.verifyEqual(vm(ok) - vc(ok), cst*ones(nnz(ok),1), 'AbsTol', 1e-9*abs(cst));
                % the leak is material: >5% of the poked segment's rms
                testCase.verifyGreaterThan(abs(cst)/rms(vc(supp)), 0.05);
                testCase.log(1, sprintf('%s: mean-ref leak on the other segments = %.4f of the poked rms', ...
                    oc.channel_names{c}, abs(cst)/rms(vc(supp))));
            end
            % the chief ray's own segment (elt 1): the chief reference moves
            % with the poke, so the others read -m(chief) -- diagnostic only
            o1 = macos.dw_dsurf(testCase.m, testCase.rx_path, 'elts', 1, ...
                                'params', {'Kr'}, 'ngridpts', testCase.NGrid, ...
                                'orient', 'xy', 'remove_ptt', false, 'opd_ref', 'chief');
            v1 = o1.dwds(:, 1);  ok = isfinite(v1);
            ix = o1.indx;                                     % m2v bookkeeping (.i/.j/.size)
            w1 = nan(ix.size);  w1(sub2ind(ix.size, ix.i, ix.j)) = v1;
            % centre-segment support: the rays whose chief-ref value is NOT the
            % common outer constant
            outer = mode(round(v1(ok) / max(abs(v1(ok))) * 1e9));   % the constant, quantised
            is_out = round(v1 / max(abs(v1(ok))) * 1e9) == outer;
            testCase.log(1, sprintf(['centre segment Kr under chief: %d px on the constant %.3e ' ...
                '(%.4f of the centre-segment rms); a nominal-anchored fixed-length reference ' ...
                '(engine OPDRefRayLen branch, opd_ref_len_set) would zero it'], ...
                nnz(is_out & ok), outer*max(abs(v1(ok)))/1e9, ...
                abs(outer*max(abs(v1(ok)))/1e9)/rms(v1(ok & ~is_out))));
            testCase.verifyTrue(all(isfinite(w1(sub2ind(ix.size, ix.i, ix.j)))));
        end

        function test_bad_mode_errors(testCase)
            testCase.verifyError(@() macos.opd_ref('centroid'), ...
                'MATLAB:validators:mustBeMember');
        end
    end
end

% ---------------------------------------------------------------------------
function cleanup_(cwd0, wd)
cd(cwd0);
if exist(wd, 'dir'), rmdir(wd, 's'); end
end
