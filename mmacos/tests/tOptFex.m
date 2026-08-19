classdef tOptFex < matlab.unittest.TestCase
%TOPTFEX  Gates for the OptFEX engine fix (macos branch optfex-fix).
%
%   Before the fix, `OptFEX= Yes` was silently a no-op: msmacosio.inc parsed
%   the keyword with only the 'No' branch, so a prescription could turn
%   CALIB's per-field FEX OFF and never ON.  Without it the CALIB WFE merit
%   is the OPD to a reference surface that never leaves the nominal field,
%   which on an off-axis design measures image-displacement TILT rather than
%   wavefront error.  Evidence and the full diagnosis:
%   challenges/rodgers1/PACKET.md Addendum 5.
%
%   These are engine gates driven from mmacos, the same pattern tPolElement
%   and tVecChain use.  All three fail against the pre-fix engine.

    properties (Constant)
        MODEL = 256;
        LAM   = 1.0e-6;
    end

    properties
        deck            % 6-element pupil deck: M1 M2 M3 FP_return ExitPupil FP
        nE
        bias            % deck chief-ray field (rad)
        apst, stand
    end

    methods (TestClassSetup)
        function build_deck(tc)
            here = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(here),'mmacos_setup.m'));
            addpath(fullfile(fileparts(here),'challenges','rodgers1'));
            t = tc.pupil_telescope();
            tc.nE = numel(t.spec.elt);
            f = [tempname '.in'];  t.save(f);
            txt = regexprep(fileread(f), '(ApType=\s*)\S+', '$1None');
            fid = fopen(f,'w'); fprintf(fid,'%s',txt); fclose(fid);
            tc.deck = f;
            g = @(k) sscanf(char(regexp(txt,[k '=\s*([^\n]*)'],'tokens','once')),'%f',3);
            cdir = g('ChfRayDir');  cpos = g('ChfRayPos');  tc.apst = g('ApStop');
            tc.stand = dot(tc.apst - cpos, cdir);
            tc.bias  = [asin(cdir(1)) asin(cdir(2))];
        end
    end

    methods
        function t = pupil_telescope(tc)
        %PUPIL_TELESCOPE  Rodgers' offset TMA at EPD 4060 with an exit pupil
        %   inserted -- the smallest design that exercises the EP merit.
            P = rodgers_common();
            t = macos.design.Telescope('family','TMA', ...
                    'aperture_diameter_mm',4060, 'wavelength_m',tc.LAM, ...
                    'model_size',tc.MODEL);
            t.add_mirror('M1','radius_mm',abs(P.ROC_mm(1)),'conic',P.K_nom(1), ...
                         'spacing_after_mm',abs(P.s12_mm));
            t.add_mirror('M2','radius_mm',abs(P.ROC_mm(2)),'conic',P.K_nom(2), ...
                         'spacing_after_mm',abs(P.s23_mm));
            t.add_mirror('M3','radius_mm',abs(P.ROC_mm(3)),'conic',P.K_nom(3), ...
                         'spacing_after','derive');
            t.set_field_bias(0.5*60);
            t.build();
            t.align_focal_plane('grid',5,'span_arcmin',6);
            t.add_pupil();
        end

        function m = merit(tc, dK3)
        %MERIT  CALIB's inner WFE evaluation at the nominal field: per-field
        %   FEX at nElt-1, then OPD there.  Gate 0 (challenges/rodgers1/
        %   gate0_merit_identity.m) shows this equals the strict metric to
        %   2.7e-9.  dK3 perturbs M3's conic.
            macos.load_rx(tc.deck);
            if nargin > 1 && dK3 ~= 0
                P = rodgers_common();
                macos.set_elt_kc(3, P.K_nom(3) + dK3);
                macos.modify();
            end
            macos.stop(1);
            macos.fex(1);
            s = macos.trace(tc.nE-1);
            m = s.rmsWFE;                       % BaseUnits (m), NOT waves
        end

        function m = merit_at_field(tc, thx, thy)
        %MERIT_AT_FIELD  Same, at a field OFFSET from the deck's own bias.
            txt = fileread(tc.deck);
            bx = tc.bias(1) + thx;  by = tc.bias(2) + thy;
            d = [sin(bx); sin(by); sqrt(max(0, 1 - sin(bx)^2 - sin(by)^2))];
            p = tc.apst - tc.stand*d;
            v3 = @(v) sprintf('%.16E  %.16E  %.16E', v(1), v(2), v(3));
            s = regexprep(txt,'(ChfRayDir=\s*)[^\n]*',['$1' v3(d)]);
            s = regexprep(s,  '(ChfRayPos=\s*)[^\n]*',['$1' v3(p)]);
            f = [tempname '.in'];
            fid = fopen(f,'w'); fprintf(fid,'%s',s); fclose(fid);
            macos.load_rx(f);  macos.stop(1);  macos.fex(1);
            r = macos.trace(tc.nE-1);
            m = r.rmsWFE;
            delete(f);
        end
    end

    methods (Test)

        % NOTE on what is NOT tested here.  An assert that the emitted deck
        % literally contains "OptFEX= Yes" was tried and dropped: optimize()
        % re-emits a plain deck after the solve, so spec.rx_path no longer
        % carries the Opt block by the time a test can read it, and reaching
        % the in-solve text would mean poking private state.  The
        % BEHAVIOURAL gate below (test_offaxis_merit_is_wavefront_not_tilt)
        % is the real test of the fix -- it is what separates a working
        % OptFEX from a silently ignored one, and it fails pre-fix.

        function test_merit_is_deterministic(tc)
        % (a) Identical state evaluated twice must be BIT-identical.  FEX
        % rewrites the ExitPupil's pose on every call (macos_ops.F:60-84);
        % if any of that leaked into the next evaluation the optimiser's
        % finite differences would be built on noise.
            m1 = tc.merit();
            m2 = tc.merit();
            tc.verifyEqual(m2, m1, ...
                'merit is not reproducible on an identical state');
            tc.verifyGreaterThan(m1, 0);
        end

        function test_fd_reset_hygiene(tc)
        % (b) +delta / -delta on one DOF must return to the baseline merit
        % EXACTLY -- no EP-pose drift accumulated through FEX's rewrite.
        % This is the hygiene the LM's finite-difference Jacobian depends on.
            base = tc.merit();
            dK   = 1e-4;                       % design_optim.F's own dcc
            mp   = tc.merit(+dK);
            mm   = tc.merit(-dK);
            back = tc.merit();
            tc.verifyEqual(back, base, ...
                'merit did not return to baseline after +/-delta -- state drift');
            % non-vacuity: the perturbations must actually move the merit,
            % otherwise "returns to baseline" is trivially true.
            tc.verifyGreaterThan(abs(mp - base)/base, 1e-6, ...
                '+delta did not move the merit -- the check would be vacuous');
            tc.verifyGreaterThan(abs(mm - base)/base, 1e-6, ...
                '-delta did not move the merit -- the check would be vacuous');
        end

        function test_offaxis_merit_is_wavefront_not_tilt(tc)
        % (c) Promoted from challenges/rodgers1/fex_in_loop_check.m.  With
        % per-field FEX the exit-pupil OPD across the box is a WAVEFRONT
        % (1e-7 m band).  Without it the reference sphere is stuck at the
        % on-axis image and the same quantity is image-displacement TILT
        % (1e-3 m band) -- four orders out, and the two agree only on axis,
        % which is exactly what makes the defect easy to miss.
            h = 0.1*pi/180;                    % +/-6 arcmin box corner
            F = [0 0; h h; -h h; h -h; -h -h; 0 h; 0 -h];
            m = arrayfun(@(k) tc.merit_at_field(F(k,1),F(k,2)), 1:size(F,1));
            tc.verifyTrue(all(isfinite(m)), 'a box field lost its merit');
            tc.verifyLessThan(max(m), 1e-5, ...
                sprintf(['off-axis exit-pupil OPD reached %.3e m -- that is ' ...
                         'the no-FEX (image-displacement tilt) band, so the ' ...
                         'per-field FEX is not running'], max(m)));
            tc.verifyGreaterThan(max(m), 1e-9, ...
                'merit implausibly small -- check the deck actually traces');
            % and the off-axis fields must be the same ORDER as the on-axis
            % one; a stuck sphere makes them 4 orders larger.
            tc.verifyLessThan(max(m)/m(1), 1e2, ...
                'off-axis merit dwarfs the on-axis one -- stuck reference sphere');
        end
    end
end
