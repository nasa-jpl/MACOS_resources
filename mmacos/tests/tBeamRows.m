classdef tBeamRows < matlab.unittest.TestCase
%TBEAMROWS  CALIB beam rows (chief direction / position) on ANY target,
%   per-field position targets, the ray centroid, and the row bookkeeping.
%
%   Engine change 2026-10-03 (design_optim.F, dyson5 addendum 14 item 4):
%     * the beam rows (OptBeamDir= / OptBeamPos= / OptBeamSize=) used to be
%       scored ONLY under OptTarget= BEAM -- a WFE or SPOT target could not
%       carry a telecentricity or image-position condition;
%     * the BEAM-only path reset its row offset to 1 for every field, so a
%       multi-field beam solve scored the LAST field alone;
%     * one target position served every field (OptBeamPos=); a spectrometer
%       smile / keystone solve needs one per field (OptBeamPosFov=);
%     * the position was the chief ray's; the centroid of the passing rays
%       is what a detector sees on a comatic field (OptBeamCentroid= Y).
%   The value, derivative and linear-solver paths now write the rows through
%   one helper (beam_rows_), with a weight against the target rows
%   (OptBeamWt=).  API: calib_set_beam / calib_set_beam_pos_fov /
%   calib_set_beam_wt.
%
%   CALIB convention met here: FIELD 1 is the source as CURRENTLY set (the
%   handler copies ChfRayDir/Pos into opt_fov(:,:,1) before nls_optim_dvr),
%   so a test that moved the source to measure must put it back on field 1.
%   Fixture Rx_BeamRows.in: a collimated beam on an f = 1 m paraboloid in
%   metres, focal plane at the focus, SPOT target, mirror TIP/TILT free.
%   Non-vacuity (pre-fix engine, measured on the same fixture): test 1's
%   OptBeamDir= under a SPOT target sized zero beam rows, so the mirror never
%   tilted (direction error stayed 2e-3); test 2's per-field keyword did not
%   exist (parser catch-all), and the BEAM path scored one field.
    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_BeamRows.in'
        Tol       = 1e-9       % the LM runs to dopt_tol 1e-12 on 1 m rays
    end
    methods (Access = private)
        function p = variant(tc, wd, name, edits)
            % Text edits on the base deck: edits is {pattern, replacement; ...}
            % applied to the (trimmed) line that starts with pattern; a
            % replacement of [] deletes the line; a pattern 'APPEND_FP:'
            % appends the replacement after the FP's OptBeamDir line.
            L = splitlines(string(fileread(rx_fixture_path(tc.RxName))));
            for k = 1:size(edits, 1)
                pat = edits{k, 1};  rep = edits{k, 2};
                if startsWith(pat, "APPEND_FP:")
                    i = find(startsWith(strtrim(L), "OptBeamDir=") | startsWith(strtrim(L), "OptBeamPos="), 1);
                    tc.assertNotEmpty(i, "no OptBeam line to append after");
                    L = [L(1:i); string(rep); L(i+1:end)];
                    continue
                end
                i = find(startsWith(strtrim(L), pat));
                tc.assertNotEmpty(i, "no line starts with " + pat);
                if isempty(rep), L(i) = []; else, L(i) = string(rep); end
            end
            p = fullfile(wd, name);
            fid = fopen(p, 'w');  fprintf(fid, '%s\n', L);  fclose(fid);
        end
        function [p, d, c] = chief_at_fp(~, m, dir)
            % Chief ray position / direction at the FP for source direction
            % dir, and the centroid of the passing rays there.
            m.set_src_fov('src_dir', dir(:));
            tr = m.trace(2);
            ri = macos.get_ray_info(tr.nRays);
            p = ri.pos(:, 1);  d = ri.dir(:, 1);
            ok = ri.ok_trace(:) & ri.ok_pass(:);  ok(1) = false;
            c = mean(ri.pos(:, ok), 2);
        end
    end
    methods (Test)
        function test_direction_rows_ride_on_a_spot_target(tc)
            % SPOT target + OptBeamDir= in the FP block (the Rx keyword path):
            % CALIB must tilt the mirror until the chief ray leaves the FP
            % along the requested direction, with the spot still tight.
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_fixture_path(tc.RxName));
            target = [0; 2e-3; -1];  target = target / norm(target);
            [~, d0] = tc.chief_at_fp(m, [0 0 1]);
            tc.assertGreaterThan(norm(d0 - target), 1e-3, 'the seed must start off the target direction');
            % The spot rows (coma grows with the tilt) and the direction rows
            % compete; at weight 1 the least-squares compromise leaves 2.6e-8
            % of direction error (measured), so weight the beam rows.
            m.calib_set_tol(1e-12);  m.calib_set_beam_wt(1e6);
            r = m.calib();
            tc.verifyTrue(r.converged, sprintf('CALIB must converge (rtn_flag %d)', r.rtn_flag));
            [~, d1] = tc.chief_at_fp(m, [0 0 1]);
            tc.verifyLessThan(norm(d1 - target), tc.Tol, ...
                sprintf('the chief direction at the FP must reach the target: |d - t| = %.3e', norm(d1 - target)));
        end
        function test_per_field_position_targets_score_every_field(tc)
            % BEAM target, two fields, one position target PER FIELD set via
            % the API, FP piston free: the two targets are the two chief
            % positions on the FP moved 0.1 m further along the chief rays
            % (exactly reachable by one piston), so BOTH must be met -- the
            % old single-target / last-field-only code could not.
            wd = tempname;  mkdir(wd);  cl = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            th = 1e-3;
            rx = tc.variant(wd, 'two_fields.in', { ...
                "ChfRayDir=",   "        ChfRayDir=  0  1.0E-03  1"; ...
                "OptTarget=",   "        OptTarget=  BEAM"; ...
                "VarElt=  1 1", "           VarElt=  0 0 0 0 0 0 0 0"; ...
                "OptBeamDir=",  "       OptBeamPos=  0  0  -1"; ...
                "ApType=  None", "           VarElt=  0 0 0 0 0 1 0 0" + newline + "           ApType=  None"});
            % the second field: OptChfRayDir/Pos pairs (the first pair IS the
            % deck's ChfRayDir/ChfRayPos, which the parser counts as field 1)
            L = splitlines(string(fileread(rx)));
            i = find(startsWith(strtrim(L), "ChfRayPos="), 1);
            L = [L(1:i); "     OptChfRayDir=  0  -1.0E-03  1"; "     OptChfRayPos=  0  0  -0.5"; L(i+1:end)];
            fid = fopen(rx, 'w');  fprintf(fid, '%s\n', L);  fclose(fid);
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx);
            dirs = {[0 th 1], [0 -th 1]};
            dz = -0.1;  P = zeros(3, 2);
            for k = 1:2
                [p, d] = tc.chief_at_fp(m, dirs{k});
                P(:, k) = p + (dz / d(3)) * d;        % where the chief crosses z = -1 + dz
            end
            tc.assertGreaterThan(abs(P(2, 1) - P(2, 2)), 1e-4, 'the two fields must have distinct targets');
            % CALIB takes its FIELD 1 from the CURRENT source (macos_cmd_loop.inc
            % sets opt_fov(:,:,1) = ChfRayDir/Pos before nls_optim_dvr), so put
            % the source back on field 1 -- the nominal loop left it on field 2.
            m.set_src_fov('src_dir', dirs{1}(:));
            m.calib_set_beam('pos', 2, [0 0 -1]);     % enable the position rows at the FP
            m.calib_set_beam_pos_fov(P);              % one target per field
            m.calib_set_tol(1e-12);  m.calib_set_iter(20);
            r = m.calib();
            tc.verifyTrue(r.converged, sprintf('CALIB must converge (rtn_flag %d)', r.rtn_flag));
            for k = 1:2
                p = tc.chief_at_fp(m, dirs{k});
                tc.verifyLessThan(norm(p - P(:, k)), tc.Tol, ...
                    sprintf('field %d: chief position must meet ITS target: |p - t| = %.3e', k, norm(p - P(:, k))));
            end
        end
        function test_centroid_option_scores_the_ray_centroid_not_the_chief(tc)
            % One comatic field (20 mrad on the F/5 paraboloid), BEAM target,
            % mirror TIP free, position target = the nominal CHIEF position.
            % Chief scoring: already met, the mirror does not move.  Centroid
            % scoring (OptBeamCentroid= Y, the Rx path): the mirror must tilt
            % until the CENTROID of the passing rays sits on the target, and
            % the chief is then off it by the coma offset.
            wd = tempname;  mkdir(wd);  cl = onCleanup(@() rmdir(wd, 's')); %#ok<NASGU>
            th = 2e-2;  dir = [0 th 1];
            base = { "ChfRayDir=",   "        ChfRayDir=  0  2.0E-02  1"; ...
                     "OptTarget=",   "        OptTarget=  BEAM"; ...
                     "VarElt=  1 1", "           VarElt=  1 0 0 0 0 0 0 0"; ...
                     "OptBeamDir=",  "       OptBeamPos=  0  0  -1"};
            rx_chief = tc.variant(wd, 'chief.in', base);
            rx_cen   = tc.variant(wd, 'centroid.in', [base; {"APPEND_FP:", "  OptBeamCentroid=  Y"}]);
            m = macos.Session(tc.ModelSize);
            m.load_rx(rx_chief);
            [p0, ~, c0] = tc.chief_at_fp(m, dir);
            off = norm(c0 - p0);
            tc.assertGreaterThan(off, 1e-6, sprintf('the field must be comatic enough to separate centroid and chief (%.3e m)', off));
            % chief scoring: target = the nominal chief -> nothing to do
            m.calib_set_beam('pos', 2, p0);  m.calib_set_tol(1e-12);
            r = m.calib();
            tc.verifyTrue(r.converged, 'chief run must converge');
            p1 = tc.chief_at_fp(m, dir);
            tc.verifyLessThan(norm(p1 - p0), tc.Tol, 'chief scoring on an already-met target must not move the mirror');
            % centroid scoring (Rx keyword): the centroid must reach the same target
            m.load_rx(rx_cen);
            m.calib_set_beam('pos', 2, p0);  m.calib_set_tol(1e-12);
            r = m.calib();
            tc.verifyTrue(r.converged, 'centroid run must converge');
            [p2, ~, c2] = tc.chief_at_fp(m, dir);
            tc.verifyLessThan(norm(c2 - p0), 1e-7, ...
                sprintf('the ray CENTROID must reach the target: |c - t| = %.3e (coma offset %.3e)', norm(c2 - p0), off));
            tc.verifyGreaterThan(norm(p2 - p0), 0.5 * off, ...
                sprintf('the chief must now be off the target by about the coma offset: %.3e vs %.3e', norm(p2 - p0), off));
            % the same through the API switch, from the chief deck
            m.load_rx(rx_chief);
            m.calib_set_beam('pos', 2, p0);  m.calib_set_beam_wt(1, true);  m.calib_set_tol(1e-12);
            r = m.calib();
            tc.verifyTrue(r.converged, 'API centroid run must converge');
            [~, ~, c3] = tc.chief_at_fp(m, dir);
            tc.verifyLessThan(norm(c3 - c2), 1e-9, 'the API centroid switch must reproduce the Rx keyword');
        end
    end
end
