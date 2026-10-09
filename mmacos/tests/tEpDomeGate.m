classdef tEpDomeGate < matlab.unittest.TestCase
%TEPDOMEGATE  EP-dome ruling gate (Dave 2026-09-08).
%   The sensitivity OPD is read at an EXIT-PUPIL reference -- never the
%   tilt-blind FocalPlane.  wf_elt_auto ERRORS on a pupil-less powered
%   nElt-1 (macos:dw_dx:noPupil); a placed pupil gives the textbook
%   bipolar tilt RAMP.  These lift the shape metrics from the runnable
%   tool mmacos/tools/ep_dome_probe/dome_probe.m; the record is
%   macos/REPORT_ep_dome_review.md.
%
%   Non-vacuity: the same reads at the FocalPlane FAIL the shape gates
%   (a segment tilt reads as a segment PISTON, a global field tilt reads
%   ~0), which is exactly why the FP-read branch was rejected.
%
%   Fixtures (committed, self-contained):
%     s3_imager_full.in  -- bare-focal segmented TMA (powered OAPim at
%                           nElt-1, terminal FocalPlane, NO pupil).
%     s3_imager_pupil.in  -- the same with the add_pupil pair inserted
%                           (flat Return @ image, Return/Conic ExitPupil
%                           at nElt-1, FEX-placed).  Regenerate via
%                           tools/ep_dome_probe/make_pupil_deck.py + FEX.
%   Model 256 -- own batch line in run_mmacos_tests.sh.

    properties (Constant)
        ModelSize = 256
        BareRel   = fullfile('templates','80_end_to_end','e2e6m','s3_imager_full.in')
        PupilRel  = fullfile('templates','80_end_to_end','e2e6m','s3_imager_pupil.in')
        Alpha     = 1e-6      % segment Rx tilt (rad)
        Ftilt     = 1e-6      % global field tilt (rad)
        Seg       = 8         % off-axis segment: largest FP piston => sharpest FP-vs-EP contrast
        Dpup      = 6         % pupil diameter (m) for the field-tilt estimate D*theta/sqrt(12)
    end

    properties
        bare_path
        pup_path
    end

    methods (TestClassSetup)
        function setup(tc)
            mmroot = fileparts(fileparts(mfilename('fullpath')));  % tests/ -> mmacos/
            tc.bare_path = fullfile(mmroot, tc.BareRel);
            tc.pup_path  = fullfile(mmroot, tc.PupilRel);
            tc.assumeTrue(exist(tc.bare_path,'file')==2, 's3_imager_full.in fixture missing');
            tc.assumeTrue(exist(tc.pup_path,'file')==2,  's3_imager_pupil.in fixture missing');
            macos.init(tc.ModelSize);
        end
    end

    methods (Test)
        function test_bare_focal_powered_nElt1_errors_noPupil(tc)
            % A powered nElt-1 with no placed pupil has no valid wavefront
            % reference -- the supervisor must REFUSE, not silently read a
            % tilt-blind (or dome) surface and return a Jacobian column.
            m = macos.Session(tc.ModelSize);
            tc.verifyError(@() macos.dw_dx(m, tc.bare_path, ...
                'dofs', 1, 'elts', tc.Seg, 'delta', 1e-8), ...
                'macos:dw_dx:noPupil');
        end

        function test_placed_pupil_auto_reads_at_nElt_minus_1(tc)
            % With the add_pupil pair in place, auto-select reads at the
            % unpowered ExitPupil Return (nElt-1) with no error.
            m = macos.Session(tc.ModelSize);
            out = macos.dw_dx(m, tc.pup_path, ...
                'dofs', 1, 'elts', tc.Seg, 'delta', 1e-8);
            nE = macos.num_elt();
            tc.verifyEqual(out.wf_elt, nE - 1, ...
                'auto-selected read surface must be the placed EP at nElt-1');
        end

        function test_segment_tilt_is_a_ramp_at_the_pupil(tc)
            % Read a segment Rx tilt at the EP sphere: it must be a bipolar
            % PLANE RAMP (near-zero mean over the segment footprint), not a
            % piston.  Footprint = the piston-poke flat-top.
            EP = tc.load_(tc.pup_path) - 1;
            W0 = tc.rd_(EP);
            Wt = tc.pokeread_(tc.pup_path, 'rotation',    [tc.Alpha;0;0], EP) - W0;
            Wp = tc.pokeread_(tc.pup_path, 'translation', [0;0;1e-7],     EP) - W0;
            [v, rampfrac, meanratio] = tc.shape_(Wt, Wp);
            tc.verifyGreaterThan(rampfrac, 0.90, ...
                'segment tilt at the EP is not a plane ramp (>90% of variance)');
            tc.verifyLessThan(meanratio, 0.5, ...
                'segment tilt at the EP reads as a piston (mean/rms), not a bipolar ramp');
            tc.verifyGreaterThan(rms_(v), 1e-7, 'EP tilt ramp implausibly small');
            tc.verifyLessThan(rms_(v),    2e-6, 'EP tilt ramp implausibly large');
        end

        function test_focal_plane_read_is_tilt_blind_segment(tc)
            % NON-VACUITY: the SAME segment tilt read at the FocalPlane is a
            % PISTON (mean ~ rms), the wrong Jacobian column the rejected
            % FP-read branch would have returned.
            FP = tc.load_(tc.bare_path);           % nElt = the FocalPlane
            W0 = tc.rd_(FP);
            Wt = tc.pokeread_(tc.bare_path, 'rotation',    [tc.Alpha;0;0], FP) - W0;
            Wp = tc.pokeread_(tc.bare_path, 'translation', [0;0;1e-7],     FP) - W0;
            [~, ~, meanratio] = tc.shape_(Wt, Wp);
            tc.verifyGreaterThan(meanratio, 0.9, ...
                ['FP segment-tilt read is not a piston -- the rejected ' ...
                 'FP-read branch would look valid (gate is vacuous)']);
        end

        function test_global_field_tilt_is_a_tilt_at_the_pupil(tc)
            % A global field tilt must appear as a tilt of the expected rms
            % at the EP, and be BLIND (~0) at the FocalPlane.
            EP = tc.load_(tc.pup_path) - 1;
            W0e = tc.rd_(EP);
            ep_rms = rms_(finitenz_(tc.tiltread_(tc.pup_path, EP) - W0e));
            FP = tc.load_(tc.bare_path);
            W0f = tc.rd_(FP);
            fp_rms = rms_(finitenz_(tc.tiltread_(tc.bare_path, FP) - W0f));
            expect = tc.Ftilt * tc.Dpup / sqrt(12);
            tc.verifyLessThan(abs(ep_rms - expect)/expect, 0.30, ...
                sprintf('EP field-tilt rms %.3e vs expected %.3e (>30%%)', ep_rms, expect));
            tc.verifyLessThan(fp_rms, 0.05*ep_rms, ...
                sprintf('FP field-tilt read %.3e is not blind vs EP %.3e', fp_rms, ep_rms));
        end
    end

    methods   % helpers
        function nE = load_(~, deck), macos.load_rx(deck); nE = macos.num_elt(); end
        function W  = rd_(~, e),      macos.trace(e); W = macos.opd(); end
        function W  = pokeread_(tc, deck, kind, vec, e)
            macos.load_rx(deck);  macos.perturb(tc.Seg, kind, vec);  W = tc.rd_(e);
        end
        function W  = tiltread_(tc, deck, e)
            macos.load_rx(deck);
            macos.set_src_fov('src_dir', [sin(tc.Ftilt); 0; cos(tc.Ftilt)]);
            W = tc.rd_(e);
        end
        function [v, rampfrac, meanratio] = shape_(~, Wtilt, Wpist)
            % plane-fit the tilt map inside the piston-poke footprint
            F = isfinite(Wpist) & (Wpist ~= 0) & (abs(Wpist) > 0.5*max(abs(Wpist(:))));
            [ii, jj] = find(F);  A = [ones(size(ii)) ii jj];
            v = Wtilt(F);
            c = A \ v;  ramp = A(:,2:3) * c(2:3);
            rampfrac  = norm(ramp - mean(ramp)) / max(norm(v - mean(v)), 1e-30);
            meanratio = abs(mean(v)) / max(rms_(v), 1e-30);
        end
    end
end

function r = rms_(v),      r = sqrt(mean(v(:).^2)); end
function w = finitenz_(W), w = W(isfinite(W) & W ~= 0); end
