classdef tDesignAfocal < matlab.unittest.TestCase
%TDESIGNAFOCAL  Gates for the AFOCAL terminal in macos.design.Telescope.
%
%   ADD_EXIT_REFERENCE ends an N-mirror train in a collimated beam instead
%   of a focus: the last mirror's spacing becomes the user's INTERFACE
%   distance and the terminal is a flat `Element= Reference` normal to the
%   exit chief.  Nothing here touches the focal path -- TDESIGNTELESCOPE is
%   the gate that it did not.
%
%   THE FIXTURE IS A MERSENNE, and it is chosen because every answer is
%   known in closed form.  Two confocal parabolas (K = -1 both) with
%   f1 = R1/2 = 2 m and a CONVEX secondary f2 = -R2/2 = -0.4 m, separated by
%   f1 + f2 = 1.6 m, form a perfect afocal beam compressor:
%
%       angular magnification  M = f1/|f2| = R1/R2 = 5      exactly
%       exit beam diameter     D/M = 0.2 m                  exactly
%       marginal exit angle    u_out = 0                    exactly
%       wavefront error        0                            exactly (no
%                              spherical -- confocal parabolas are stigmatic
%                              on axis), so RUNG 1 OF THE AFOCAL LADDER MUST
%                              READ NUMERICAL ZERO, not "small"
%       exit pupil             the image of M1 through M2: object 1.6 m,
%                              f = -0.4 m -> v = -0.32 m, i.e. 0.32 m
%                              UPSTREAM of M2 -> z = -1.92 m, at pupil
%                              magnification 0.32/1.6 = 1/5
%
%   So a builder change that quietly breaks the afocal condition, the
%   magnification, the terminal's element type, or the pupil-crossing
%   construction cannot hide behind "close enough".
%
%   AND THE TERMINAL MUST BE `Reference`, NEVER `Return`.  A Return reverses
%   the ray directions at its surface while leaving the OPL untouched, so
%   the error is invisible to any piston-only check and shows up only in a
%   metric built from the exit chief -- where it read 4017 um instead of
%   359 nm (rodgers2 PACKET section 5.0).  That is gated here by reading the
%   emitted text, because it is a text-level mistake.

    properties
        deck
        t
    end

    properties (Constant)
        D  = 1.0        % entrance aperture, m
        R1 = 4.0        % primary radius (magnitude), f1 = 2 m
        R2 = 0.8        % secondary radius (magnitude), f2 = -0.4 m (convex)
        SEP = 1.6       % f1 + f2, the confocal condition
        IFACE = 1.0     % M2 -> interface plane
        M  = 5.0        % R1/R2
    end

    methods (TestClassSetup)
        function setup(tc)
            here = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(here),'mmacos_setup.m'));
            macos.init(256);
            tc.deck = [tempname '.in'];
            tc.t = tc.mersenne_();
            tc.t.build(tc.deck);
            tc.addTeardown(@() tc.rm_(tc.deck));
        end
    end

    methods
        function t = mersenne_(tc)
            t = macos.design.Telescope('family','tma', ...
                    'aperture_diameter_m', tc.D, 'wavelength_m', 1e-6, ...
                    'grid_npts', 21, 'model_size', 256);
            t.add_mirror('M1','radius_m',tc.R1,'spacing_after_m',tc.SEP, ...
                         'conic',-1);
            t.add_mirror('M2','radius_m',tc.R2,'spacing_after_m',tc.IFACE, ...
                         'conic',-1,'convex',true);
            t.add_exit_reference('ColdStop');
        end
        function rm_(~, p), if exist(p,'file'), delete(p); end, end
    end

    methods (Test)

        function test_first_order_is_the_closed_form(tc)
        %  the afocal condition, M, and the exit beam -- all exact.
            d = tc.t.spec.derived;
            tc.verifyEqual(d.u_out, 0, 'AbsTol', 1e-14, ...
                ['the marginal ray does not leave parallel: the train is ' ...
                 'not afocal, and u_out IS the afocal condition']);
            tc.verifyEqual(d.mag, tc.M, 'RelTol', 1e-12, ...
                'a Mersenne magnifies angles by exactly f1/|f2| = R1/R2');
            tc.verifyEqual(d.exit_dia, tc.D/tc.M, 'RelTol', 1e-12);
            tc.verifyEqual(d.iface_dist, tc.IFACE, 'RelTol', 1e-12, ...
                'the last mirror''s spacing_after must become the interface distance');
            tc.verifyTrue(isinf(d.EFL), ...
                'an afocal system has no finite EFL -- reporting one is a lie');
        end

        function test_terminal_is_a_reference_not_a_return(tc)
        %  the text-level gate: Return would reverse the ray directions and
        %  hide from every piston-only check.
            txt = fileread(tc.deck);
            % ^ anchored and multi-line ON PURPOSE: an unanchored 'iElt='
            % also matches inside psiElt=, and the split then lands
            % mid-element with the name already behind it.
            blocks = regexp(txt, '(?m)^\s*iElt=\s*\d+', 'split');
            last = blocks{end};
            tc.verifyTrue(contains(last, 'ColdStop'), ...
                'the last element is not the declared afocal terminal');
            tc.verifyTrue(~isempty(regexp(last, 'Element=\s*Reference', 'once')), ...
                ['the afocal terminal must be Element= Reference (got: ' ...
                 regexprep(char(regexp(last,'Element=[^\n]*','match','once')),'\s+',' ') ')']);
            tc.verifyEmpty(regexp(last, 'Element=\s*Return', 'once'), ...
                ['Element= Return REVERSES the ray directions at the surface ' ...
                 '-- OPL unchanged, so it hides from a piston-only check']);
            tc.verifyTrue(~isempty(regexp(last, 'Surface=\s*Flat', 'once')), ...
                'the interface pupil is a flat, not a sphere');
        end

        function test_the_output_is_collimated(tc)
        %  every exit ray parallel to every other -- the thing "afocal" means.
            nE = numel(tc.t.spec.elt);
            tr = macos.trace(nE);
            ri = macos.get_ray_info(tr.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);
            tc.assumeGreaterThan(nnz(ok), 100, 'too few rays survived to score');
            d  = ri.dir(:,ok);   d = d ./ vecnorm(d);
            dm = mean(d,2);      dm = dm/norm(dm);
            ang = max(acos(min(1, dm.'*d)));
            tc.verifyLessThan(ang, 1e-9, ...
                sprintf(['the exit beam is not collimated (%.3g rad of ' ...
                         'spread): confocal parabolas have no spherical ' ...
                         'aberration, so this is a layout error, not residual'], ang));
        end

        function test_rung1_wfe_is_numerical_zero(tc)
        %  the afocal ladder on a system with no aberration to find.
            L = afocal_ladder_deck(tc.deck, [0 0], 'init', false);
            tc.verifyLessThan(L(1,1), 1e-12, ...
                sprintf(['rung 1 reads %.3g m on a Mersenne, which is exact ' ...
                         '-- a nonzero answer here is the terminal, the ' ...
                         'conics, or the spacing, not the optics'], L(1,1)));
        end

        function test_traced_magnification_and_exit_pupil(tc)
        %  the 2-pass hook, against the closed-form pupil image.
            r = tc.t.exit_pupil();
            tc.verifyEqual(r.mag, tc.M, 'RelTol', 1e-5, ...
                'the TRACED angular magnification must be the parabola focal ratio');
            tc.verifyLessThan(r.miss_m, 1e-9, ...
                ['the two probe chief rays do not actually cross -- the ' ...
                 'returned pupil station is a least-squares fiction']);
            % image of M1 through M2: 1/v = 1/(-0.4) + 1/(-1.6) -> v = -0.32,
            % i.e. 0.32 m upstream of M2 along the exit chief
            tc.verifyEqual(r.dist_m, -0.32, 'AbsTol', 1e-6, ...
                'the exit pupil is not where the paraxial pupil image is');
            zM2 = -tc.SEP;
            tc.verifyEqual(r.vpt(3), zM2 - 0.32, 'AbsTol', 1e-6);
            % and the interface plane is NOT the pupil here -- the offset is
            % the design question add_exit_reference exists to expose
            tc.verifyEqual(r.offset_m, (zM2 - 0.32) - (zM2 + tc.IFACE), ...
                'AbsTol', 1e-6, ...
                'the pupil-to-interface offset is not the declared geometry');
        end

        function test_alignment_is_idempotent_on_a_coaxial_train(tc)
        %  the first-order seed is already exact when there is no bias, so
        %  pass 2 must be a no-op.  If it moves the plane here, the seed and
        %  the traced chief disagree about which way the beam leaves.
            t2 = tc.mersenne_();
            t2.build();
            a = t2.align_exit_reference();
            tc.verifyLessThan(a.moved_m, 1e-9, ...
                'chief alignment moved an unbiased coaxial terminal');
            b = t2.align_exit_reference();
            tc.verifyLessThan(b.moved_m, 1e-12, 'alignment is not idempotent');
            tc.verifyEqual(abs(a.chief_dir(3)), 1, 'AbsTol', 1e-12, ...
                'the exit chief of an unbiased coaxial train is the axis');
        end

        function test_focal_path_still_needs_derive(tc)
        %  the guard that keeps the focal path honest: without an afocal
        %  terminal the last spacing is still the derived focus.
            f = macos.design.Telescope('family','tma', ...
                    'aperture_diameter_m', tc.D, 'model_size', 256);
            f.add_mirror('M1','radius_m',8,'spacing_after_m',3);
            f.add_mirror('M2','radius_m',2,'spacing_after_m',4.5);
            f.add_mirror('M3','radius_m',4,'spacing_after_m',1.0);
            tc.verifyError(@() f.build(), ...
                'macos:design:Telescope:nmirror:lastDerive');
        end

    end
end
