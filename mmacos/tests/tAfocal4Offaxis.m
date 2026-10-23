classdef tAfocal4Offaxis < matlab.unittest.TestCase
%TAFOCAL4OFFAXIS  Gates for the afocal4 OFF-AXIS slice (BRIEF_afocal4_offaxis).
%
%   The slice exists because the rigid-body probe could establish only that
%   the coaxial point is a LOCAL OPTIMUM UNDER PERTURBATION -- it never left
%   the coaxial basin, so it said nothing about the off-axis FAMILY.  What is
%   gated here is therefore not a result but the machinery that makes an
%   off-axis result trustworthy: that the seed is genuinely off axis, that it
%   is exactly afocal and exactly 30x, that every ray is present when a
%   number is taken, and that the two traps this slice actually fell into
%   stay closed.
%
%     1  THE SEED IS OFF AXIS, and a coaxial deck FAILS the same check.  The
%        entering beam's footprint on the primary must lie entirely to one
%        side of the parent axis; a coaxial beam straddles it.  This is the
%        brief's non-vacuity requirement and it is the test that would catch
%        a "decenter" that silently did nothing.
%     2  THE MERSENNE IS EXACT OFF AXIS.  A confocal parabola pair is afocal
%        and 30x for a beam entering ANYWHERE on it, so collimation and
%        magnification are identities of the geometry, not residuals of a
%        solve.  Asserted against the traced rays at several decenters.
%     3  THE MEASURING PASS REMOVES APERTURES; WIDENING THEM DOES NOT WORK.
%        Both halves are asserted -- the fix AND the trap -- because the
%        widened pass fails by losing every ray while still reporting a
%        plausible magnification, which is the failure mode that gets
%        believed.
%     4  A CLIPPED BEAM REPORTS AS A DIFFERENT TELESCOPE.  Pinned as a fact,
%        so the ray-count guard in front of every quoted number keeps its
%        justification.
%     5  THE MIRROR COUNT IS SET BY PACKAGING PARITY.  The closure's last
%        spacing absorbs the free tail spacing exactly, so the last powered
%        mirror's station is INDEPENDENT of t2 -- which is why the Mersenne
%        front end needs an odd element count, not a longer lever.
%     6  DESCENT_CLOSE IS SINGULAR ON AN ALREADY-CLOSED FRONT END.  Recorded
%        as an assertion rather than a comment: the marginal state after a
%        Mersenne pair is exactly the closure's target, so its numerator
%        vanishes.  A future change that "fixes" the closure by perturbing
%        this would break the recorded explanation of why the Mersenne seed
%        cannot be carried through it.
%
%   NOT asserted: any wavefront number, any comparison with the coaxial
%   floor, or whether off axis helps.  Those are the slice's RESULTS and a
%   test that pinned them would pin the answer to the question being asked.
%
%   Size 256 (SUITE_FREEFORM), ~2 min.

    properties
        P
        here
        oax
        tmp
    end

    methods (TestClassSetup)
        function setup(tc)
            h = fileparts(mfilename('fullpath'));
            run(fullfile(fileparts(h),'mmacos_setup.m'));
            tc.here = fullfile(fileparts(h),'challenges','afocal4');
            tc.oax  = fullfile(tc.here,'offaxis');
            addpath(tc.here);  addpath(tc.oax);
            addpath(fullfile(tc.here,'clearing'));
            addpath(fullfile(tc.here,'descent'));
            addpath(fullfile(tc.here,'wall'));
            macos.init(256);
            tc.P   = afocal4_params();
            tc.tmp = tempname;   mkdir(tc.tmp);
        end
    end

    methods (TestClassTeardown)
        function teardown(tc)
            if exist(tc.tmp,'dir'), rmdir(tc.tmp,'s'); end
        end
    end

    methods
        function deck = mersenne_(tc, f1, form)
        %MERSENNE_  A bare confocal parabola pair at P.M, coaxial, unedited.
            f2 = f1/tc.P.M;
            switch form
            case 'cass', sep = f1 - f2;  cvx = true;
            case 'greg', sep = f1 + f2;  cvx = false;
            end
            deck = fullfile(tc.tmp, sprintf('mers_%s_%g.in', form, f1));
            t = macos.design.Telescope('family','tma', ...
                    'aperture_diameter_m',tc.P.D, 'wavelength_m',tc.P.lambda, ...
                    'grid_npts',tc.P.ngrid, 'model_size',tc.P.model_size);
            t.add_mirror('M1','radius_m',2*f1,'spacing_after_m',sep, ...
                         'convex',false,'conic',-1);
            t.add_mirror('M2','radius_m',2*f2,'spacing_after_m',tc.P.iface, ...
                         'convex',cvx,'conic',-1);
            t.add_exit_reference('ColdStop','dist_m',tc.P.iface);
            t.build(deck);
        end

        function [ylo, yhi, nlost] = m1_footprint_(~, deck)
        %M1_FOOTPRINT_  Where the entering beam lands on the primary, in the
        %   GLOBAL decenter direction (+Y), from the ray history -- plus the
        %   count of rays that do not make it through.
        %
        %   NLOST is taken from RAY_INFO (ok_trace AND ok_pass), NOT from
        %   RAY_HIST's `ok`.  RAY_HIST's flag is GEOMETRIC validity, and an
        %   obscured ray keeps a perfectly valid intersection -- the engine
        %   sets the flux flag and leaves RayPos alone.  A count taken from
        %   it reports a fully vignetted beam as lossless, which is exactly
        %   the failure these two tests exist to catch.
            macos.load_rx(deck);
            macos.ray_hist('on');   t = macos.trace();
            H = macos.ray_hist(t.nRays);   macos.ray_hist('off');
            nE  = macos.num_elt();   off = size(H.P,3) - nE;
            ok  = logical(H.ok(:,1+off));
            V   = macos.get_elt_vpt(1);
            d   = squeeze(H.P(:,ok,1+off)) - V;
            ylo = min(d(2,:));   yhi = max(d(2,:));
            macos.load_rx(deck);
            tr = macos.trace(macos.num_elt());
            ri = macos.get_ray_info(tr.nRays);
            nlost = nnz(~(ri.ok_trace(:) & ri.ok_pass(:)));
        end
    end

    methods (Test)

        % ---- 1: the seed is off axis, and a coaxial deck is not ----------
        function test_the_seed_is_off_axis_and_a_coaxial_deck_is_not(tc)
            f1 = 1.25;
            % coaxial: the beam STRADDLES the parent axis
            dco = tc.mersenne_(f1, 'cass');
            [ylo, yhi] = tc.m1_footprint_(dco);
            tc.verifyLessThan(ylo, 0, ...
                'a coaxial beam must reach below the parent axis');
            tc.verifyGreaterThan(yhi, 0, ...
                'a coaxial beam must reach above the parent axis');

            % off axis: the beam lies ENTIRELY to one side of it
            doa = fullfile(tc.tmp,'oa_check.in');   copyfile(dco, doa);
            h = 0.55;
            offaxis_decenter(doa, h, 'quiet',true);
            [ylo2, yhi2] = tc.m1_footprint_(doa);
            tc.verifyGreaterThan(ylo2, 0, ...
                ['the off-axis seed''s entering beam must not cross the ' ...
                 'parent axis -- if it does, the deck is not off axis and ' ...
                 'every number taken from it is a coaxial number']);
            % and it moved by the amount asked for
            tc.verifyEqual((ylo2+yhi2)/2, (ylo+yhi)/2 + h, 'AbsTol',5e-3, ...
                'the footprint centre must move by exactly the decenter');
        end

        % ---- 2: the Mersenne is exact off axis ---------------------------
        function test_the_mersenne_is_exactly_afocal_and_30x_off_axis(tc)
            for f1 = [1.25 2.5]
                base = tc.mersenne_(f1, 'cass');
                for h = [0 0.55 0.80]
                    d = fullfile(tc.tmp, sprintf('mx_%g_%g.in', f1, h));
                    copyfile(base, d);
                    if h ~= 0
                        I = offaxis_decenter(d, h, 'quiet',true);
                    else
                        I = struct('traced', tc.traced_local_(d), 'nlost',0);
                    end
                    tc.verifyEqual(I.traced.collimation_urad, 0, 'AbsTol',1.0, ...
                        sprintf(['a confocal parabola pair must recollimate ' ...
                         'EXACTLY at f1 %.2f, h %.2f -- it images a parallel ' ...
                         'beam to its focus from any part of its surface'], f1, h));
                    tc.verifyEqual(I.traced.mag, tc.P.M, 'RelTol',5e-3, ...
                        sprintf('and must be exactly %gx at f1 %.2f, h %.2f', ...
                                tc.P.M, f1, h));
                end
            end
        end

        % ---- 3: remove the apertures; do NOT widen them ------------------
        function test_the_measuring_pass_removes_apertures_not_widens_them(tc)
            % Run on the N = 5 Mersenne seed, which is where the trap was
            % MEASURED and where it fires.  It does NOT fire on a bare
            % two-mirror pair: there the only fast surface is the secondary
            % and the beam reaches it close to its vertex, so an absurd
            % aperture is merely absurd.  It is the seed's third and fifth
            % elements that a widened aperture lets rays reach at radii the
            % conic cannot serve.  Gating this on the bare pair would have
            % been a test that passes for a reason unrelated to the defect.
            S = offaxis_seed(tc.P, 'cass', 'N',5, 'f1',1.25);
            S.decenter = 0;                       % emit COAXIAL, edit below
            base = fullfile(tc.tmp,'seedN5.in');
            descent_build(tc.P, S, base, 'defer_union',true, ...
                          'verify',false, 'quiet',true);
            h = 0.55;

            % the trap: displace the beam and WIDEN every aperture
            d2 = fullfile(tc.tmp,'ap_wide.in');   copyfile(base, d2);
            txt = fileread(d2);
            cp  = tc.grab3_(txt,'ChfRayPos');   st = tc.grab3_(txt,'ApStop');
            txt = tc.put3_(txt,'ChfRayPos', cp + [0;h;0]);
            txt = tc.put3_(txt,'ApStop',    st + [0;h;0]);
            txt = regexprep(txt,'(?m)(^\s*ApVec=\s*)[^\n]*', ...
                    ['$1' sprintf('%.16E  %.16E  %.16E', 4*(h+1), 0, 0)]);
            txt = regexprep(txt,'(?m)(^\s*ApType=\s*)None','$1  Circular');
            fid = fopen(d2,'w');  fprintf(fid,'%s',txt);  fclose(fid);
            [~,~,nlost_wide] = tc.m1_footprint_(d2);
            tc.verifyGreaterThan(nlost_wide, 0, ...
                ['NON-VACUITY: the widened-aperture pass must LOSE rays.  ' ...
                 'If this ever passes cleanly the trap is gone and the ' ...
                 'ApType=None machinery is no longer earning its keep']);

            % the fix: ApType=None, then apertures fitted to the footprints
            d1 = fullfile(tc.tmp,'ap_none.in');   copyfile(base, d1);
            I = offaxis_decenter(d1, h, 'quiet',true);
            tc.verifyEqual(I.nlost, 0, ...
                ['with the apertures removed for the measurement and then ' ...
                 'FITTED to the measured footprints, the decentered beam ' ...
                 'must trace complete -- otherwise the footprints were ' ...
                 'fitted to their own vignetting']);
            tc.verifyLessThan(I.nlost, nlost_wide, ...
                'and the fix must beat the trap it replaced');
        end

        % ---- 4: a clipped beam reports as a different telescope ----------
        function test_a_clipped_beam_reports_as_a_different_telescope(tc)
            % The coaxial apertures centre on the vertex, so a decentered
            % beam walks off the primary.  The survivors still trace, still
            % collimate, and still report a magnification -- a WRONG one.
            base = tc.mersenne_(2.5, 'cass');
            d = fullfile(tc.tmp,'clip.in');   copyfile(base, d);
            txt = fileread(d);
            cp  = tc.grab3_(txt,'ChfRayPos');   st = tc.grab3_(txt,'ApStop');
            txt = tc.put3_(txt,'ChfRayPos', cp + [0;0.8;0]);
            txt = tc.put3_(txt,'ApStop',    st + [0;0.8;0]);
            fid = fopen(d,'w');  fprintf(fid,'%s',txt);  fclose(fid);

            s = tc.traced_local_(d);
            [~,~,nlost] = tc.m1_footprint_(d);
            tc.verifyGreaterThan(nlost, 0, ...
                'the clipped case must actually lose rays');
            tc.verifyGreaterThan(abs(s.mag/tc.P.M - 1), 0.10, ...
                ['a clipped beam must report a magnification that is WRONG ' ...
                 'by more than 10 %% -- this is why every quoted number in ' ...
                 'this slice carries its ray count']);
        end

        % ---- 5: the mirror count is set by packaging parity --------------
        function test_the_packaging_station_is_independent_of_the_tail_spacing(tc)
            S0 = offaxis_seed(tc.P, 'cass', 'N',4, 'f1',1.25);
            z = nan(1,3);   k = 0;
            for t2 = [0.5 1.5 3.0]
                k = k + 1;
                S = S0;   S.t(2) = t2;
                C = descent_close(tc.P, struct('N',4,'R',S.R,'convex',S.convex, ...
                        't',S.t,'iface',S.iface,'K',S.K), ...
                        'window',[-1.5 9], 'npts',241);
                tc.assumeTrue(isfield(C,'found') && C.found, ...
                    'the N = 4 Mersenne closure must have a root to compare');
                z(k) = C.behind_m1;
            end
            tc.verifyEqual(max(abs(z - z(1))), 0, 'AbsTol',1e-9, ...
                ['the closure''s last spacing absorbs the free tail spacing ' ...
                 'exactly, so the packaging station is a CONSTANT of the ' ...
                 'front end -- which is why an N = 4 Mersenne cannot be ' ...
                 'packaged by lengthening the train, only by changing the ' ...
                 'parity']);
            tc.verifyLessThan(z(1), 0, ...
                ['and that constant is on the wrong side of the primary ' ...
                 'for an N = 4 Mersenne front end']);
        end

        % ---- 6: the closure is singular on an already-closed front end ---
        function test_the_closure_is_singular_on_a_mersenne_front_end(tc)
            f1 = 1.25;   f2 = f1/tc.P.M;
            % march the marginal ray through the Mersenne pair by hand, in
            % the closure's own thin-lens convention
            y = tc.P.D/2;   u = 0;
            R = [2*f1 2*f2];   cvx = [false true];   t = [f1 - f2, 0];
            for k = 1:2
                phi = 2/R(k);   if cvx(k), phi = -phi; end
                u = u - y*phi;   y = y + t(k)*u;
            end
            tc.verifyEqual(u, 0, 'AbsTol',1e-12, ...
                'a Mersenne pair leaves the marginal ray COLLIMATED');
            tc.verifyEqual(y, (tc.P.D/2)/tc.P.M, 'RelTol',1e-12, ...
                'and at exactly the exit height the specification asks for');
            % which is precisely DESCENT_CLOSE's b = (yout - ym)/u2 numerator
            tc.verifyEqual((tc.P.D/2)/tc.P.M - y, 0, 'AbsTol',1e-12, ...
                ['so the closure''s magnification lever has a ZERO numerator ' ...
                 'on this front end.  It is not broken -- it is being asked ' ...
                 'to solve a problem that is already solved, which is why ' ...
                 'the Mersenne seed is measured directly and not carried ' ...
                 'through DESCENT_CLOSE']);
        end
    end

    methods
        function s = traced_local_(~, deck)
            tk = regexp(fileread(deck),'(?m)^\s*Aperture=\s*([^\n]*)','tokens','once');
            Dap = sscanf(strrep(tk{1},'D','E'),'%f',1);
            macos.load_rx(deck);
            tr = macos.trace(macos.num_elt());
            ri = macos.get_ray_info(tr.nRays);
            ok = ri.ok_trace(:) & ri.ok_pass(:);   ok(1) = false;
            dd = ri.dir(:,ok);   dd = dd ./ vecnorm(dd);
            dm = mean(dd,2);     dm = dm/norm(dm);
            q  = ri.pos(:,ok) - mean(ri.pos(:,ok),2);
            q  = q - dm*(dm.'*q);
            dia = 2*max(vecnorm(q));
            s = struct('exit_dia',dia, 'mag',Dap/max(dia,realmin), ...
                       'collimation_urad', max(acos(min(1, dm.'*dd)))*1e6);
        end
        function v = grab3_(~, txt, key)
            tk = regexp(txt, ['(?m)^\s*' key '=\s*([^\n]*)'], 'tokens', 'once');
            v  = sscanf(strrep(tk{1},'D','E'), '%f', 3);   v = v(:);
        end
        function txt = put3_(~, txt, key, v)
            txt = regexprep(txt, ['(?m)(^\s*' key '=\s*)[^\n]*'], ...
                    ['$1' sprintf('%.16E  %.16E  %.16E', v(1),v(2),v(3))], 'once');
        end
    end
end
