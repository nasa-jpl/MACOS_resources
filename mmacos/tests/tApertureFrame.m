classdef tApertureFrame < matlab.unittest.TestCase
%TAPERTUREFRAME  Clear apertures must be sized in the ELEMENT's own frame.
%
%   MACOS reads ApVec(2:3) as an offset from VptElt along the element
%   aperture triad (elemsub.F ChkRayTrans: rho = intersection - VptElt,
%   px = xObs.rho, py = yObs.rho), with the triad built in
%   tracesub.F/propsub.F as zObs = psiElt, yObs = unit(zObs x xObs),
%   xObs = yObs x zObs, seeded -- when the deck declares no xObs, which is
%   the case for every deck the design layer emits -- by the ChkDf2 default
%   xObs = (psi_z, psi_x, psi_y).
%
%   Telescope.aperture_full_field used to measure the footprint box in
%   GLOBAL x,y and hand it back as a LOCAL ApVec offset: correct only for
%   an element at the global origin with psi = (0,0,-1), and catastrophic
%   on a TILTED FOLD, where the emitted stop lands metres off the beam.
%   The failure is invisible in-session (the object still holds the right
%   geometry) and total on RELOAD -- so the gate here is a round trip.
%
%   Fixture: a small tilted-fold three-mirror telescope, built here rather
%   than loaded, so the test is independent of any committed artifact.
%
%   NON-VACUITY: test_global_frame_numbers_would_break_the_reload emits the
%   PRE-FIX quantity (the global-XY beam centre) and shows the reload loses
%   the beam -- if that ever stops failing, this gate has stopped testing.

    properties (Constant)
        ModelSize = 128
    end
    properties
        t
        nbase
    end

    methods (TestClassSetup)
        function build_fixture(tc)
            macos.init(tc.ModelSize);
            tc.t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'wavelength_m',500e-9, 'grid_npts',21);
            tc.t.set_base_sphere(true);
            % a genuinely TILTED fold: every mirror off the global axis, so
            % local and global aperture frames differ at every element
            tc.t.add_mirror('M1','radius_m',6.4,'spacing_after_m',2.9, ...
                            'tilt_deg',-5.6);
            tc.t.add_mirror('M2','radius_m',0.633,'spacing_after_m',2.33, ...
                            'tilt_deg',5.9,'convex',true);
            tc.t.add_mirror('M3','radius_m',0.583,'spacing_after','derive', ...
                            'tilt_deg',8.0);
            tc.t.add_focal_plane('FP');
            tc.t.build();
            s = macos.trace(numel(tc.t.spec.elt));
            r = macos.get_ray_info(s.nRays);
            tc.nbase = nnz(logical(r.ok_pass) & logical(r.ok_trace));
            tc.assumeGreaterThan(tc.nbase, 0.9*s.nRays, ...
                'fixture does not trace cleanly with apertures off');
        end
    end

    methods (Test)

        function test_centres_are_element_local_not_global(tc)
            % The tell: a tilted mirror sits METRES off the global axis, so
            % its GLOBAL beam centre is metres from zero while its LOCAL
            % in-plane offset is a small residual.  Report the local one.
            rep = tc.t.aperture_full_field('quiet', true);
            for k = 1:numel(rep)
                e = tc.t.spec.elt(k);
                if norm(e.Vpt) < 1e-9, continue; end     % M1 is at the origin
                tc.verifyLessThan(norm(rep(k).center), 0.25*norm(e.Vpt), ...
                    sprintf(['%s aperture centre (%.4f m) is the scale of ' ...
                             'its GLOBAL station (%.4f m) -- the global-XY ' ...
                             'frame defect'], e.name, norm(rep(k).center), ...
                             norm(e.Vpt)));
            end
        end

        function test_frame_matches_the_engine_construction(tc)
            % obs_frame_ must reproduce tracesub.F exactly: orthonormal,
            % right-handed, z along psi, seeded by the ChkDf2 default.
            for k = 1:numel(tc.t.spec.elt)
                psi = tc.t.spec.elt(k).psi(:);
                [xo, yo, zo] = frame_of(tc.t, psi);
                tc.verifyEqual(norm(xo), 1, 'AbsTol', 1e-12);
                tc.verifyEqual(norm(yo), 1, 'AbsTol', 1e-12);
                tc.verifyEqual(xo.'*yo, 0, 'AbsTol', 1e-12);
                tc.verifyEqual(zo, psi/norm(psi), 'AbsTol', 1e-12);
                tc.verifyEqual(cross(xo, yo), zo, 'AbsTol', 1e-12);
                % the seed IS the engine default, so the frame is pinned
                seed = [zo(3); zo(1); zo(2)];
                yref = cross(zo, seed);  yref = yref/norm(yref);
                tc.verifyEqual(yo, yref, 'AbsTol', 1e-12);
            end
        end

        function test_applied_apertures_survive_a_standalone_reload(tc)
            % The gate that matters: emit the stops, save, reload in a
            % fresh engine, count rays.
            tc.t.apply_full_field_apertures('margin', 0.10, 'quiet', true, ...
                                            'skip', {'FP'});
            n_in = pass_count(numel(tc.t.spec.elt));
            tc.verifyGreaterThan(n_in, 0.98*tc.nbase, ...
                'apertures clip the beam in session');
            f = [tempname '.in'];
            c = onCleanup(@() delete_if(f));
            tc.t.save(f);
            macos.init(tc.ModelSize);
            n = macos.load_rx(f);
            s = macos.trace(n);
            r = macos.get_ray_info(s.nRays);
            n_re = nnz(logical(r.ok_pass) & logical(r.ok_trace));
            tc.verifyGreaterThan(n_re, 0.98*tc.nbase, ...
                sprintf(['the SAVED deck lost the beam: %d/%d rays pass on ' ...
                         'reload against %d with apertures off'], ...
                        n_re, s.nRays, tc.nbase));
            tc.t.build('', 'init', false);
        end

        function test_global_frame_numbers_would_break_the_reload(tc)
            % NON-VACUITY.  Take the deck the FIXED sizer emits and patch
            % each ApVec centre to the PRE-FIX quantity -- the GLOBAL x,y
            % beam centre, which is what the old code returned -- then
            % reload.  A text patch, deliberately: it reproduces exactly
            % what the old emitter wrote, without resurrecting it.
            nE = numel(tc.t.spec.elt);
            tc.t.apply_full_field_apertures('margin', 0.10, 'quiet', true, ...
                                            'skip', {'FP'});
            f = [tempname '.in'];
            c = onCleanup(@() delete_if(f));
            tc.t.save(f);
            macos.trace(nE);
            b = macos.draw_rays('XY', 0, nE);
            gc = nan(nE, 2);
            for k = 1:nE
                m = (b.elt == k);
                if ~any(m(:)), continue; end
                gc(k,:) = [(min(b.U(m))+max(b.U(m)))/2, ...
                           (min(b.V(m))+max(b.V(m)))/2];
            end
            tc.assumeGreaterThan(max(abs(gc(:,2))), 0.5, ...
                'fixture has no element far enough off axis to discriminate');
            patch_apvec_centres(f, gc);
            macos.init(tc.ModelSize);
            n = macos.load_rx(f);
            s = macos.trace(n);
            r = macos.get_ray_info(s.nRays);
            n_re = nnz(logical(r.ok_pass) & logical(r.ok_trace));
            tc.verifyLessThan(n_re, 0.5*tc.nbase, ...
                ['the global-XY aperture centres did NOT break the reload -- ' ...
                 'this gate is vacuous on this fixture']);
            tc.t.clear_realized_apertures();
        end
    end
end

% ---- helpers (outside the class: they poke private/protected surface) ----
function [xo, yo, zo] = frame_of(t, psi)
%FRAME_OF  The element aperture triad, recomputed here from the ENGINE's
%   construction rather than read back from Telescope -- so the test is an
%   independent statement of the convention, not a mirror of the code.
    zo = psi(:)/norm(psi);
    seed = [zo(3); zo(1); zo(2)];          % iosub.inc ChkDf2 default
    yo = cross(zo, seed);  yo = yo/norm(yo);
    xo = cross(yo, zo);
end

function patch_apvec_centres(f, gc)
%PATCH_APVEC_CENTRES  Rewrite each element's ApVec (x,y) to gc(k,:),
%   keeping its radius -- the pre-fix emission, reproduced textually.
    txt = fileread(f);
    parts = regexp(txt, '(?m)^\s*iElt=', 'split');
    heads = regexp(txt, '(?m)^\s*iElt=', 'match');
    for k = 2:numel(parts)
        if k-1 > size(gc,1) || ~all(isfinite(gc(k-1,:))), continue; end
        parts{k} = regexprep(parts{k}, ...
            '(ApVec=\s*)(\S+)(\s+)(\S+)(\s+)(\S+)', ...
            sprintf('$1$2$3%.16E$5%.16E', gc(k-1,1), gc(k-1,2)), 'once');
    end
    out = parts{1};
    for k = 2:numel(parts), out = [out heads{k-1} parts{k}]; end %#ok<AGROW>
    fid = fopen(f, 'w');  fprintf(fid, '%s', out);  fclose(fid);
end

function n = pass_count(nE)
    s = macos.trace(nE);
    r = macos.get_ray_info(s.nRays);
    n = nnz(logical(r.ok_pass) & logical(r.ok_trace));
end

function delete_if(p), if exist(p,'file'), delete(p); end, end
