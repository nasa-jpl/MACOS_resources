classdef tPropLayout < matlab.unittest.TestCase
%TPROPLAYOUT  macos.design.prop_layout -- geometric deck -> diffraction deck.
%
%   The recipe's invariants, checked on a deck built here rather than on a
%   committed artifact:
%
%     * the emitted deck puts the CHIEF RAY where the geometric one did,
%       at every original station.  An inserted plane that is subtly
%       mis-posed still traces and still makes a PSF -- just not in the
%       right place;
%     * the far-field terminal lands the on-axis PSF on the FFT DC pixel;
%     * a focal-mask QUARTET with no mask applied is the IDENTITY.  That
%       is what makes it safe to insert one wherever a mask MIGHT go: the
%       sandwich must cost nothing when it is empty.  It is also the check
%       that pins the sign the whole recipe turns on -- both spheres carry
%       zElt = +R, so propsub's NF1 chirp argument is zero.  EPreturn2 at
%       -R gives an argument of order 2R, a defocus no ray check catches,
%       which is exactly why this is gated on the FIELD and not on rays.
%
%   NON-VACUITY: test_the_sign_is_load_bearing rewrites EPreturn2's zElt
%   to -R and requires the round trip to STOP being the identity.

    properties (Constant)
        ModelSize = 256
        NGrid     = 63
    end
    properties
        geo         % the bare geometric deck
        kinds
        tmpdir
    end

    methods (TestClassSetup)
        function build_fixture(tc)
            macos.init(tc.ModelSize);
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);

            % A small bench: source -> collimating OAP -> pupil marker ->
            % focusing OAP -> focus -> collimating OAP -> pupil -> focusing
            % OAP -> detector.  Metres, so nothing depends on a unit
            % convention elsewhere.
            aoi = 6;  conj_ = @(f) f/cosd(aoi)^2;
            f1 = 0.60;  f2 = 0.45;  f3 = 0.45;  f4 = 0.45;
            back = 0.15;
            b = macos.design.Bench('tprop', 'baseunits','m', ...
                    'pos',[0;0;0], 'dir',[0;0;1], 'wavelen', 500e-9, ...
                    'aperture', 2*atan(1/(2*20)), 'ngridpts', tc.NGrid, ...
                    'zsource', back);
            b.add_oap(back + conj_(f1), fold_(b.dir, aoi, +1), ...
                      'mode','collimate','f',f1,'name','OAP1');
            b.add_reference(0.10, 'Apodizer');
            b.add_oap(0.10, fold_(b.dir, aoi, -1), 'mode','focus','f',f2,'name','OAP2');
            b.add_reference(conj_(f2), 'FPM');
            b.add_oap(conj_(f3), fold_(b.dir, aoi, +1), ...
                      'mode','collimate','f',f3,'name','OAP3');
            b.add_reference(0.10, 'Lyot');
            b.add_oap(0.10, fold_(b.dir, aoi, -1), 'mode','focus','f',f4,'name','OAP4');
            b.add_detector(conj_(f4), 'Science');
            tc.geo = fullfile(tc.tmpdir,'tprop_geo.in');
            b.emit(tc.geo);

            nm = regexp(fileread(tc.geo), '^\s*EltName=\s*(\S+)', ...
                        'tokens','lineanchors');
            nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
            tc.kinds = repmat({'optic'},1,numel(nm));
            for k = 1:numel(nm)
                switch nm{k}
                    case {'Apodizer','Lyot'}, tc.kinds{k} = 'marker';
                    case 'FPM',               tc.kinds{k} = 'focus';
                    case 'Science',           tc.kinds{k} = 'image';
                end
            end
            macos.init(tc.ModelSize);
            n = macos.load_rx(tc.geo);
            s = macos.trace(n);
            r = macos.get_ray_info(s.nRays);
            tc.assumeGreaterThan(nnz(logical(r.ok_pass) & logical(r.ok_trace)), ...
                0.9*s.nRays, 'fixture bench does not trace');
        end
    end

    methods (TestClassTeardown)
        function clean(tc)
            if ~isempty(tc.tmpdir) && isfolder(tc.tmpdir), rmdir(tc.tmpdir,'s'); end
        end
    end

    methods (Test)

        function test_chief_ray_and_psf_centring(tc)
            out = fullfile(tc.tmpdir,'tprop_prop.in');
            info = macos.design.prop_layout(tc.geo, tc.kinds, 'out', out, ...
                        'model', tc.ModelSize, 'stop_name','Apodizer', ...
                        'verify', true);
            tc.verifyLessThan(info.chk.chief_max, 1e-9, ...
                'the diffraction deck moved the chief ray');
            tc.verifyTrue(info.chk.psf_centred, ...
                sprintf('PSF peak at [%d %d], centre %d', ...
                        info.chk.psf_row, info.chk.psf_col, info.chk.psf_centre));
            tc.verifyGreaterThan(info.R.FPM, 0);
            tc.verifyGreaterThan(info.R.ExitPupil, 0);
        end

        function test_empty_quartet_is_the_identity(tc)
            % With no mask applied, the pupil AFTER the focal quartet must
            % reproduce the pupil BEFORE it.  Compare the Apodizer pupil
            % (upstream of the quartet) with the Lyot pupil (downstream):
            % same beam, same plane class, nothing in between but the
            % sandwich.
            out = fullfile(tc.tmpdir,'tprop_prop.in');
            info = macos.design.prop_layout(tc.geo, tc.kinds, 'out', out, ...
                        'model', tc.ModelSize, 'stop_name','Apodizer', ...
                        'verify', false);
            macos.init(tc.ModelSize);
            macos.load_rx(out);
            macos.intensity(info.ix.Apodizer);
            Ia = macos.intensity(info.ix.Apodizer, 'reset_trace', false);
            Il = macos.intensity(info.ix.Lyot,     'reset_trace', false);
            pa = max(Ia(:));  pl = max(Il(:));
            tc.verifyGreaterThan(pa, 0);
            tc.verifyEqual(pl, pa, 'RelTol', 1e-6, ...
                ['an EMPTY focal quartet changed the pupil -- the sandwich ' ...
                 'is not transparent']);
        end

        function test_the_sign_is_load_bearing(tc)
            % NON-VACUITY.  Flip EPreturn2's zElt from +R to -R -- the one
            % sign the recipe warns about -- and the round trip must stop
            % being the identity.  If this ever passes, the test above has
            % stopped testing anything.
            out = fullfile(tc.tmpdir,'tprop_prop.in');
            info = macos.design.prop_layout(tc.geo, tc.kinds, 'out', out, ...
                        'model', tc.ModelSize, 'stop_name','Apodizer', ...
                        'verify', false);
            macos.init(tc.ModelSize);
            macos.load_rx(out);
            macos.intensity(info.ix.Apodizer);
            Ia = macos.intensity(info.ix.Apodizer, 'reset_trace', false);
            Il = macos.intensity(info.ix.Lyot,     'reset_trace', false);
            ref = max(Il(:)) / max(Ia(:));

            bad = fullfile(tc.tmpdir,'tprop_badsign.in');
            flip_zelt_(out, bad, 'FPM_EPreturn2');
            macos.init(tc.ModelSize);
            macos.load_rx(bad);
            macos.intensity(info.ix.Apodizer);
            Ia2 = macos.intensity(info.ix.Apodizer, 'reset_trace', false);
            Il2 = macos.intensity(info.ix.Lyot,     'reset_trace', false);
            badratio = max(Il2(:)) / max(Ia2(:));
            tc.verifyThat(abs(badratio - ref) > 1e-3*ref, ...
                matlab.unittest.constraints.IsTrue, ...
                ['flipping EPreturn2''s zElt sign did NOT disturb the round ' ...
                 'trip -- the identity test is vacuous on this fixture']);
        end
    end
end

% ---- helpers -------------------------------------------------------------
function o = fold_(d, aoi_deg, sgn)
    d = d(:)/norm(d);
    a = [1;0;0];  a = a - (a.'*d)*d;
    if norm(a) < 1e-9, a = [0;1;0] - ([0;1;0].'*d)*d; end
    a = sgn * a/norm(a);
    th = pi - 2*deg2rad(aoi_deg);
    o = cos(th)*d + sin(th)*a;  o = o/norm(o);
end

function flip_zelt_(src, dst, eltname)
%FLIP_ZELT_  Negate one named element's zElt, leaving everything else.
    txt = fileread(src);
    blocks = regexp(txt, '(?m)^\s*iElt=', 'split');
    heads  = regexp(txt, '(?m)^\s*iElt=', 'match');
    for k = 2:numel(blocks)
        nm = regexp(blocks{k}, '^\s*EltName=\s*(\S+)', 'tokens','once','lineanchors');
        if isempty(nm) || ~strcmp(nm{1}, eltname), continue; end
        z = regexp(blocks{k}, '^\s*zElt=\s*(\S+)', 'tokens','once','lineanchors');
        blocks{k} = regexprep(blocks{k}, '^(\s*zElt=)\s*\S+', ...
                        sprintf('$1%.16E', -str2double(z{1})), ...
                        'lineanchors','once');
    end
    out = blocks{1};
    for k = 2:numel(blocks), out = [out heads{k-1} blocks{k}]; end %#ok<AGROW>
    fid = fopen(dst,'w');  fprintf(fid,'%s',out);  fclose(fid);
end
