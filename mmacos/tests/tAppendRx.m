classdef tAppendRx < matlab.unittest.TestCase
%TAPPENDRX  macos.design.append_rx -- splicing a back end onto a telescope.
%
%   Only ONE train can carry a telescope perturbation through to a
%   coronagraph contrast number, so a bench built separately has to become
%   MORE ELEMENTS of the telescope's deck.  These tests build a small
%   telescope and a small bench, splice them, and check the spliced deck
%   against the ENGINE -- element count, and the ray count of the base
%   train reproduced through the splice.
%
%   The unit check is the one that would otherwise pass silently: element
%   coordinates are raw numbers, so splicing a millimetre bench onto a
%   metre telescope shrinks it by 1000 and still "traces".

    properties (Constant)
        ModelSize = 128
    end
    properties
        tel_in
        tmpdir
    end

    methods (TestClassSetup)
        function build_base(tc)
            macos.init(tc.ModelSize);
            tc.tmpdir = tempname;  mkdir(tc.tmpdir);
            t = macos.design.Telescope('family','TMA', ...
                'aperture_diameter_m',1.0, 'model_size',tc.ModelSize, ...
                'wavelength_m',500e-9, 'grid_npts',21);
            t.set_base_sphere(true);
            t.add_mirror('M1','radius_m',6.4,'spacing_after_m',2.9,'tilt_deg',-5.6);
            t.add_mirror('M2','radius_m',0.633,'spacing_after_m',2.33, ...
                         'tilt_deg',5.9,'convex',true);
            t.add_mirror('M3','radius_m',0.583,'spacing_after','derive', ...
                         'tilt_deg',8.0);
            t.add_focal_plane('FP');
            t.build();
            tc.tel_in = fullfile(tc.tmpdir,'tel.in');
            t.save(tc.tel_in);
        end
    end

    methods (TestClassTeardown)
        function clean(tc)
            if ~isempty(tc.tmpdir) && isfolder(tc.tmpdir)
                rmdir(tc.tmpdir, 's');
            end
        end
    end

    methods (Test)

        function test_mismatched_units_are_refused(tc)
            % A mm bench onto an m telescope must ERROR, not splice.
            b = macos.design.Bench('mmbench', 'aperture', 0.05, ...
                                   'ngridpts', 11);          % default mm
            b.add_mirror(100, 'name','FoldA');
            b.add_detector(100, 'DetA');
            f = fullfile(tc.tmpdir,'mmbench.in');  b.emit(f);
            tc.verifyError(@() macos.design.append_rx(tc.tel_in, f, ...
                    fullfile(tc.tmpdir,'bad.in')), ...
                'macos:design:append_rx:units');
        end

        function test_metre_bench_emits_metre_units(tc)
            b = macos.design.Bench('mbench', 'baseunits','m', ...
                                   'wavelen', 500e-9, 'aperture', 0.05, ...
                                   'ngridpts', 11);
            f = fullfile(tc.tmpdir,'mbench_units.in');
            b.add_detector(0.1, 'Det');
            b.emit(f);
            txt = fileread(f);
            u = regexp(txt, '(?m)^\s*BaseUnits=\s*(\S+)', 'tokens', 'once');
            tc.verifyEqual(u{1}, 'm');
            w = regexp(txt, '(?m)^\s*WaveUnits=\s*(\S+)', 'tokens', 'once');
            tc.verifyEqual(w{1}, 'm');
        end

        function test_splice_counts_and_traces(tc)
            % Base ray count first, then the same through the spliced deck:
            % appending elements downstream of the focal plane must not
            % change what the telescope itself does.
            macos.init(tc.ModelSize);
            n0 = macos.load_rx(tc.tel_in);
            s0 = macos.trace(n0);
            r0 = macos.get_ray_info(s0.nRays);
            p0 = nnz(logical(r0.ok_pass) & logical(r0.ok_trace));
            tc.assumeGreaterThan(p0, 0.9*s0.nRays, 'base telescope does not trace');

            % a two-element metre bench, placed on the telescope's exit
            % chief so the appended elements are actually reachable
            macos.trace(n0);
            ri = macos.get_ray_info(s0.nRays);
            p = ri.pos(:,1);  d = ri.dir(:,1);
            b = macos.design.Bench('back', 'baseunits','m', ...
                    'pos', p - 0.5*d, 'dir', d, ...
                    'wavelen', 500e-9, 'aperture', 0.05, 'ngridpts', 11);
            b.add_reference(0.5, 'Relay');
            b.add_detector(0.4, 'Science');
            f = fullfile(tc.tmpdir,'back.in');  b.emit(f);

            out = fullfile(tc.tmpdir,'spliced.in');
            info = macos.design.append_rx(tc.tel_in, f, out, ...
                        'drop_base_tail', 1, 'rename','bk');
            tc.verifyEqual(info.n_out, info.n_base + info.n_add);
            tc.verifyEqual(info.baseunits, 'm');

            macos.init(tc.ModelSize);
            n1 = macos.load_rx(out);
            tc.verifyEqual(n1, info.n_out, ...
                'the engine did not load the spliced element count');
            s1 = macos.trace(n1);
            r1 = macos.get_ray_info(s1.nRays);
            p1 = nnz(logical(r1.ok_pass) & logical(r1.ok_trace));
            tc.verifyGreaterThan(p1, 0.9*p0, ...
                sprintf(['the spliced train lost the beam: %d/%d rays pass ' ...
                         'against %d in the base telescope'], p1, s1.nRays, p0));
            tc.verifyTrue(contains(fileread(out), 'bk_Science'), ...
                'the rename prefix did not reach the appended EltNames');
        end

        function test_missing_terminator_is_refused(tc)
            % The nOutCord block is the parser's element-list end marker;
            % a deck without it must be rejected loudly, not spliced into
            % something that loads as nElt = 0.
            f = fullfile(tc.tmpdir,'noterm.in');
            txt = fileread(tc.tel_in);
            i = regexp(txt, '(?m)^%\s*Output Coordinate System', 'once');
            fid = fopen(f,'w');  fprintf(fid,'%s',txt(1:i-1));  fclose(fid);
            tc.verifyError(@() macos.design.append_rx(f, tc.tel_in, ...
                    fullfile(tc.tmpdir,'bad2.in')), 'macos:design:append_rx:terminator');
        end
    end
end
