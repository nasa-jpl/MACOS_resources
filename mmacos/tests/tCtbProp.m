classdef tCtbProp < matlab.unittest.TestCase
%TCTBPROP  Pins for the CTB coronagraph diffraction chain (bench_ctb).
%   The chain rests on three conventions that fail SILENTLY when broken:
%   a quartet whose two exit-pupil spheres carry the same signed zElt, a
%   focus that lands on the FFT DC pixel floor(N/2)+1, and a generator that
%   emits both from the bare bench deck.  Each gets a pin here.
%
%   Checks
%     (a) quartet audit on both committed decks -- zElt equality to every
%         digit, spheres centred one radius off the focus flat, even Return
%         count (an odd one leaves the beam reversed at the exit)
%     (b) NF1/NF2 round-trip identity with no mask: the complex field at the
%         post-mask sphere equals the field at the pre-mask sphere
%     (c) centred-PSF pin at floor(N/2)+1, both decks
%     (d) bare compact-vs-full correlation pin
%     (e) PROPER arbiter on the FPM through-focus leg -- the only external
%         check in the file; skipped with a message when PROPER is absent
%     (f) the generator reproduces the committed decks
%
%   Asset-gated like tRunMet: the whole class is skipped when the bench_ctb
%   decks are not present.  Runs at model_size 512 (the decks declare
%   nGridpts=255, and one MATLAB process per model size is the house rule),
%   so it gets its own batch in run_mmacos_tests.sh.
%
%   See also: ctb_prop_layout, ctb_coro_compare, ctb_proper_compare,
%   examples/design/bench_ctb/README.md.

    properties (Constant)
        ModelSize = 512
        % Committed measurements the pins are taken from -- ctb_dcr.in /
        % ctb_s2s_dcr.in at model 512, nGridpts 255, lambda 500 nm.
        CorrCompactFull = 0.998863     % bare, peak-normalised
        PeakCompact     = 7.00582e-2
        PeakFull        = 6.02965e-2
        DxFpaM          = 2.4039e-5
    end

    properties
        bench           % examples/design/bench_ctb
        compact, full   % committed hand decks
    end

    methods (TestClassSetup)
        function gate(tc)
            here     = fileparts(mfilename('fullpath'));         % mmacos/tests
            tc.bench = fullfile(fileparts(here), 'examples', 'design', 'bench_ctb');
            tc.compact = fullfile(tc.bench, 'ctb_dcr.in');
            tc.full    = fullfile(tc.bench, 'ctb_s2s_dcr.in');
            tc.assumeTrue(isfile(tc.compact) && isfile(tc.full), ...
                'bench_ctb decks not present');
            addpath(tc.bench);
            macos.init(tc.ModelSize);
        end
    end

    methods (Test)

        % --- (a) quartet audit ------------------------------------------
        function test_quartet_audit_compact(tc)
            tc.check_quartets_(tc.compact);
        end

        function test_quartet_audit_full(tc)
            tc.check_quartets_(tc.full);
        end

        % --- (b) NF1/NF2 round-trip identity ----------------------------
        function test_nf_roundtrip_identity(tc)
            % With no mask applied, NF1 (sphere -> focus) followed by NF2
            % (focus -> sphere) must return the field unchanged.  It does so
            % only because the chirp argument is zero, which holds only when
            % the two spheres' zElt agree INCLUDING SIGN -- EPreturn2 at -R
            % gives S ~ 2R and a spurious defocus that no ray check catches.
            q = tCtbProp.quartets_(tc.compact);
            tc.verifyEqual(numel(q), 3, 'expected three focal quartets');
            macos.init(tc.ModelSize);
            macos.load_rx(tc.compact);
            first = true;
            for k = 1:numel(q)
                a = tCtbProp.cfield_(q(k).iEP,  first);   % pre-mask sphere
                b = tCtbProp.cfield_(q(k).iEP2, false);   % post-mask sphere
                first = false;
                amp = max(abs(a(:)));
                tc.verifyGreaterThan(amp, 0, ...
                    sprintf('%s: no field on the pre-mask sphere', q(k).name));
                rel = max(abs(b(:) - a(:))) / amp;
                tc.verifyLessThan(rel, 1e-10, sprintf( ...
                    ['%s: NF1/NF2 round trip is not the identity ' ...
                     '(max|d| / max|a| = %.3e) -- check that the two ' ...
                     'sphere zElts match, sign included'], q(k).name, rel));
            end
        end

        % --- (c) centred PSF --------------------------------------------
        function test_psf_centred_compact(tc)
            tc.check_psf_(tc.compact, tc.PeakCompact);
        end

        function test_psf_centred_full(tc)
            tc.check_psf_(tc.full, tc.PeakFull);
        end

        % --- (d) bare compact-vs-full correlation -----------------------
        function test_bare_models_agree(tc)
            Ic = tCtbProp.psf_(tc.compact, tc.ModelSize);
            If = tCtbProp.psf_(tc.full,    tc.ModelSize);
            n  = @(A) A / max(A(:));
            c  = corr2(n(Ic), n(If));
            tc.verifyEqual(c, tc.CorrCompactFull, 'AbsTol', 1e-4, ...
                sprintf(['bare compact-vs-full correlation moved: %.6f ' ...
                         '(pinned %.6f)'], c, tc.CorrCompactFull));
        end

        % --- (e) PROPER arbiter -----------------------------------------
        function test_proper_arbiter_fpm_leg(tc)
            % Model-vs-model agreement is not validation.  This is the one
            % external check: the FPM through-focus leg against MATLAB
            % PROPER at matched sampling (beam_ratio 1, so the two pitches
            % are equal by construction and the ratio is a real test of the
            % geometry, not of a resampling).
            tc.assumeTrue(exist('prop_begin','file')==2 && ...
                          exist('prop_lens','file')==2, ...
                'MATLAB PROPER not on the path (~/dev/proper_matlab)');
            out = ctb_proper_compare('model_size', tc.ModelSize, ...
                                     'outdir', tempdir, 'visible', false);
            tc.verifyTrue(out.have_proper);
            tc.verifyEqual(out.dx_f_proper_m / out.dx_f_macos_m, 1, ...
                'RelTol', 1e-4, 'focal pitch macos vs PROPER');
            tc.verifyGreaterThan(out.metrics.corr, 0.9999, ...
                'peak-normalised correlation macos vs PROPER');
            tc.verifyLessThan(hypot(out.metrics.dcx, out.metrics.dcy), 0.05, ...
                'centroid offset macos vs PROPER (px)');
        end

        % --- (f) generator reproduces the committed decks ---------------
        function test_generator_reproduces_hand_decks(tc)
            % Fast subset of the ctb_prop_layout acceptance: element count,
            % quartet audit, sphere radii, and -- for the full model, whose
            % committed deck already carries the correct DM1->DM2 leg length
            % -- the bare PSF peak.  (The committed COMPACT deck propagates
            % that leg over 399.94 mm against a 499.92 mm chief distance, so
            % its generated counterpart differs by ~0.23% in peak by design;
            % see the generator header.)
            d = fullfile(tempname); mkdir(d);
            restore = onCleanup(@() rmdir(d, 's'));
            info = ctb_prop_layout('outdir', d, 'model', tc.ModelSize, ...
                                   'verify', false);
            tc.verifyEqual(numel(info), 2);
            want = struct('compact', 31, 'full', 44);
            for k = 1:numel(info)
                tc.verifyEqual(info(k).nElt, want.(info(k).name), ...
                    sprintf('%s: element count', info(k).name));
                tc.check_quartets_(info(k).out);
            end
            % sphere radii against the committed compact deck
            ref = struct('Focus23_EPreturn',    7.0178526119080789E+03, ...
                         'FPM_EPreturn1',       1.0000841052379988E+03, ...
                         'FieldStop_EPreturn1', 4.1592775988551006E+02, ...
                         'ExitPupil',           3.5994606924986113E+02);
            got = info(strcmp({info.name},'compact')).R;
            map = struct('Focus23_EPreturn','Focus23', ...
                         'FPM_EPreturn1','FPM', ...
                         'FieldStop_EPreturn1','FieldStop', ...
                         'ExitPupil','ExitPupil');
            for f = fieldnames(ref).'
                tc.verifyEqual(got.(map.(f{1})), ref.(f{1}), 'RelTol', 1e-9, ...
                    sprintf('%s: FEX radius', f{1}));
            end
            % the full model must reproduce the committed PSF peak
            gfull = info(strcmp({info.name},'full')).out;
            Ig = tCtbProp.psf_(gfull,    tc.ModelSize);
            Ih = tCtbProp.psf_(tc.full,  tc.ModelSize);
            tc.verifyEqual(max(Ig(:)), max(Ih(:)), 'RelTol', 1e-6, ...
                'generated full deck: bare PSF peak');
            n = @(A) A / max(A(:));
            tc.verifyGreaterThan(corr2(n(Ig), n(Ih)), 0.9999, ...
                'generated full deck: bare PSF correlation');
        end
    end

    % ==================================================================
    methods (Access = private)

        function check_quartets_(tc, deck)
            q = tCtbProp.quartets_(deck);
            [~, nm, ex] = fileparts(deck);
            tag = [nm ex];
            tc.verifyEqual(numel(q), 3, ...
                sprintf('%s: expected three NF1/NF2 quartets', tag));
            for k = 1:numel(q)
                tc.verifyEqual(q(k).zEPstr, q(k).zEP2str, ...
                    sprintf(['%s / %s: the two exit-pupil spheres must ' ...
                             'carry the SAME zElt text (got "%s" and ' ...
                             '"%s") -- a sign flip here is a 2R defocus'], ...
                            tag, q(k).name, q(k).zEPstr, q(k).zEP2str));
                tc.verifyGreaterThan(q(k).R, 0, ...
                    sprintf('%s / %s: sphere radius', tag, q(k).name));
                tc.verifyLessThan(abs(q(k).centre_err), 1e-9, sprintf( ...
                    ['%s / %s: sphere vertex is %.3e mm off one radius ' ...
                     'from the focus flat'], tag, q(k).name, q(k).centre_err));
            end
            E = tCtbProp.elements_(deck);
            nRet = sum(strcmp({E.elem}, 'Return'));
            tc.verifyEqual(mod(nRet, 2), 0, sprintf( ...
                ['%s: Return count is %d (odd) -- each pair reverses then ' ...
                 'restores the chief, so an odd count leaves the beam ' ...
                 'running backwards at the exit'], tag, nRet));
        end

        function check_psf_(tc, deck, peak)
            I = tCtbProp.psf_(deck, tc.ModelSize);
            [pk, idx] = max(I(:));
            [r, c] = ind2sub(size(I), idx);
            ctr = floor(tc.ModelSize/2) + 1;
            [~, nm, ex] = fileparts(deck);
            % floor(N/2)+1, NOT (N-1)/2: MACOS's FarField / NF2 focus lands
            % on the FFT DC pixel.  The half-pixel error this pins cost a
            % round of asymmetric occulter leak before it was found.
            tc.verifyEqual([r c], [ctr ctr], sprintf( ...
                '%s%s: PSF peak at [%d,%d], expected the DC pixel [%d,%d]', ...
                nm, ex, r, c, ctr, ctr));
            tc.verifyEqual(pk, peak, 'RelTol', 1e-3, ...
                sprintf('%s%s: bare PSF peak', nm, ex));
        end
    end

    % ==================================================================
    methods (Static, Access = private)

        function I = psf_(deck, N)
            macos.init(N);
            nE = macos.load_rx(deck);
            I  = macos.intensity(nE);
        end

        function cf = cfield_(iElt, reset)
            % The veneer passes a third 'plane' argument that predates some
            % installed mexes; fall back to the two-argument raw dispatch,
            % as ctb_proper_compare does.
            try
                cf = macos.complex_field(iElt, 'reset_trace', reset);
            catch
                cf = mmacos('complex_field', double(iElt), double(reset));
            end
        end

        function E = elements_(deck)
            %ELEMENTS_  Parse a prescription into a per-element struct array.
            %   Every pattern is line-anchored: 'iElt' is a substring of
            %   'psiElt', so an unanchored match eats psiElt's leading digit.
            ln = regexp(fileread(deck), '\r?\n', 'split');
            ie = find(~cellfun('isempty', regexp(ln, '^\s*iElt=', 'once')));
            fi = find(~cellfun('isempty', regexp(ln, '^\s*nOutCord=','once')), 1);
            if isempty(fi), fi = numel(ln) + 1; end
            E = struct('name',{},'elem',{},'prop',{},'zstr',{}, ...
                       'kr',{},'vpt',{});
            for k = 1:numel(ie)
                if k < numel(ie), b = ie(k+1)-1; else, b = fi-1; end
                blk = strjoin(ln(ie(k):b), newline);
                g = @(key) strtrim(char(regexp(blk, ...
                        ['^\s*' key '=\s*(.*)$'], 'tokens', 'once', ...
                        'lineanchors', 'dotexceptnewline')));
                d = @(s) str2double(strrep(strrep(s,'D','E'),'d','e'));
                v = @(s) sscanf(strrep(strrep(s,'D','E'),'d','e'), '%f', 3);
                E(k) = struct('name', g('EltName'), 'elem', g('Element'), ...
                              'prop', g('PropType'), 'zstr', g('zElt'), ...
                              'kr',   d(g('KrElt')), 'vpt', v(g('VptElt')));
            end
        end

        function q = quartets_(deck)
            %QUARTETS_  Locate every NF1 / NF2 / sphere triple in light order.
            E = tCtbProp.elements_(deck);
            q = struct('name',{},'iEP',{},'iMask',{},'iEP2',{}, ...
                       'zEPstr',{},'zEP2str',{},'R',{},'centre_err',{});
            for k = 1:numel(E)-2
                if ~strcmp(E(k).prop, 'NF1'), continue; end
                assert(strcmp(E(k+1).prop, 'NF2'), ...
                    'NF1 at %s is not followed by an NF2 plane', E(k).name);
                q(end+1) = struct( ...                              %#ok<AGROW>
                    'name', E(k).name, 'iEP', k, 'iMask', k+1, 'iEP2', k+2, ...
                    'zEPstr', E(k).zstr, 'zEP2str', E(k+2).zstr, ...
                    'R', abs(E(k).kr), ...
                    'centre_err', norm(E(k+1).vpt - E(k).vpt) - abs(E(k).kr));
            end
        end
    end
end
