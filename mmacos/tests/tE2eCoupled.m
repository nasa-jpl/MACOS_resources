classdef tE2eCoupled < matlab.unittest.TestCase
%TE2ECOUPLED  e2e stage 9: the coupled telescope + CTB diffraction model.
%   The CTB's three propagation conventions (tCtbProp) on the telescope
%   train: the generator reproduces the committed coupled decks from the
%   committed stage-8 fold (sphere radii to 1e-9, element counts), every
%   quartet's two spheres carry the same zElt to the digit with an even
%   Return count, the NF1/NF2 round trip is the identity, the PSF sits on
%   the centre pixel at the recorded bare peak, and the masked chain
%   reproduces the stage's static dark zone on the telescope -- and what
%   sets it: the same deck without the secondary's shadow is below the
%   bench's own floor, and the central Lyot blocker the stage picked
%   reproduces its number.  One refusal: a wrong 'lead' is refused by
%   count.  Model 512 (the CTB's), the SUITE_CTB_512 group of
%   run_mmacos_tests.sh; the Jacobian and the EFC dig are the stage's
%   (s9_report.txt), not re-run here.
%
%   PLAN_CONSOLIDATION 3a, the coupled model (CCL, 2026-10-29).

    properties (Constant)
        ModelSize = 512
        % the stage's recorded numbers (s9_report.txt, 2026-10-29)
        PeakCompact  = 7.8540e-2
        ContrastStatic = 5.795e-6     % plain Lyot disk 0.50
        ContrastLyot   = 5.267e-7     % with the central blocker, 0.08 pupil radii
        LyotInner      = 0.08
        ContrastNoObs  = 2.363e-7     % the secondary's shadow removed
        ContrastBench  = 2.934e-7     % tCtbDm's pin, the bench's own
        RadiiM = struct('Focus23', 7.0229991301, 'FPM', 0.9997313865, ...
                        'FieldStop', 0.4165944046, 'ExitPupil', 0.3593849976)
    end
    properties
        e2e; bench; src; compact; work
    end

    methods (TestClassSetup)
        function setupClass(tc)
            root = fullfile(fileparts(mfilename('fullpath')), '..');
            run(fullfile(root, 'mmacos_setup.m'));
            tc.e2e   = fullfile(root, 'templates', '80_end_to_end', 'e2e');
            tc.bench = fullfile(root, 'templates', '30_instruments', 'bench_ctb');
            addpath(tc.bench);
            tc.src     = fullfile(tc.e2e, 's8_fold_full.in');
            tc.compact = fullfile(tc.e2e, 's9_coupled.in');
            tc.assumeTrue(isfile(tc.src) && isfile(tc.compact), 'the e2e stage-8/9 decks are not committed here');
            tc.work = tempname;  mkdir(tc.work);
        end
    end
    methods (TestClassTeardown)
        function teardownClass(tc)
            if isfolder(tc.work), rmdir(tc.work, 's'); end
        end
    end

    methods (Test)
        function test_generator_reproduces_the_committed_decks(tc)
            info = ctb_prop_layout('src', tc.src, 'lead', 9, 'stem', 's9_coupled', 'suffix', '', ...
                                   'outdir', tc.work, 'model', tc.ModelSize, 'verify', false);
            tc.verifyEqual(numel(info), 2);
            want = struct('compact', 40, 'full', 53);
            for k = 1:numel(info)
                tc.verifyEqual(info(k).nElt, want.(info(k).name), sprintf('%s: element count', info(k).name));
            end
            got = info(strcmp({info.name}, 'compact')).R;
            for f = fieldnames(tc.RadiiM).'
                tc.verifyEqual(got.(f{1}), tc.RadiiM.(f{1}), 'RelTol', 1e-9, sprintf('%s: FEX radius through the telescope', f{1}));
            end
            % the committed compact deck carries these radii (its text)
            E = tE2eCoupled.elements_(tc.compact);
            tc.verifyEqual(numel(E), 40, 'the committed compact deck has 40 elements');
            tc.verifyEqual(abs(E(strcmp({E.name}, 'FPM_EPreturn')).kr), tc.RadiiM.FPM, 'RelTol', 1e-9, 'the committed FPM sphere');
        end

        function test_wrong_lead_is_refused(tc)
            tc.verifyError(@() ctb_prop_layout('src', tc.src, 'lead', 8, 'stem', 'x', 'outdir', tc.work, ...
                                               'model', tc.ModelSize, 'verify', false), ?MException);
        end

        function test_quartets_and_round_trip(tc)
            q = tE2eCoupled.quartets_(tc.compact);
            tc.verifyEqual(numel(q), 3, 'three focal quartets behind the telescope');
            for k = 1:numel(q)
                tc.verifyEqual(q(k).zEPstr, q(k).zEP2str, sprintf('%s: the two spheres'' zElt to the digit', q(k).name));
            end
            E = tE2eCoupled.elements_(tc.compact);
            tc.verifyEqual(mod(nnz(strcmp({E.elem}, 'Return')), 2), 0, 'an even Return count');
            macos.init(tc.ModelSize);  macos.load_rx(tc.compact);
            first = true;
            for k = 1:numel(q)
                a = macos.complex_field(q(k).iEP,  'reset_trace', first);  first = false;
                b = macos.complex_field(q(k).iEP2, 'reset_trace', false);
                tc.verifyGreaterThan(max(abs(a(:))), 0, sprintf('%s: a field on the pre-mask sphere', q(k).name));
                tc.verifyLessThan(max(abs(b(:) - a(:))) / max(abs(a(:))), 1e-10, sprintf('%s: NF1/NF2 round trip', q(k).name));
            end
        end

        function test_psf_centred_and_static_dark_zone(tc)
            macos.init(tc.ModelSize);  n = macos.load_rx(tc.compact);
            I = macos.intensity(n);
            [pk, idx] = max(I(:));  [r, c] = ind2sub(size(I), idx);
            ctr = floor(tc.ModelSize/2) + 1;
            tc.verifyEqual([r c], [ctr ctr], 'the PSF peak on the centre pixel');
            tc.verifyEqual(pk, tc.PeakCompact, 'RelTol', 1e-4, 'the recorded bare peak');
            % the masked chain on the DM deck, the stage's static number
            rr = ctb_dm_rx('rx_in', tc.compact, 'rx_out', fullfile(tc.work, 's9_coupled_dm.in'));
            tc.verifyEqual(rr.cbm, 1, 'the coupled deck is in metres');
            E = tE2eCoupled.elements_(tc.compact);  ix = @(nm) find(strcmp({E.name}, nm), 1);
            e = struct('DM1', ix('DM1'), 'DM2', ix('DM2'), 'Apodizer', ix('Apodizer'), 'FPM', ix('FPM'), ...
                       'Lyot', ix('Lyot'), 'ExitPupil', ix('ExitPupil'), 'FPA', ix('FPA'));
            here = cd(tc.work);  restore = onCleanup(@() cd(here));   % GridFile resolves against the cwd
            ch = ctb_chain('rx', rr.rx_out, 'elt', e, 'model_size', tc.ModelSize);
            Ef = ch.run();  M = ch.dz_mask(3, 15);
            tc.verifyEqual(mean(abs(Ef(M)).^2) / ch.peak_bare, tc.ContrastStatic, 'RelTol', 1e-3, ...
                'the static dark zone on the telescope (s9_report)');
            tc.verifyGreaterThan(mean(abs(Ef(M)).^2) / ch.peak_bare, tc.ContrastBench * 5, 'well above the bench''s own floor');
        end

        function test_the_shadow_sets_the_halo_and_the_blocker_takes_it(tc)
            rr = ctb_dm_rx('rx_in', tc.compact, 'rx_out', fullfile(tc.work, 's9_coupled_dm.in'));
            E = tE2eCoupled.elements_(tc.compact);  ix = @(nm) find(strcmp({E.name}, nm), 1);
            e = struct('DM1', ix('DM1'), 'DM2', ix('DM2'), 'Apodizer', ix('Apodizer'), 'FPM', ix('FPM'), ...
                       'Lyot', ix('Lyot'), 'ExitPupil', ix('ExitPupil'), 'FPA', ix('FPA'));
            here = cd(tc.work);  restore = onCleanup(@() cd(here));
            % the counterfactual: the same deck with the secondary's obscuration removed
            txt = fileread(rr.rx_out);
            noobs = regexprep(txt, '(?m)^\s*nObs=\s*1\s*\n\s*ObsType=\s*Circle\s*\n\s*ObsVec=[^\n]*\n', '             nObs=  0\n', 'once');
            tc.assertFalse(strcmp(noobs, txt), 'the coupled deck carries the secondary''s obscuration');
            fn = fullfile(tc.work, 's9_noobs_dm.in');  fid = fopen(fn, 'w');  fwrite(fid, noobs);  fclose(fid);
            ch = ctb_chain('rx', fn, 'elt', e, 'model_size', tc.ModelSize);
            En = ch.run();  M = ch.dz_mask(3, 15);  cn = mean(abs(En(M)).^2) / ch.peak_bare;
            tc.verifyEqual(cn, tc.ContrastNoObs, 'RelTol', 1e-3, 'without the shadow (s9_report)');
            tc.verifyLessThan(cn, tc.ContrastBench, 'without the shadow the telescope is below the bench''s own floor: its wavefront is not the limit');
            % the blocker the stage picked
            ch = ctb_chain('rx', rr.rx_out, 'elt', e, 'model_size', tc.ModelSize, 'r_lyot_inner_frac', tc.LyotInner);
            El = ch.run();  cl = mean(abs(El(M)).^2) / ch.peak_bare;
            tc.verifyEqual(cl, tc.ContrastLyot, 'RelTol', 1e-3, 'the annular stop''s static dark zone (s9_report)');
            tc.verifyLessThan(cl, tc.ContrastStatic / 5, 'the blocker takes the shadow''s diffraction (x5)');
            tc.verifyEqual(ch.config{find(strcmp(ch.config, 'r_lyot_inner_frac')) + 1}, tc.LyotInner, 'the blocker is in the chain''s stamp');
        end
    end

    methods (Static)
        function E = elements_(deck)
            ln = regexp(fileread(deck), '\r?\n', 'split');
            ie = find(~cellfun('isempty', regexp(ln, '^\s*iElt=', 'once')));
            fi = find(~cellfun('isempty', regexp(ln, '^\s*nOutCord=', 'once')), 1);
            if isempty(fi), fi = numel(ln) + 1; end
            E = struct('name', {}, 'elem', {}, 'prop', {}, 'zstr', {}, 'kr', {});
            for k = 1:numel(ie)
                if k < numel(ie), b = ie(k+1)-1; else, b = fi-1; end
                blk = strjoin(ln(ie(k):b), newline);
                g = @(key) strtrim(char(regexp(blk, ['^\s*' key '=\s*(.*)$'], 'tokens', 'once', 'lineanchors', 'dotexceptnewline')));
                d = @(s) str2double(strrep(strrep(s, 'D', 'E'), 'd', 'e'));
                E(k) = struct('name', g('EltName'), 'elem', g('Element'), 'prop', g('PropType'), 'zstr', g('zElt'), 'kr', d(g('KrElt')));
            end
        end
        function q = quartets_(deck)
            E = tE2eCoupled.elements_(deck);
            q = struct('name', {}, 'iEP', {}, 'iEP2', {}, 'zEPstr', {}, 'zEP2str', {});
            for k = 1:numel(E)-2
                if ~strcmp(E(k).prop, 'NF1'), continue; end
                assert(strcmp(E(k+1).prop, 'NF2'), 'NF1 at %s is not followed by an NF2 plane', E(k).name);
                q(end+1) = struct('name', E(k).name, 'iEP', k, 'iEP2', k+2, 'zEPstr', E(k).zstr, 'zEP2str', E(k+2).zstr); %#ok<AGROW>
            end
        end
    end
end
