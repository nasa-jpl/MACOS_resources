classdef tCtbDm < matlab.unittest.TestCase
%TCTBDM  Pins for the CTB DM layer (grid-surface DMs + EFC machinery).
%   The DM layer rests on conventions that fail SILENTLY when broken:
%   the emitted grid channel must sit in the DM element's OWN frame
%   (the e5 "central dot" lesson -- a wrong frame paints pokes about
%   the aperture center), the sag-to-OPD scale is 2 cos(AOI) and is
%   only readable at the EXIT PUPIL (a bare macos.trace() reads the
%   FPA station, where a pupil bump smears into a global low-order
%   term ~10x larger -- the gate1 trap), and the EFC Jacobian must be
%   measured through the SAME masked chain the loop closes on.
%
%   Checks
%     (a) ctb_dm_rx emits a deck that loads; both DMs carry ng grids;
%         the emitted grid frame equals the element frame of the source
%         deck (pData=VptElt, zData=psiElt, xData=xObs)
%     (b) grid readback: engine holds the commanded surface exactly
%     (c) sag->OPD amplitude/sign/location at the exit pupil:
%         |dW|peak = 2 cos(5 deg) * sag within tolerance, negative
%         (longer-positive OPL: a bump toward the beam shortens),
%         peak at the commanded lattice site, mirror poke lands on the
%         mirrored pixel exactly
%     (d) FPA speckle pair: 180-deg rotation symmetry of |dE|
%     (e) masked-chain contrast pin (the shipped coronagraph config
%         reproduced through ctb_chain)
%     (f) Jacobian column linearity: dE/h at h and 2h agree
%     (g) EFC smoke: a small engine-in-the-loop run digs the dark zone
%
%   Asset-gated like tCtbProp; model_size 512 (own batch in
%   run_mmacos_tests.sh, SUITE_CTB_512).
%
%   See also: ctb_dm_rx, ctb_dm, ctb_chain, ctb_dm_jacobian, ctb_efc.

    properties (Constant)
        ModelSize = 512
        % Pins measured 2026-08-25 (dev-candidate, ctb_dm.in from
        % ctb_dcr.in, model 512, nGridpts 255, lambda 500 nm):
        ContrastCoro = 2.934e-7     % ctb_chain dark zone 3-15 lam/D mean
        RayPxMm      = 0.084        % ray-grid pitch at the DM (21.3/254)
    end

    properties
        bench, dm_rx    % bench dir + emitted deck info
    end

    methods (TestClassSetup)
        function gate(tc)
            here     = fileparts(mfilename('fullpath'));        % mmacos/tests
            tc.bench = fullfile(fileparts(here), 'templates', ...
                                '30_instruments', 'bench_ctb');
            tc.assumeTrue(isfile(fullfile(tc.bench, 'ctb_dcr.in')), ...
                'bench_ctb decks not present');
            addpath(tc.bench);
            prev = pwd;
            tc.addTeardown(@() cd(prev));
            cd(tc.bench);                  % GridFile= resolves against cwd
            tc.dm_rx = ctb_dm_rx();
            macos.init(tc.ModelSize);
        end
    end

    methods (Test)

        % --- (a) emitter: deck loads, grids present, frames faithful ----
        function test_emitted_deck_loads_with_grids(tc)
            r = tc.dm_rx;
            nE = macos.load_rx(r.rx_out);
            tc.verifyEqual(nE, 31, 'nElt changed by augmentation');
            for k = 1:2
                tc.verifyEqual( ...
                    double(mmacos('elt_srf_grid_size', r.ielt(k), 1)), ...
                    double(r.ng), sprintf('DM%d grid size', k));
            end
        end

        function test_grid_frame_equals_element_frame(tc)
            src = fileread(fullfile(tc.bench, 'ctb_dcr.in'));
            gen = fileread(tc.dm_rx.rx_out);
            % DM1 block of each (iElt= 2 .. iElt= 3)
            sB = extractBetween(src, 'EltName=  DM1', 'iElt=  3');
            gB = extractBetween(gen, 'EltName=  DM1', 'iElt=  3');
            tc.assertNotEmpty(sB); tc.assertNotEmpty(gB);
            getv = @(blk, key) sscanf(strrep(upper(char( ...
                regexp(char(blk), [key '=([^\n]*)'], 'tokens', 'once'))), ...
                'D', 'E'), '%g').';
            tc.verifyEqual(getv(gB{1},'pData'), getv(sB{1},'VptElt'), ...
                'AbsTol', 1e-12, 'pData must equal VptElt');
            tc.verifyEqual(getv(gB{1},'zData'), getv(sB{1},'psiElt'), ...
                'AbsTol', 1e-12, 'zData must equal psiElt');
            tc.verifyEqual(getv(gB{1},'xData'), getv(sB{1},'xObs'), ...
                'AbsTol', 1e-12, 'xData must equal xObs');
            tc.verifySubstring(char(gB{1}), 'GridData', ...
                'Surface must be GridData');
        end

        % --- (b) readback ------------------------------------------------
        function test_grid_readback_exact(tc)
            r = tc.dm_rx;
            macos.load_rx(r.rx_out);
            [G, ~] = tc.bump_(r, 5e-6, 1.0, 5.0, 0.0);
            macos.elt_grid_add(r.ielt(1), G);
            gb = macos.get_elt_grid(r.ielt(1));
            tc.verifyEqual(gb.mat, G, 'AbsTol', 0, ...
                'engine grid must equal the commanded surface bit-exactly');
            tc.verifyEqual(gb.dx, r.gdx_mm(1), 'RelTol', 1e-12);
        end

        % --- (c) sag -> OPD at the exit pupil ---------------------------
        function test_sag_to_opd_amplitude_sign_location(tc)
            r = tc.dm_rx;
            macos.load_rx(r.rx_out);
            amp = 5e-6;                                  % 5 nm sag
            macos.trace(30);  W0 = macos.opd();          % EXIT PUPIL, not FPA
            [G, ~] = tc.bump_(r, amp, 1.0, 5.0, 0.0);
            macos.set_elt_grid(r.ielt(1), r.gdx_mm(1), G);
            macos.trace(30);  W1 = macos.opd();
            dW = W1 - W0;  dW(~isfinite(dW)) = 0;
            [pk, im] = max(abs(dW(:)));
            [ic, jc] = ind2sub(size(dW), im);
            expct = 2 * amp * cosd(5);
            tc.verifyGreaterThan(pk/expct, 0.90, 'sag->OPD amplitude low');
            tc.verifyLessThan(pk/expct, 1.05, 'sag->OPD amplitude high');
            tc.verifyLessThan(dW(ic, jc), 0, ...
                'positive sag must read NEGATIVE OPL (shorter path)');
            n = size(dW, 1);  c = (n+1)/2;
            tc.verifyEqual((ic - c) * tc.RayPxMm, 5.0, 'AbsTol', 0.4, ...
                'bump must land at the commanded +x position');
            tc.verifyEqual(jc, round(c), 'AbsTol', 2, ...
                'bump must stay on the y=0 row');
            % mirror poke: exact mirror through the pupil center
            [Gm, ~] = tc.bump_(r, amp, 1.0, -5.0, 0.0);
            macos.set_elt_grid(r.ielt(1), r.gdx_mm(1), Gm);
            macos.trace(30);  W2 = macos.opd();
            dW2 = W2 - W0;  dW2(~isfinite(dW2)) = 0;
            [~, im2] = max(abs(dW2(:)));
            [ic2, jc2] = ind2sub(size(dW2), im2);
            tc.verifyEqual([ic2 jc2], [n+1-ic, jc], 'AbsTol', 1, ...
                'mirror poke must land on the mirrored pixel');
            macos.set_elt_grid(r.ielt(1), r.gdx_mm(1), zeros(r.ng));
        end

        % --- (d) speckle pair at the FPA --------------------------------
        function test_speckle_pair_symmetry(tc)
            r = tc.dm_rx;
            macos.load_rx(r.rx_out);
            E0 = macos.complex_field(31);
            [G, ~] = tc.bump_(r, 5e-6, 1.0, 7.0, 3.0);
            macos.set_elt_grid(r.ielt(1), r.gdx_mm(1), G);
            dE = macos.complex_field(31) - E0;
            A = abs(dE);  cc = corrcoef(A(:), reshape(rot90(A,2), [], 1));
            tc.verifyGreaterThan(cc(1,2), 0.95, ...
                'phase-bump speckles must come in symmetric pairs');
            macos.set_elt_grid(r.ielt(1), r.gdx_mm(1), zeros(r.ng));
        end

        % --- (e) masked-chain contrast pin ------------------------------
        function test_chain_reproduces_shipped_contrast(tc)
            r = tc.dm_rx;
            ch = ctb_chain('rx', r.rx_out, 'model_size', tc.ModelSize);
            E = ch.run();
            M = ch.dz_mask(3, 15);
            c = mean(abs(E(M)).^2) / ch.peak_bare;
            tc.verifyEqual(c, tc.ContrastCoro, 'RelTol', 0.05, ...
                'masked chain no longer reproduces the shipped contrast');
        end

        % --- (f) Jacobian column linearity ------------------------------
        function test_jacobian_column_linearity(tc)
            r = tc.dm_rx;
            ch = ctb_chain('rx', r.rx_out, 'model_size', tc.ModelSize);
            dm1 = ctb_dm('ielt', r.ielt(1), 'ng', r.ng, ...
                         'gdx_mm', r.gdx_mm(1));
            M = ch.dz_mask(3, 15);
            E0 = ch.run();
            j = find(dm1.active, 1, 'first') + 137;      % an interior actuator
            a = zeros(dm1.nact^2, 1);
            h = 2e-6;
            a(j) = h;    dm1.apply(a);  E1 = ch.run();
            a(j) = 2*h;  dm1.apply(a);  E2 = ch.run();
            dm1.clear();
            g1 = (E1(M) - E0(M)) / h;
            g2 = (E2(M) - E0(M)) / (2*h);
            tc.verifyLessThan(norm(g2 - g1)/norm(g1), 0.05, ...
                'dE/h must be step-independent at nm strokes');
            tc.verifyGreaterThan(norm(g1), 0, 'poke must reach the dark zone');
        end

        % --- (g) EFC smoke ----------------------------------------------
        function test_efc_digs_the_dark_zone(tc)
            J = ctb_dm_jacobian('model_size', tc.ModelSize, 'nact', 12, ...
                'inner_lamD', 3, 'outer_lamD', 8, 'save', false, ...
                'verbose', false);
            out = ctb_efc('jac', J, 'niter', 3, 'save', false);
            tc.verifyLessThan(out.c_after, 0.5 * out.c_before, ...
                'EFC must dig the dark zone at least 2x in 3 iterations');
        end
    end

    methods
        function [G, ax] = bump_(tc, r, amp, sig, x0, y0)
            ng = r.ng;  gdx = r.gdx_mm(1);
            ax = ((1:ng) - (ng+1)/2) * gdx;
            [X, Y] = ndgrid(ax, ax);
            G = amp * exp(-((X-x0).^2 + (Y-y0).^2) / (2*sig^2));
        end
    end
end
