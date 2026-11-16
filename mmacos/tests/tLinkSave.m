classdef tLinkSave < matlab.unittest.TestCase
%TLINKSAVE  WS2: "Link= iElt" rigid-group linkage survives a SAVE round-trip.
%
%   The linked-element perturbation feature (an element carrying "Link= A" is
%   dragged rigidly when A is perturbed; funcsub.F CPERTURB -> LnkEltCPERTURB)
%   was fully live but the SAVE writer never emitted Link=, so a
%   load->perturb->save->reload silently lost the linkage.  PrtSingleEltInfo
%   now emits "Link= <iElt>".  These tests confirm (a) the linkage is active,
%   (b) it survives save->reload byte-for-behavior, (c) the keyword is written,
%   and (d) a NEGATIVE control (no link) proves the check has teeth.

    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Master    = 2    % Primary (Reflector)
        Linked    = 3    % Secondary (Reflector) -- carries "Link= 2"
        RotRad    = 1e-3
    end

    properties
        rx_path
    end

    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
        end
    end

    methods (Access = private)
        function p = rx_with_link(testCase, wd, fname)
            % Copy the fixture, injecting "Link= <Master>" into the Linked
            % element's block (after its "iElt=" line) -- mirrors
            % tDwDxGroups.rx_with_eltgrp.
            L = splitlines(string(fileread(testCase.rx_path)));
            out = strings(0, 1);
            for k = 1:numel(L)
                out(end+1, 1) = L(k); %#ok<AGROW>
                t = strtrim(L(k));
                if startsWith(t, "iElt=")
                    v = sscanf(char(extractAfter(t, "=")), '%d', 1);
                    if ~isempty(v) && v == testCase.Linked
                        out(end+1, 1) = sprintf("          Link=  %d", ...
                            testCase.Master); %#ok<AGROW>
                    end
                end
            end
            p = fullfile(wd, fname);
            fid = fopen(p, 'w');  fprintf(fid, '%s\n', out);  fclose(fid);
        end

        function d = master_opd_response(testCase, m)
            % Max |OPD| change when the Master element is rotated (restored
            % afterward).  With Link= 2 on elt 3, rotating elt 2 also moves
            % elt 3, so this differs from the stock (unlinked) response -- a
            % robust, frame-getter-independent probe of the linkage.
            wf = m.num_elt() - 1;
            m.trace(wf);  W0 = m.opd();
            m.perturb(testCase.Master, 'rotation', [testCase.RotRad; 0; 0]);
            m.modify();  m.trace(wf);  W1 = m.opd();
            m.perturb(testCase.Master, 'rotation', [-testCase.RotRad; 0; 0]);
            m.modify();
            v = (W0 ~= 0) & (W1 ~= 0);
            d = max(abs(W1(v) - W0(v)));
        end
    end

    methods (Test)
        function test_save_emits_link_only_when_present(testCase)
            % The writer emits "Link= <master>" for the linked element, and
            % NOTHING for the stock (unlinked) fixture (LnkElt inits to -1).
            wd = tempname;  mkdir(wd);
            c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_with_link(wd, 'cass_link.in');
            m = macos.Session(testCase.ModelSize);
            m.load_rx(rx);
            saved = fullfile(wd, 'cass_link_saved.in');
            m.save_rx(saved);
            txt = fileread(saved);
            testCase.verifyTrue(contains(txt, 'Link='), ...
                'SAVE must emit the Link= keyword for a linked element');
            testCase.verifyTrue(~isempty(regexp(txt, 'Link=\s*2\>', 'once')), ...
                'emitted Link= must carry the master element id (2)');

            ms = macos.Session(testCase.ModelSize);
            ms.load_rx(testCase.rx_path);           % stock, no Link=
            saved2 = fullfile(wd, 'cass_stock_saved.in');
            ms.save_rx(saved2);
            testCase.verifyFalse(contains(fileread(saved2), 'Link='), ...
                'unlinked elements must not emit Link=');
        end

        function test_link_active_and_distinct_from_stock(testCase)
            % The linkage is live: rotating the master perturbs the linked
            % element too, so the OPD response differs from the stock deck.
            wd = tempname;  mkdir(wd);
            c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_with_link(wd, 'cass_link.in');
            ml = macos.Session(testCase.ModelSize);  ml.load_rx(rx);
            r_link = testCase.master_opd_response(ml);
            ms = macos.Session(testCase.ModelSize);  ms.load_rx(testCase.rx_path);
            r_stock = testCase.master_opd_response(ms);
            testCase.verifyGreaterThan(r_link, 0);
            testCase.verifyGreaterThan(abs(r_link - r_stock) / r_stock, 1e-2, ...
                'Link= must change the master''s OPD response (drags elt 3)');
        end

        function test_link_survives_save_round_trip(testCase)
            wd = tempname;  mkdir(wd);
            c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_with_link(wd, 'cass_link.in');

            m0 = macos.Session(testCase.ModelSize);  m0.load_rx(rx);
            r_direct = testCase.master_opd_response(m0);

            % load -> SAVE -> reload, then measure the same response
            saved = fullfile(wd, 'cass_link_saved.in');
            m1 = macos.Session(testCase.ModelSize);  m1.load_rx(rx);
            m1.save_rx(saved);
            m1b = macos.Session(testCase.ModelSize);  m1b.load_rx(saved);
            r_reload = testCase.master_opd_response(m1b);

            testCase.verifyEqual(r_reload, r_direct, 'RelTol', 1e-9, ...
                'round-tripped linkage must reproduce the original response');

            % SAVE-again must still carry it (parse-of-emitted-form round-trips)
            saved2 = fullfile(wd, 'cass_link_saved2.in');
            m1b.save_rx(saved2);
            testCase.verifyTrue(contains(fileread(saved2), 'Link='), ...
                'Link= must survive a second SAVE (parse of the emitted form)');
        end
    end
end
