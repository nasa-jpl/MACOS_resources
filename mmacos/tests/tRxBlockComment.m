classdef tRxBlockComment < matlab.unittest.TestCase
%TRXBLOCKCOMMENT  /* ... */ and CommentBegin/CommentEnd blocks in an Rx.
%
%   The engine's parser (GET_EQ) has always skipped block comments, but two
%   things around it were broken until 2026-09-29 (macos 223a6ff):
%     * the Phase-1 VALIDATOR, which runs before the parser and is shared by
%       the CLI and both bindings, knew only whole-line '%'/'!' -- a '/*'
%       line has no '=' and was read as a continuation row of the last
%       multi-row key, so a deck the parser reads fine was REFUSED;
%     * SAVE lost every block: capture saw only '%' lines.
%   These tests run the binding path (macos.load_rx -> smacosio MBFile6 ->
%   the validator -> the parser; macos.save_rx -> the writers), so they
%   prove the fix reaches MATLAB and not just the CLI.  The variants are
%   generated from the stock deck at run time, so the NEGATIVE CONTROL -- the
%   same commented-out keyword written live -- differs by exactly the block
%   markers and proves the "must not apply" check has teeth.
    properties (Constant)
        ModelSize = 128
        RxName    = 'Rx_Cass_FarField.in'
        Target    = 2              % the Primary (a conic Reflector): the block tries to set its KrElt
        BogusKr   = -9.99e2        % the value inside the block; must NOT apply
    end
    properties
        rx_path
        kr_stock
    end
    methods (TestClassSetup)
        function setupClass(testCase)
            testCase.rx_path = rx_fixture_path(testCase.RxName);
            macos.init(testCase.ModelSize);
            m = macos.Session(testCase.ModelSize);
            m.load_rx(testCase.rx_path);
            testCase.kr_stock = m.get_elt_kr(testCase.Target);
        end
    end
    methods (Access = private)
        function p = rx_variant(testCase, wd, fname, mode)
            % 'blocks'       -- a /* */ block ahead of the Target element that
            %                   carries a commented-out KrElt, a '%' line and a
            %                   blank; a CommentBegin/CommentEnd block inside
            %                   element 3; an in-line '% note' on element 1.
            % 'live'         -- the SAME KrElt line, NOT in a block (control).
            % 'unterminated' -- a '/*' that is never closed.
            L = splitlines(string(fileread(testCase.rx_path)));
            out = strings(0, 1);  seen_name1 = false;
            krline = sprintf("            KrElt=  %.2E   commented-out: must not apply", testCase.BogusKr);
            for k = 1:numel(L)
                t = strtrim(L(k));
                iv = [];
                if startsWith(t, "iElt="), iv = sscanf(char(extractAfter(t, "=")), '%d', 1); end
                % Insert at the END of the Target element's block -- i.e. just
                % before the next element's iElt= -- so a live KrElt there is the
                % LAST one the Target sees and wins (a flat element or an earlier
                % KrElt would otherwise mask the control).
                if ~isempty(iv) && iv == testCase.Target + 1
                    switch mode
                        case 'blocks'
                            out(end+1:end+5, 1) = ["/*"; krline; ...
                                "% a percent line INSIDE the block belongs to the block"; ...
                                ""; "*/"];
                        case 'live'
                            out(end+1, 1) = krline;
                        case 'unterminated'
                            out(end+1:end+2, 1) = ["/*"; "   never closed"];
                    end
                end
                out(end+1, 1) = L(k); %#ok<AGROW>
                if strcmp(mode, 'blocks')
                    if ~isempty(iv) && iv == testCase.Target + 1
                        out(end+1:end+3, 1) = ["CommentBegin"; ...
                            "   KcElt=  -1.5E+00   old value, kept for reference"; "CommentEnd"];
                    end
                    if ~seen_name1 && startsWith(t, "EltName=")
                        out(end) = L(k) + "     % the first element, an in-line note";
                        seen_name1 = true;
                    end
                end
            end
            p = fullfile(wd, fname);
            fid = fopen(p, 'w');  fprintf(fid, '%s\n', out);  fclose(fid);
        end
    end
    methods (Test)
        function test_block_commented_deck_loads_and_blocks_do_not_apply(testCase)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_variant(wd, 'cass_blocks.in', 'blocks');
            m = macos.Session(testCase.ModelSize);
            n = m.load_rx(rx);
            ms = macos.Session(testCase.ModelSize);
            n0 = ms.load_rx(testCase.rx_path);
            testCase.verifyEqual(n, n0, 'element count must match the stock deck');
            testCase.verifyEqual(m.get_elt_kr(testCase.Target), ms.get_elt_kr(testCase.Target), ...
                'AbsTol', 0, 'a KrElt inside a /* */ block must NOT apply');
            testCase.verifyEqual(m.get_elt_kr(testCase.Target), testCase.kr_stock, ...
                'AbsTol', 0, 'the Target element must carry its stock KrElt');
        end
        function test_negative_control_the_same_keyword_live_does_apply(testCase)
            % Identical deck minus the block markers: the keyword is live and
            % must change the element it lands on.  Without this the previous
            % test could pass vacuously (e.g. if KrElt were simply ignored).
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_variant(wd, 'cass_live.in', 'live');
            m = macos.Session(testCase.ModelSize);
            m.load_rx(rx);
            testCase.verifyEqual(m.get_elt_kr(testCase.Target), testCase.BogusKr, ...
                'RelTol', 1e-12, 'the control: a live KrElt line must apply');
        end
        function test_save_round_trip_is_idempotent_and_keeps_the_blocks(testCase)
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_variant(wd, 'cass_blocks.in', 'blocks');
            m = macos.Session(testCase.ModelSize);
            m.load_rx(rx);
            A = fullfile(wd, 'A.in');  m.save_rx(A);
            m2 = macos.Session(testCase.ModelSize);
            m2.load_rx(A);
            B = fullfile(wd, 'B.in');  m2.save_rx(B);
            tA = fileread(A);  tB = fileread(B);
            testCase.verifyEqual(tA, tB, 'SAVE -> load -> SAVE must be byte-identical');
            for tok = ["/*", "*/", "CommentBegin", "CommentEnd", ...
                       "belongs to the block", "commented-out: must not apply"]
                testCase.verifyTrue(contains(tA, tok), ...
                    sprintf('the SAVEd deck must keep the block text "%s"', tok));
            end
            % and the re-loaded SAVEd deck still keeps the block INERT
            testCase.verifyEqual(m2.get_elt_kr(testCase.Target), ...
                testCase.kr_stock, 'AbsTol', 0, ...
                'the block must be inert on reload of the SAVEd deck too');
        end
        function test_unterminated_block_is_refused(testCase)
            % The parser would swallow the rest of the file silently; the
            % validator now refuses it before the parser sees it.
            wd = tempname;  mkdir(wd);  c = onCleanup(@() rmdir(wd, 's'));
            rx = testCase.rx_variant(wd, 'cass_open.in', 'unterminated');
            m = macos.Session(testCase.ModelSize);
            testCase.verifyError(@() m.load_rx(rx), ?MException, ...
                'a /* with no */ must be refused at load');
        end
    end
end
