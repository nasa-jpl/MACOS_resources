classdef tRxHeaderRewrite < matlab.unittest.TestCase
%TRXHEADERREWRITE  Every header-key rewrite in the tree is anchored to its line.
%   The engine's header keys nest: OptRayGrid= ends in yGrid=, OptChfRayDir=
%   and JZLouChfRayDir= end in ChfRayDir=, OptChfRayPos= and JZLouChfRayPos=
%   in ChfRayPos= (msmacosio.inc), and SAVE writes the Opt* rows whenever a
%   CALIB setup is non-default (iosub.inc).  A regexprep whose pattern can
%   match 'KEY=' mid-line rewrites the sibling as well.  On 2026-10-30 the
%   unanchored  (yGrid=\s*)[^\n]*  in pupil_map's field rewrite overwrote
%   OptRayGrid's count with a direction vector and the engine refused the
%   reload (CCMac: tPupilFindMethod 9/9 errored on a zoom deck carrying the
%   key); the ChfRayDir= form overwrites every OptChfRayDir= row of a
%   multi-field CALIB block silently (challenges/dyson5/dyson5_s4_r4n_seed.in
%   carries four).  The anchored form is '(?m)^(\s*KEY=\s*)[^\n]*'
%   (tma_longslit's precedent).  The same four lines had been copied into
%   thirty files; this gate scans the tree so the copy cannot spread again.
%   The behavioural gates (a planted sibling key leaves the metric bit for
%   bit) are in tPupilMap and tAfocalKernel.
%
%   PLAN_CONSOLIDATION 1.2, CCMac's 2026-10-30 roll-up (CCL).

    methods (Test)
        function test_every_header_key_rewrite_is_line_anchored(tc)
            root = fileparts(fileparts(mfilename('fullpath')));
            dirs = {'design', 'src', 'templates', 'tests', 'challenges'};
            keys = '(ChfRayDir|ChfRayPos|xGrid|yGrid)=';
            bad = {};
            for d = dirs
                fl = dir(fullfile(root, d{1}, '**', '*.m'));
                for k = 1:numel(fl)
                    f = fullfile(fl(k).folder, fl(k).name);
                    lines = splitlines(string(fileread(f)));
                    for i = 1:numel(lines)
                        ln = char(lines(i));
                        if ~contains(ln, 'regexprep(') || ~isempty(regexp(ln, '^\s*%', 'once'))
                            continue
                        end
                        pats = regexp(ln, 'regexprep\([^'']*''([^'']*)''', 'tokens');
                        for p = 1:numel(pats)
                            pat = pats{p}{1};
                            if ~isempty(regexp(pat, keys, 'once')) && ...
                                    ~(startsWith(pat, '(?m)^') || startsWith(pat, '^'))
                                bad{end+1} = sprintf('%s:%d: %s', f, i, strtrim(ln)); %#ok<AGROW>
                            end
                        end
                    end
                end
            end
            tc.verifyTrue(isempty(bad), sprintf(['%d header-key rewrites can match a ' ...
                'sibling key mid-line (anchor them: ''(?m)^(\\s*KEY=\\s*)[^\\n]*''):\n%s'], ...
                numel(bad), strjoin(bad, newline)));
        end
    end
end
