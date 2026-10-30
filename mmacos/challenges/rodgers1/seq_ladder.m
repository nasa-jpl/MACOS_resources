function L = seq_ladder()
%SEQ_LADDER  The reference-freedom ladder at the CODE V .seq truth.
%
%   L = SEQ_LADDER()
%
%   Runs STRICT_LADDER's pre-existing rungs -- strict / bestfoc / -tilt /
%   -astig -- on Rodgers' own stage-2 and stage-3 conics at EPD 5000, with the
%   M1 hole, on his 15-point half box.  This is what SIZES the one thing the
%   .seq files cannot contain: the CODE V field-map ANALYSIS convention.
%
%   The rungs are defined a priori in STRICT_LADDER's docstring, ordered by
%   permissiveness, and were written before these .seq files existed.  Reading
%   the result off the ladder is therefore a measurement: the adjacent rungs
%   BRACKET the answer (piston-only is 2x high, +astig overshoots below 1x),
%   and two different designs must agree on the same rung.
%
%   STAGE 4 IS DELIBERATELY OMITTED.  STRICT_LADDER sets conics only; it does
%   not apply the stage-4 M2/M3 rigid body, so a stage-4 row would describe a
%   design nobody built.  Adding the rigid body to the ladder is the remaining
%   piece of this measurement.
%
%   Writes rodgers1_seq_ladder.mat.  See PACKET.md Addendum 8.7.

    here = fileparts(mfilename('fullpath'));
    root = fileparts(fileparts(here));
    run(fullfile(root,'mmacos_setup.m'));
    addpath(here);
    P = rodgers_common('seq');
    lam_nm = P.lambda_m*1e9;

    cfg = { struct('st',2,'K',P.K_nom,'gt',P.gt.s2_box), ...
            struct('st',3,'K',P.K_s3, 'gt',P.gt.s3_box) };
    L = struct('stage',cell(1,numel(cfg)));

    for i = 1:numel(cfg)
        C = cfg{i};
        banner('LADDER AT THE .seq TRUTH -- his stage %d conics', C.st);
        Li = strict_ladder([], 'seq', C.K);
        g  = C.gt * lam_nm;
        L(i).stage = C.st;  L(i).L = Li;  L(i).gt = g;
        fprintf('\n  Rodgers reported (nm): min %.3f  max %.3f  avg %.3f\n', g(1), g(2), g(3));
        fprintf('  %-9s %10s %8s   %10s %8s\n','rung','max nm','max x','avg nm','avg x');
        for f = {'strict','bestfoc','notilt','noastig'}
            v = Li.(f{1})*1e9;
            L(i).(f{1}) = [max(v) mean(v)];
            fprintf('  %-9s %10.3f %7.3fx   %10.3f %7.3fx\n', ...
                    f{1}, max(v), max(v)/g(2), mean(v), mean(v)/g(3));
        end
    end

    banner('LADDER SUMMARY (max ratio vs Rodgers, .seq truth)');
    fprintf('  rung      %10s %10s\n','S2','S3');
    for f = {'strict','bestfoc','notilt','noastig'}
        fprintf('  %-9s %9.3fx %9.3fx\n', f{1}, ...
                L(1).(f{1})(1)/L(1).gt(2), L(2).(f{1})(1)/L(2).gt(2));
    end
    fprintf(['\n  READ: piston-only (the ruling) is ~2x high; removing per-field\n' ...
             '  TIP/TILT lands BOTH designs at ~1.09x; removing astigmatism as\n' ...
             '  well OVERSHOOTS below 1x.  The convention is bracketed.\n']);

    save(fullfile(here,'rodgers1_seq_ladder.mat'),'L');
    fprintf('\nsaved rodgers1_seq_ladder.mat\n');
end

function banner(varargin)
    fprintf('\n=================================================================\n');
    fprintf(' %s\n', sprintf(varargin{:}));
    fprintf('=================================================================\n');
end
