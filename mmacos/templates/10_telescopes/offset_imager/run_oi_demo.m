%RUN_OI_DEMO  Batch RUNNER for the live adjacent-problem beat.
%
%   This is the demo-day launcher: it exists so the solve can be kicked
%   off in a SEPARATE MATLAB at the top of the demo (Dave's ruling
%   2026-08-28: background structure -- solve starts, the Twyman-Green
%   beat runs while it works, reveal after) without blocking the console
%   the beat is driven from.
%
%   Usage (from a shell, backgrounded):
%
%     OI_DEMO_WIDTH=12 \
%     MACOS_HOME=/home/dcr/dev/macos/macos_f90 \
%     matlab -batch "run('<...>/offset_imager/run_oi_demo.m')" \
%       > /tmp/oi_demo.log 2>&1 &
%
%   The frontier PREDICTION prints within seconds of launch (before any
%   solving), so the narration can quote it while the solve runs:
%
%     grep -A6 'frontier prediction' /tmp/oi_demo.log
%
%   and the reveal is the verdict block, re-printable at any time from the
%   file the wrapper wrote beside the figures:
%
%     grep -n 'verdict *:' /tmp/oi_demo.log        % the path
%     cat <that path>
%
%   Environment:
%     OI_DEMO_WIDTH     box full-width, deg (default 12 -- the fallback
%                       spec, deliberately BETWEEN committed steps)
%     OI_DEMO_TAG       artifact prefix (default: the wrapper's timestamp)
%     OI_DEMO_OUTDIR    artifact directory (default demo_adjacent/)
%     OI_DEMO_ITERS     Gauss-Newton cap (default 1 -- the pinned demo knob)
%     OI_DEMO_NSOLVE    solve grid n, must be ODD (default 5 -- the pinned
%                       demo knob; do NOT lower it to buy time, see
%                       OI_DEMO_STEP's help and the README knob study)
%
%   This file is a SCRIPT and ends in exit(0) because it is a batch
%   runner; OI_DEMO_STEP itself never calls exit (the standing trap:
%   exit(0) belongs in the runner, never in the wrapper).
%
%   See also OI_DEMO_STEP, OI_WALK, demo_adjacent/REHEARSAL.md.

oi_demo_here_ = fileparts(mfilename('fullpath'));
run(fullfile(oi_demo_here_,'..','..','..','mmacos_setup.m'));
addpath(oi_demo_here_);

oi_demo_w_ = 12;   oi_demo_it_ = 1;   oi_demo_ns_ = 5;
if ~isempty(getenv('OI_DEMO_WIDTH')),  oi_demo_w_  = str2double(getenv('OI_DEMO_WIDTH'));  end
if ~isempty(getenv('OI_DEMO_ITERS')),  oi_demo_it_ = str2double(getenv('OI_DEMO_ITERS'));  end
if ~isempty(getenv('OI_DEMO_NSOLVE')), oi_demo_ns_ = str2double(getenv('OI_DEMO_NSOLVE')); end
oi_demo_nv_ = {'gn_iters', oi_demo_it_, 'nsolve', oi_demo_ns_};
if ~isempty(getenv('OI_DEMO_TAG'))
    oi_demo_nv_ = [oi_demo_nv_, {'tag', getenv('OI_DEMO_TAG')}];
end
if ~isempty(getenv('OI_DEMO_OUTDIR'))
    oi_demo_nv_ = [oi_demo_nv_, {'outdir', getenv('OI_DEMO_OUTDIR')}];
end

fprintf('OI-DEMO START %s  (box %g deg)\n', ...
        datestr(now,31), oi_demo_w_); %#ok<TNOW1,DATST>
oi_demo_t0_ = tic;

OUT = oi_demo_step(oi_demo_w_, oi_demo_nv_{:});

if OUT.refused
    fprintf('OI-DEMO REFUSED %s (%.2f min)\n', ...
            datestr(now,31), toc(oi_demo_t0_)/60); %#ok<TNOW1,DATST>
else
    fprintf('OI-DEMO DONE  %s  (%.2f min, verdict %s, map max %.1f nm)\n', ...
            datestr(now,31), toc(oi_demo_t0_)/60, ...
            OUT.verdict, OUT.map.max_nm); %#ok<TNOW1,DATST>
    fprintf('OI-DEMO verdict : %s\n', OUT.files.verdict);
    fprintf('OI-DEMO map     : %s\n', OUT.files.map);
    fprintf('OI-DEMO layout  : %s\n', OUT.files.layout);
    fprintf('OI-DEMO fields  : %s\n', OUT.files.fields);
end
exit(0);
