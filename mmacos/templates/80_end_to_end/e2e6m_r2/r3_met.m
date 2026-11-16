function OUT = r3_met(over)
%R3_MET  e2e6m round 2: the MET (Dave item 4) -- run_met on this system.
%
%   Round 1 skipped MET, naming the integration work: "reconciling
%   run_met's body list with a Jacobian harvested on the FULL train".
%   Measured, that gap dissolves: `run_met` selects dwdx columns BY
%   BODY (element id) and tolerates extra columns, and the spliced
%   round-2 deck preserves the telescope's element ids (segments 1-19,
%   M2 = 20).  So the committed runner consumes the full-train
%   r3_sens.mat directly:
%
%     bodies   = [Seg1..Seg19, hub = M2(20)]
%     edge     = the SegMirMaker Hx sidecar (round 1's s2_segmentedHx)
%     products = dedx/dldx + the estimator blocks dxde/dxdl/dwde/dwdl
%                that R4's metrology-driven correction consumes
%
%   No aft ring (the perforated-PM line-of-sight study is out of this
%   demo's scope; stated, not implied).
%
%   OUT = R3_MET()      defaults
%   OUT = R3_MET(OVER)  with e2e6m_r2_params overrides
%
%   See also RUN_MET, R3_SENSITIVITIES, ../e2e6m/s2_segmentation.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);

    seg_in = fullfile(P.r1dir, 's2_segmented.in');
    hx     = fullfile(P.r1dir, 's2_segmentedHx.m');
    jac    = fullfile(P.outdir, 'r3_sens.mat');
    assert(isfile(seg_in) && isfile(hx), 'r3_met: round-1 segmentation missing');
    assert(isfile(jac), 'r3_met: %s not found -- run r3_sensitivities first', jac);

    art = run_met(string(seg_in), ...
            'hx', string(hx), ...
            'hub', 20, ...                 % M2 carries the fiducials
            'jac', jac, ...
            'model_size', P.sn.model, ...
            'out_dir', string(P.outdir), 'name', "r3", ...
            'visible', false, 'verbose', true);

    % POINTERS ONLY (the round-1 lesson: saving art verbatim duplicates
    % a 475 MB artifact).  The matrices live in run_met's own r3_met.mat
    % (gitignored); consumers load '-struct' fields from there.
    OUT = struct('met_mat', char(art.mat), 'met_in', char(art.met_in), ...
                 'P', P, 'when', datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r3_met_run.mat'), 'OUT');
end
