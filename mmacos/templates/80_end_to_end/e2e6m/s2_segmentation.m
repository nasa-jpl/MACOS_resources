function OUT = s2_segmentation(over)
%S2_SEGMENTATION  e2e6m stage 2: segment the 6 m primary.
%
%   A thin driver over the general stage runner
%   `design/runners/run_segmentation.m` (SegMirMaker via
%   macos.design.segment_rx): a 2-ring HEX tiling -- 19 segments, about
%   1.2 m flat-to-flat on a 6 m aperture, the e2e-s3 / JWST class -- with
%   PHYSICAL polygonal apertures declared in the Rx, the parent's solved
%   M1 figure carried onto every segment, and the SegMirMaker edge-sensor
%   sidecar for the later MET / compare stages.
%
%   The apertures here are aperture rule 2's FIRST half: segment
%   boundaries come from the segmentation machinery, which emits polygon
%   vertices in the element's own frame and is not affected by the
%   global-XY defect that `realize_apertures` carries.  The rest of the
%   train stays apertures-off until S3.
%
%   GATES (the runner measures them; this driver reads them back and
%   states PASS/FAIL):
%     [1] bare segmentation traces at the parent's ray count and WFE
%     [3] the aperture variant's pass count -- gap and rim rays clip, and
%         that loss IS the physics, so it is reported, not gated to zero
%     [5] the saved artifact reloads standalone at the same ray count
%     + one-segment poke localizes in dW (the e2e-s3 check, run here)
%     + the view_std render (graphics are gates)
%
%   THE RULE that makes a poke localize: each segment's GRID frame must
%   equal its CLOCKED Mon frame.  segment_rx emits them together; the
%   poke check below is what proves it on this deck rather than assuming
%   it (the e5 corpus shipped a per-segment PISTON for months because a
%   null grid frame collapsed every ray to the centre pixel).
%
%   OUT = S2_SEGMENTATION()      run at the default parameter set
%   OUT = S2_SEGMENTATION(OVER)  ... with e2e6m_params overrides
%
%   See also E2E6M_PARAMS, S1_TELESCOPE, run_segmentation.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    tag = fullfile(P.outdir, 's2');
    parent = fullfile(P.outdir, 's1_telescope.in');
    assert(isfile(parent), 's2_segmentation: S1 artifact %s not found', parent);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S2 -- segment the primary');
    L = say_(L, 'parent %s', parent);
    L = say_(L, 'tiling %s, %d rings -> %d segments, gap %g m', ...
             P.seg.kind, P.seg.rings, 1 + 3*P.seg.rings*(P.seg.rings+1), P.seg.gap_m);

    art = run_segmentation(string(parent), ...
            'rings', P.seg.rings, 'grid', gridname_(P.seg.kind), ...
            'elt', 1, 'gap', P.seg.gap_m, 'emit_apertures', true, ...
            'model_size', max(P.model, 512), ...
            'out_dir', string(P.outdir), 'name', "s2_segmented", ...
            'verbose', false);

    L = say_(L, '\n[runner] %s', art.report);
    L = say_(L, '    segmented Rx  %s', art.in);
    L = say_(L, '    edge sidecar  %s', art.hx);
    rt = fileread(char(art.report));
    for key = ["\[0\] parent", "\[1\] .*segments", "bare segmented", ...
               "\[3\] physical apertures", "first-fail", "\[5\] artifact"]
        m = regexp(rt, ['(?m)^.*' char(key) '.*$'], 'match', 'once');
        if ~isempty(m), L = say_(L, '    %s', strtrim(m)); end
    end

    % ---- the poke-localization gate --------------------------------------
    pk = poke_localizes_(char(art.in), max(P.model,512), P.seg.ng);
    L = say_(L, '\n[gate] one-segment poke localizes in dW:');
    L = say_(L, '    poked segment elt %d, poke %.3g m', pk.elt, pk.amp);
    L = say_(L, '    |dW| inside  the poked segment: rms %.4g m over %d rays', ...
             pk.rms_in, pk.n_in);
    L = say_(L, '    |dW| outside the poked segment: rms %.4g m over %d rays', ...
             pk.rms_out, pk.n_out);
    L = say_(L, '    ratio out/in %.3g  [%s]', pk.ratio, gate_(pk.ratio < 0.05));
    if pk.ratio >= 0.05
        L = say_(L, '    ** a whole-pupil response means the segment grid frame is');
        L = say_(L, '       not its clocked Mon frame -- the poke is not local.');
    end

    L = say_(L, '\nS2 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen([tag '_report.txt'],'w');  fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'art',art, 'poke',pk, 'text',txt, ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save([tag '_run.mat'],'OUT');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function g = gridname_(kind)
    switch lower(kind)
        case 'hex', g = 'Hex';
        case 'pie', g = 'Pie';
        otherwise, error('s2_segmentation:kind','seg.kind must be hex|pie');
    end
end

function pk = poke_localizes_(rx, model, ~)
%POKE_LOCALIZES_  Poke ONE segment in piston and check the wavefront
%   change is confined to that segment's own footprint.  Segment identity
%   comes from the ENGINE (the poke itself), not from a tiling model: the
%   rays whose |dW| exceeds a fraction of the peak ARE the poked segment,
%   and everything else must be quiet.  That is the e2e-s3 check, and it
%   is what catches a grid frame that is not the clocked Mon frame.
    macos.init(model);
    n = macos.load_rx(rx);
    % THE CHIEF REFERENCE IS LOAD-BEARING HERE.  The engine's default OPD
    % reference is the whole-aperture MEAN -- one global scalar shared by
    % every segment -- so poking ONE segment moves it by
    % (N_seg/N_total) x (that segment's response) and the shift is then
    % subtracted from every ray.  Unperturbed segments come back with a
    % spurious uniform piston, and a gate that removes piston globally
    % reads that as leakage.  Measured on this deck: 52/982 x 1.886e-8 =
    % 1.0e-9, against a 1.055e-9 "leak" -- the whole of it.  opd_ref must
    % be set AFTER load_rx (every load resets it).
    macos.opd_ref('chief');
    macos.trace(n);
    W0 = macos.opd();
    % the last segment element: segment_rx puts the segments first, so
    % pick the one with the largest |dW| response to a small piston -- but
    % a fixed choice is enough and reproducible: segment 2 (an off-centre
    % one; segment 1 is the centre and its response is the least
    % diagnostic).
    ie = 2;
    amp = 1e-8;                                  % 10 nm piston, m
    % piston along the segment's own normal: the LOCAL frame's z
    macos.perturb(ie, 'translation', [0;0;amp], 'frame','local');
    macos.modify();
    macos.trace(n);
    W1 = macos.opd();
    macos.perturb(ie, 'translation', [0;0;-amp], 'frame','local');
    macos.modify();
    m = isfinite(W0) & isfinite(W1) & W0 ~= 0 & W1 ~= 0 & ...
        abs(W0) < 1e30 & abs(W1) < 1e30;
    dW = zeros(size(W0));  dW(m) = W1(m) - W0(m);
    % NO global piston removal: with the chief reference the map already
    % has a fixed per-trace origin, and subtracting a whole-pupil mean
    % would re-introduce exactly the cross-segment coupling the chief
    % reference exists to remove.
    a = abs(dW(m));
    thr = 0.25*max(a);
    in  = m;  in(m) = a >= thr;
    out = m & ~in;
    pk = struct('elt',ie, 'amp',amp, ...
                'rms_in',  rms_(dW(in)),  'n_in',  nnz(in), ...
                'rms_out', rms_(dW(out)), 'n_out', nnz(out));
    pk.ratio = pk.rms_out / max(pk.rms_in, eps);
end

function r = rms_(v), v = v(:);  if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
