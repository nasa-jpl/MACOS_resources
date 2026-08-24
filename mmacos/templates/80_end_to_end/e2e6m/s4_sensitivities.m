function OUT = s4_sensitivities(over)
%S4_SENSITIVITIES  e2e6m stage 4: the linear model of the full train.
%
%   A thin driver over the general stage runner
%   `design/runners/run_sensitivities.m`.  It harvests the three
%   wavefront Jacobian channels on the FULL train -- segmented primary
%   PLUS the coronagraph back end, one deck -- each in the canonical
%   stacked form `wall = J*x + w0`:
%
%     dwdx     rigid-body 6-DOF per optic, LOCAL/TElt triads, PLUS the
%              19 segments as ONE RIGID GROUP (GPERTURB).  The group's
%              six columns are appended after the per-element block.
%     dwdz     segment-LOCAL MonZernike figure modes
%     dwdgrid  per-segment grid-poke channel on a grid-augmented Rx,
%              through a Gram-Schmidt Zernike influence basis
%
%   WHY THE GROUP MATTERS, and why it is the exhibit.  A per-segment
%   error budget and a whole-PM error budget are different questions.
%   Move 19 segments as one body and their responses ADD where they are
%   alike and CANCEL where they are not; the group/member column-norm
%   ratio measures which.  A ratio near the member count is the "19 alike
%   columns" case and a per-segment budget is conservative; a ratio below
%   1 is intra-group COMPENSATION, and there a per-segment budget
%   OVERSTATES what the assembly actually does.  `group_exhibit` writes
%   that table into the report.
%
%   THE DECK IS THE DIFFRACTION DECK (`s3_seg_prop.in`), not the bare
%   geometric train.  Two reasons: it is the deck S5 propagates, so the
%   linear model and the contrast series describe the same object; and
%   the OPD needs an exit-pupil anchor, which the geometric train does
%   not have (its nElt-1 is a powered mirror, and FEX refuses).  The
%   Return/NF surfaces are transparent to rays, so the geometric harvest
%   is unaffected by their presence.
%
%   METRIC TAG: OPD at the CORONAGRAPH exit pupil (the `ExitPupil`
%   element of the spliced train), per-field exit-pupil reset by FEX,
%   field set = centre + 4 corners of the +-0.35' design box.
%
%   OUT = S4_SENSITIVITIES()      run at the default parameter set
%   OUT = S4_SENSITIVITIES(OVER)  ... with e2e6m_params overrides
%
%   See also E2E6M_PARAMS, S3_CORO, run_sensitivities, group_exhibit.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    setup_(here);
    P = e2e6m_params(over);
    if isempty(P.outdir), P.outdir = here; end
    addpath(fullfile(here,'..','..','..','sensitivities'));

    rx = fullfile(P.outdir, P.sn.rx);
    assert(isfile(rx), 's4_sensitivities: %s not found -- run S3 first', rx);

    % ---- who is a real optic --------------------------------------------
    % The diffraction deck carries Return/Reference surfaces that are not
    % hardware and must not be perturbed.  Pick the optics by KIND from
    % the deck rather than by an index list that would drift the moment
    % the segment count or the quartet layout changes.
    [optics, names] = optic_indices_(rx);
    segs = find(strcmp(names(optics), 'Segment'));
    nseg = numel(segs);
    assert(nseg > 0, 's4_sensitivities: %s declares no segments', rx);

    groups = containers.Map();
    groups('PM') = optics(segs);          % the 19 segments as one body

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m S4 -- sensitivities');
    L = say_(L, 'deck   %s', rx);
    L = say_(L, 'metric OPD at the CORONAGRAPH exit pupil, per-field FEX reset;');
    L = say_(L, '       field set = centre + 4 corners of the +-%g'' box, %g nm', ...
             P.tel.fov_arcmin, P.lambda_m*1e9);
    L = say_(L, 'optics %d perturbed (%d segments + %d rest); PM group = %d members', ...
             numel(optics), nseg, numel(optics)-nseg, numel(groups('PM')));

    art = run_sensitivities(string(rx), ...
            'fov_rad', deg2rad(P.tel.fov_arcmin/60), ...
            'elts', optics(:), ...
            'groups', groups, ...
            'zmodes_fig',  P.sn.zmodes_fig, ...
            'zmodes_grid', P.sn.zmodes_grid, ...
            'ng', P.sn.ng, ...
            'model_size', P.sn.model, ...
            'out_dir', string(P.outdir), 'name', "s4", ...
            'verbose', false);

    L = say_(L, '\n[runner] %s', art.report);
    rt = fileread(char(art.report));
    for key = ["segments:", "dwdxall", "group channel", "dwdzall", "dwdgall", ...
               "cond", "rank"]
        m = regexp(rt, ['(?m)^.*' char(key) '.*$'], 'match', 'dotexceptnewline');
        for q = 1:min(numel(m),4), L = say_(L, '    %s', strtrim(m{q})); end
    end

    % ---- the group-vs-member exhibit ------------------------------------
    % One representative per RING, not all 19: a 19-row table says the
    % same thing 19 times.  Segment 1 is the centre, 2 the first ring,
    % 8 the second.
    reps = intersect(groups('PM'), optics(segs([1, min(2,nseg), min(8,nseg)])));
    group_exhibit(art.ox, groups, char(art.report), 'members', reps(:).');
    L = say_(L, '\n[group exhibit] appended to %s (members %s)', ...
             art.report, mat2str(reps(:).'));
    rt2 = fileread(char(art.report));
    tail = regexp(rt2, '(?m)^.*(group|ratio|PM).*$', 'match', 'dotexceptnewline');
    for q = max(1,numel(tail)-12):numel(tail)
        L = say_(L, '    %s', strtrim(tail{q}));
    end

    L = say_(L, '\nS4 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'s4_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);

    OUT = struct('P',P, 'art',art, 'optics',optics, 'groups',{{groups}}, ...
                 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'s4_run.mat'),'OUT','-v7.3');
end

% =========================================================================
function setup_(here)
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
end

function [idx, kinds] = optic_indices_(rx)
%OPTIC_INDICES_  Elements that are real, perturbable optics.
%   Segment and Reflector are hardware; Return / Reference / FocalPlane
%   are propagation structures and mask/pupil markers, which a rigid-body
%   perturbation of "the optics" must not touch.
    kinds = regexp(fileread(rx), '^\s*Element=\s*(\S+)', 'tokens', 'lineanchors');
    kinds = cellfun(@(c) c{1}, kinds, 'UniformOutput', false);
    idx = find(ismember(kinds, {'Segment','Reflector'}));
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});
    L{end+1} = s;
    fprintf('%s\n', s);
end
