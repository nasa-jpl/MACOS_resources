function OUT = r3_sensitivities(over)
%R3_SENSITIVITIES  e2e6m round 2: the linear model of the DM-bearing train.
%
%   Round 1's S4 on the round-2 deck: harvests dwdx (rigid-body 6-DOF
%   per optic -- now INCLUDING DM1/DM2 and the 8 OAPs -- plus the 19
%   segments as ONE RIGID GROUP), dwdz (segment MonZernike figure) and
%   dwdgrid (segment influence basis) on `r1_seg_prop.in`, the
%   diffraction deck R4 propagates.
%
%   METRIC TAG: OPD at the CORONAGRAPH exit pupil (`ox.wf_elt` --
%   consumers pair engine OPD with the Jacobian ONLY at that surface;
%   design/src/jacobian_check.m enforces it, R0's lesson), per-field
%   FEX reset, field set = centre + 4 corners of the design box.
%
%   OUT = R3_SENSITIVITIES()      defaults
%   OUT = R3_SENSITIVITIES(OVER)  with e2e6m_r2_params overrides
%
%   See also E2E6M_R2_PARAMS, ../e2e6m/s4_sensitivities,
%   run_sensitivities, jacobian_check, group_exhibit.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','..','sensitivities'));
    addpath(fullfile(here,'..','..','..','design','src'));

    rx = fullfile(P.outdir, 'r1_seg_prop.in');
    assert(isfile(rx), 'r3_sensitivities: %s not found -- run r1_coro first', rx);

    [optics, names] = optic_indices_(rx);
    segs = find(strcmp(names(optics), 'Segment'));
    nseg = numel(segs);
    assert(nseg > 0, 'r3_sensitivities: %s declares no segments', rx);

    groups = containers.Map();
    groups('PM') = optics(segs);

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R3 -- sensitivities, DM-bearing train');
    L = say_(L, 'deck   %s', rx);
    L = say_(L, 'metric OPD at the CORONAGRAPH exit pupil (ox.wf_elt), per-field FEX;');
    L = say_(L, '       field set = centre + 4 corners of the +-%g'' box, %g nm', ...
             P.tel.fov_arcmin, P.lambda_m*1e9);
    L = say_(L, 'optics %d perturbed (%d segments + %d rest incl. DM1/DM2 + 8 OAPs);', ...
             numel(optics), nseg, numel(optics)-nseg);
    L = say_(L, '       PM group = %d members', numel(groups('PM')));

    art = run_sensitivities(string(rx), ...
            'fov_rad', deg2rad(P.tel.fov_arcmin/60), ...
            'elts', optics(:), ...
            'groups', groups, ...
            'zmodes_fig',  P.sn.zmodes_fig, ...
            'zmodes_grid', P.sn.zmodes_grid, ...
            'ng', P.sn.ng, ...
            'model_size', P.sn.model, ...
            'out_dir', string(P.outdir), 'name', "r3", ...
            'verbose', false);

    L = say_(L, '\n[runner] %s', art.report);
    rt = fileread(char(art.report));
    for key = ["segments:", "dwdxall", "group channel", "dwdzall", "dwdgall", ...
               "cond", "rank"]
        m = regexp(rt, ['(?m)^.*' char(key) '.*$'], 'match', 'dotexceptnewline');
        for q = 1:min(numel(m),4), L = say_(L, '    %s', strtrim(m{q})); end
    end

    reps = intersect(groups('PM'), optics(segs([1, min(2,nseg), min(8,nseg)])));
    group_exhibit(art.ox, groups, char(art.report), 'members', reps(:).');
    L = say_(L, '\n[group exhibit] appended to %s (members %s)', ...
             art.report, mat2str(reps(:).'));

    % ---- the R0 closure gate, run on THIS harvest ------------------------
    % jacobian_check at the harvest's own wf_elt: sample a segment, a DM
    % and an OAP, all six DOFs.  This bakes R0's fix into the stage gate
    % rather than trusting it.
    sample = [optics(segs(min(2,nseg))), pick_(rx,'DM1'), pick_(rx,'OAP4')];
    chk = jacobian_check(rx, art.ox, 'elts', sample(:), ...
            'model', P.sn.model, 'verbose', false);
    L = say_(L, '\n[closure] jacobian_check at wf_elt %d on elts %s:', ...
             chk.wf_elt, mat2str(sample));
    L = say_(L, '    worst rel %.3g over %d pairs (%d null)  [%s]', ...
             chk.worst, numel(chk.rel), chk.n_null, ...
             gate_(chk.worst < P.ts.tol_linear));

    art_light = rmfield_if_(art, {'ox','oz','og','os'});
    OUT = struct('P',P, 'art',art_light, 'optics',optics, 'chk',chk, ...
                 'groups',{{groups}}, 'text','', ...
                 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'r3_run.mat'),'OUT');

    fp = fullfile(P.outdir,'r3_sens.fp.json');
    S = struct('dwdxall',art.ox.dwdxall);
    if ~isempty(art.oz), S.dwdzall = art.oz.dwdxall; end
    if ~isempty(art.og), S.dwdgall = art.og.dwdxall; end
    jac_fingerprint('write', fp, S, struct( ...
        'rx', string(rx), 'fov_arcmin', P.tel.fov_arcmin, ...
        'model_size', P.sn.model, 'n_optics', numel(optics), ...
        'n_segments', nseg, 'group', "PM(19)", ...
        'zmodes_fig', mat2str(P.sn.zmodes_fig), ...
        'zmodes_grid', mat2str(P.sn.zmodes_grid), ...
        'when', string(datestr(now,31)))); %#ok<TNOW1,DATST>
    L = say_(L, '\n[fingerprint] %s (the .mat is %.0f MB and gitignored)', ...
             fp, dir_mb_(fullfile(P.outdir,'r3_sens.mat')));

    L = say_(L, '\nR3 sensitivities DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'r3_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT.text = txt;
    save(fullfile(P.outdir,'r3_run.mat'),'OUT');
end

% =========================================================================
function i = pick_(rx, name)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    i = find(strcmp(nm, name), 1);
    assert(~isempty(i), 'r3: %s not found in %s', name, rx);
end

function S = rmfield_if_(S, f)
    for k = 1:numel(f)
        if isfield(S, f{k}), S = rmfield(S, f{k}); end
    end
end

function mb = dir_mb_(p)
    d = dir(p);
    if isempty(d), mb = 0; else, mb = d.bytes/1e6; end
end

function [idx, kinds] = optic_indices_(rx)
%OPTIC_INDICES_  Real, perturbable optics: Segment and Reflector.
    kinds = regexp(fileread(rx), '^\s*Element=\s*(\S+)', 'tokens', 'lineanchors');
    kinds = cellfun(@(c) c{1}, kinds, 'UniformOutput', false);
    idx = find(ismember(kinds, {'Segment','Reflector'}));
end

function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
