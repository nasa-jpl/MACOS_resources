function OUT = r0_closure_check()
%R0_CLOSURE_CHECK  e2e6m R0.3: the round-1 check table, at the RIGHT surface.
%
%   Re-runs round 1's engine-vs-linear-model check against the stored S4
%   Jacobian with the ONE fix R0.1 established: the engine OPD is
%   evaluated at the harvest's wf_elt (the coronagraph exit-pupil
%   Return, nElt-1), not at nElt (the Science focal plane).  Via the new
%   shared design/src/jacobian_check.m.
%
%   GATE: all six rigid-body DOFs close at FD-linearity level on every
%   sampled element -- which un-shrinks the S5 control basis from
%   piston-only to the full six.
%
%   Sampled: Seg1 (center), Seg2 (ring 1), Seg19 (ring 2, the round-1
%   fingerprint row), and M2 (elt 20, first non-segment optic).

    here = fileparts(mfilename('fullpath'));
    r1   = fullfile(here, '..', 'e2e6m');
    run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
    addpath(r1);
    addpath(fullfile(here, '..', '..', '..', 'design', 'src'));
    P  = e2e6m_params(struct());
    rx = fullfile(r1, P.sn.rx);
    S4 = load(fullfile(r1, 's4_sens.mat'), 'ox');

    ELTS = [1 2 19 20];
    TOL  = P.ts.tol_linear;      % 0.05, the round-1 gate value
    dofn = {'Rx','Ry','Rz','Tx','Ty','Tz'};

    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R0.3 -- six-DOF closure at wf_elt');
    L = say_(L, 'deck %s', rx);
    L = say_(L, 'wf_elt %d (the harvest surface); pokes %g nrad / %g nm; tol %g', ...
             S4.ox.wf_elt, P.ts.d_rot*1e9, P.ts.d_trans*1e9, TOL);

    chk = jacobian_check(rx, S4.ox, 'elts', ELTS(:), ...
            'd_rot', P.ts.d_rot, 'd_trans', P.ts.d_trans, ...
            'model', P.sn.model);
    for k = 1:numel(chk.rel)
        L = say_(L, '    elt %2d %s: |engine| %.4g  |model| %.4g  rel.err %.3g', ...
                 chk.elt(k), dofn{chk.dof(k)+1}, chk.n_eng(k), ...
                 chk.n_mod(k), chk.rel(k));
    end
    L = say_(L, '    worst over %d (elt, DOF) pairs: %.3g  [%s]', ...
             numel(chk.rel), chk.worst, gate_(chk.worst < TOL));
    L = say_(L, '\nround-1 reference: the SAME table at nElt gave rel.err');
    L = say_(L, '~1.0 on the active rotation/translation axes and ~55x on');
    L = say_(L, 'the null ones (s5_report.txt [1]); only Tz closed.');
    L = say_(L, '\nR0.3 DONE in %.1f min', toc(t0)/60);

    txt = strjoin(L, newline);
    fid = fopen(fullfile(here,'r0_closure_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('chk',chk, 'tol',TOL, 'pass',chk.worst < TOL, 'text',txt);
    save(fullfile(here,'r0_closure.mat'), 'OUT');
end

function s = gate_(ok), if ok, s = 'PASS'; else, s = 'FAIL'; end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
