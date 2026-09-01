% RUN_OFFAXIS_WFE  Does the WAVEFRONT FLOOR move when the pupil goes OFF AXIS?
%
%   THE ISOLATION EXPERIMENT.  Everything here is the descent's committed
%   machinery -- the same closure, the same merit, the same DOFs, the same
%   solver settings, the same starting designs -- with exactly ONE variable
%   changed: the pupil decenter h.  h = 0 is the coaxial control and must
%   reproduce the descent's own recorded floors; h > 0 is the same design
%   used OFF AXIS.  Any difference between the two columns is attributable to
%   the decenter and to nothing else.
%
%   WHY THIS AND NOT A MERSENNE SEED.  The off-axis Mersenne is the textbook
%   off-axis afocal seed and it is measured separately by
%   RUN_OFFAXIS_MERSENNE -- but it cannot be carried through DESCENT_CLOSE,
%   which is singular on a front end that has already met the specification
%   (measured: the pair arrives at mirror 2 with y = 0.016667 m and
%   u = 0.000000, so the closure's b = (yout - ym)/u2 has a zero numerator;
%   handed five mirrors it inserts a strong third mirror to BREAK the
%   Mersenne and then re-closes with a 2.74 m lever, giving a design that is
%   paraxially exact and traces at M = 26.73).  So the Mersenne answers "what
%   can the family do with the pupil requirement dropped" and THIS answers
%   "what does moving off axis buy, all else equal".  Neither substitutes for
%   the other and the pair of them is the slice's evidence.
%
%   THE RIGID-BODY PROBE IS THE THING THIS SUPERSEDES.  That probe was free to
%   move each mirror by +-15 deg and +-300 mm and used 0.92 deg / 3.1 mm --
%   it never left the coaxial basin, so it measured a local optimum and not
%   the family.  A decenter is not reachable by rigid-body perturbation of
%   the mirrors at all: it moves the PUPIL, i.e. which part of each parent
%   surface the light uses.  That is the variable the probe could not vary.
%
%   OW_N        comma list of N ("4,5,7")
%   OW_H        comma list of decenters, m ("0,0.55,0.75,1.0")
%   OW_EVALS    evaluations per round (400)
%   OW_ROUNDS   restart rounds (2)
%   OW_TAG      artifact prefix ('OAW')
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing'));
addpath(fullfile(up,'wall')); addpath(fullfile(up,'descent'));

Ns  = str2double(strsplit(getenv_d('OW_N','4,5,7'), ','));
Hs  = str2double(strsplit(getenv_d('OW_H','0,0.55,0.75,1.0'), ','));
wEv = str2double(getenv_d('OW_EVALS','400'));
wRd = str2double(getenv_d('OW_ROUNDS','2'));
tag = getenv_d('OW_TAG','OAW');

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central';  P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8;  P.solve.tol_x=1e-9;  P.solve.tol_opt=1e-8;
P.solve.max_fev = wEv;
dofs = {'conic','radius','spacing','tilt'};

fprintf('\n==== OFF-AXIS WAVEFRONT FLOOR vs PUPIL DECENTER ====\n');
fprintf('  DOFs %s; pupil ladder NOT scored.\n', strjoin(dofs,', '));
fprintf('  coaxial reference: 3841.8 nm (N=7, no-tilt control); target 71 nm.\n\n');

rows = struct('N',{},'h',{},'wfe0',{},'wfe',{},'gain_pct',{},'M',{}, ...
              'coll',{},'lost',{},'union',{},'nfev',{},'deck',{},'ok',{});
for N = Ns
    D = start_design_(P, N, up, here);
    if isempty(D), fprintf('  N=%d: no starting design -- skipped\n', N); continue; end
    D.ngrid = P.ngrid;   D.bias_deg = P.bias_deg;

    for h = Hs
        Dh = D;   Dh.decenter = h;
        lbl = sprintf('N%d_h%g', N, h);
        d0  = fullfile(here, sprintf('afocal4_%s_%s_start.in', tag, lbl));
        Pr  = P;   Pr.pack.enforce = false;

        % ---- the start: does it even trace off axis? ---------------------
        try
            o0 = descent_build(Pr, Dh, d0, 'oa_fields',P.Fsolve, ...
                               'verify',true, 'quiet',true);
        catch ME
            fprintf('  N=%d h=%.2f  BUILD FAILED: %s\n', N, h, ME.message);
            continue;
        end
        lost0 = 0;  if ~isempty(o0.offaxis), lost0 = o0.offaxis.nlost; end
        S0 = afocal4_score(P, d0, 'fields',P.Fsolve, ...
                           'nodes',P.solve.nodes_score, 'pupil',false);
        fprintf(['  N=%d h=%.2f m  start WFE %10.1f nm | traced M %8.4f, ' ...
                 'coll %9.1f urad, %d ray(s) lost\n'], N, h, S0.wfe_max_nm, ...
                o0.traced.mag, o0.traced.collimation_urad, lost0);

        % ---- the floor ---------------------------------------------------
        deck = fullfile(here, sprintf('afocal4_%s_%s.in', tag, lbl));
        Dc = Dh;   mprev = Inf;   nf = 0;   okrun = true;
        for r = 1:wRd
            try
                Rr = descent_solve(P, Dc, 'dofs',dofs, 'deck',deck, ...
                         'pupil',false, 'max_iter',400, ...
                         'label',sprintf('%s r%d',lbl,r), 'quiet',true);
            catch ME
                fprintf('    round %d FAILED: %s\n', r, ME.message);
                okrun = false;   break;
            end
            m = Rr.S.merit;   nf = nf + Rr.nfev;
            fprintf('    round %d: %4d evals, %5.1f min, xfl %d, WFE %10.1f nm\n', ...
                    r, Rr.nfev, Rr.seconds/60, Rr.exitflag, Rr.S.wfe_max_nm);
            Dc = Rr.D;   Dc.decenter = h;   % the decenter is NOT a DOF
            if isfinite(mprev) && (mprev-m)/max(abs(mprev),eps) < 1e-6, break; end
            mprev = m;
        end

        Dr = Dc;   Dr.ngrid = P.ngrid;   Dr.decenter = h;
        o1 = descent_build(Pr, Dr, deck, 'oa_fields',P.Fsolve, ...
                           'verify',true, 'quiet',true);
        S1 = afocal4_score(P, deck, 'fields',P.Fsolve, ...
                           'nodes',P.solve.nodes_score, 'pupil',false);
        K  = afocal4_union(deck, 'fields',P.Fsolve, ...
                 'body_k',  getf_(P.pack,'union_body_k',1.15), ...
                 'body_pad',getf_(P.pack,'union_body_pad',0.015), 'quiet',true);
        lost1 = 0;  if ~isempty(o1.offaxis), lost1 = o1.offaxis.nlost; end
        g = 100*(1 - S1.wfe_max_nm/S0.wfe_max_nm);
        fprintf(['    FLOOR  WFE %10.1f nm  (%.1f %% below start, %.0fx target) | ' ...
                 'M %8.4f, union %+8.1f mm, %d lost\n\n'], S1.wfe_max_nm, g, ...
                S1.wfe_max_nm/71, o1.traced.mag, K.floor_m*1e3, lost1);

        rows(end+1) = struct('N',N,'h',h,'wfe0',S0.wfe_max_nm,'wfe',S1.wfe_max_nm, ...
            'gain_pct',g,'M',o1.traced.mag,'coll',o1.traced.collimation_urad, ...
            'lost',lost1,'union',K.floor_m,'nfev',nf,'deck',deck,'ok',okrun); %#ok<SAGROW>
        save(fullfile(here, sprintf('offaxis_%s.mat', tag)), 'rows','P','-v7.3');
    end
end

fprintf('==== THE OFF-AXIS WAVEFRONT FLOOR ====\n');
fprintf('  %3s %7s %12s %12s %8s %9s %9s %7s\n', ...
        'N','h m','start nm','floor nm','x target','M','union mm','lost');
for i = 1:numel(rows)
    r = rows(i);
    fprintf('  %3d %7.2f %12.1f %12.1f %8.0f %9.4f %+9.1f %7d\n', ...
            r.N, r.h, r.wfe0, r.wfe, r.wfe/71, r.M, r.union*1e3, r.lost);
end
fprintf('\n  coaxial control is the h = 0 row of each N.\n');
exit(0);

% =====================================================================
function D = start_design_(P, N, up, here)
%START_DESIGN_  The same starting designs RUN_DESCENT_WFE uses, so the h = 0
%   column is a reproduction of a recorded number and not a new experiment.
    D = [];
    if N == 4
        src = fullfile(up,'afocal4_b2long_343mm.in');
        if ~isfile(src), return; end
        D0 = wall_recover(P, src);
        fo = afocal_first_order([abs(P.parent.R(1)) D0.R2], D0.t1, [false true], ...
                                'D',P.D, 'stop_ahead',P.stop_ahead);
        a  = -fo.y_marginal(2)/fo.u_marginal(2) - D0.fm_standoff;
        D  = struct('N',4, 'R',[abs(P.parent.R(1)) D0.R2], 'convex',[false true], ...
                    't',[D0.t1 a], 'K',D0.K(:).', 'iface',D0.iface, ...
                    'tilt_deg',zeros(1,4));
    else
        f = fullfile(here, '..', 'descent', sprintf('descent_ASC_N%d.mat', N));
        if ~isfile(f), return; end
        Z = load(f,'R');   D = Z.R.D;
    end
end

function v = getenv_d(k,d), v = getenv(k); if isempty(v), v = d; end, end
function v = getf_(s,f,d), if isstruct(s)&&isfield(s,f), v=s.(f); else, v=d; end, end
