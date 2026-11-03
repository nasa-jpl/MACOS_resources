% RUN_DESCENT_WFE  Does the WAVEFRONT FLOOR move with N?
%
%   The descent's premise is that somewhere above four mirrors the 71 nm
%   target becomes reachable.  Every full solve so far has missed it by two
%   orders of magnitude -- but a full solve is a COMPETITION, and a wavefront
%   that will not move might simply be losing an argument to the pupil terms.
%   This asks the wavefront question alone, at each N, from that N's own
%   warm-started design, with every DOF free.
%
%   S4 ran exactly this A/B at N = 4 and measured 8467 nm against a frozen
%   8835 -- 4 %, i.e. the DOFs do not touch it.  If that floor does not fall
%   as N grows, "how many mirrors does the spec need" has an answer of a
%   different SHAPE than the brief expects: not "more than four", but "not
%   reachable by adding mirrors to this family at all".
%
%   WFE_N       comma list of N to run ("4,5,6,7")
%   WFE_EVALS   evaluations per round (400)
%   WFE_ROUNDS  restart rounds (2)
%   WFE_TAG     artifact prefix ('WFE')
run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
here = fileparts(mfilename('fullpath'));   up = fileparts(here);
addpath(here); addpath(up); addpath(fullfile(up,'clearing')); addpath(fullfile(up,'wall'));

wN   = getenv('WFE_N');       if isempty(wN), wN = '4,5,6,7'; end
Ns   = str2double(strsplit(wN,','));
wEv  = str2double(getenv('WFE_EVALS'));   if isnan(wEv), wEv = 400; end
wRd  = str2double(getenv('WFE_ROUNDS'));  if isnan(wRd), wRd = 2;   end
wTag = getenv('WFE_TAG');     if isempty(wTag), wTag = 'WFE'; end
wDof = getenv('WFE_DOFS');
if isempty(wDof), dofs = {'conic','radius','spacing','tilt'};
else,             dofs = strsplit(wDof,','); end

macos.init(256);
P = afocal4_params();
P.pack.enforce = true;
P.solve.fd_type='central'; P.solve.fd_step=1e-4;
P.solve.tol_fun=1e-8; P.solve.tol_x=1e-9; P.solve.tol_opt=1e-8;
P.solve.max_fev = wEv;

fprintf('\n==== WAVEFRONT-ONLY FLOOR vs N ====\n');
fprintf('  DOFs: %s; the pupil ladder is\n', strjoin(dofs,', '));
fprintf('  NOT scored, so the wavefront is not competing with anything.\n\n');

rows = struct('N',{},'wfe0',{},'wfe',{},'gain_pct',{},'nfev',{},'xfl',{}, ...
              'src',{},'deck',{});
for N = Ns
    % the starting design at this N: the committed deck at 4, the ascent's
    % warm-started rung above it.
    if N == 4
        src = fullfile(up,'afocal4_b2long_343mm.in');
        D0  = wall_recover(P, src);
        fo  = afocal_first_order([abs(P.parent.R(1)) D0.R2], D0.t1, [false true], ...
                                 'D',P.D,'stop_ahead',P.stop_ahead);
        a   = -fo.y_marginal(2)/fo.u_marginal(2) - D0.fm_standoff;
        D = struct('N',4,'R',[abs(P.parent.R(1)) D0.R2],'convex',[false true], ...
                   't',[D0.t1 a],'K',D0.K(:).','iface',D0.iface, ...
                   'tilt_deg',zeros(1,4));
        srcname = 'committed 4-mirror deck';
    else
        f = fullfile(here, sprintf('descent_ASC_N%d.mat', N));
        if ~isfile(f)
            fprintf('  N=%d: no ascent rung yet (%s) -- skipped\n', N, f);
            continue;
        end
        Z = load(f,'R');   D = Z.R.D;   srcname = sprintf('ascent rung N%d', N);
    end
    D.ngrid = P.ngrid;   D.bias_deg = P.bias_deg;

    % where it starts, wavefront-wise
    d0 = fullfile(here, sprintf('afocal4_%s_N%d_start.in', wTag, N));
    Pr = P;  Pr.pack.enforce = false;
    descent_build(Pr, D, d0, 'verify',false);
    S0 = afocal4_score(P, d0, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score, ...
                       'pupil',false);
    fprintf('  N=%d  (%s)\n    start  WFE %9.1f nm\n', N, srcname, S0.wfe_max_nm);

    deck = fullfile(here, sprintf('afocal4_%s_N%d.in', wTag, N));
    Dc = D;  mprev = Inf;  nf = 0;  xfl = NaN;
    for r = 1:wRd
        try
            Rr = descent_solve(P, Dc, 'dofs',dofs, ...
                     'deck',deck, 'pupil',false, 'max_iter',400, ...
                     'label',sprintf('N%d wfe r%d',N,r), 'quiet',true);
        catch ME
            fprintf('    round %d FAILED: %s\n', r, ME.message);
            break;
        end
        m = Rr.S.merit;  nf = nf + Rr.nfev;  xfl = Rr.exitflag;
        fprintf('    round %d: %4d evals, %5.1f min, xfl %d, WFE %9.1f nm\n', ...
                r, Rr.nfev, Rr.seconds/60, Rr.exitflag, Rr.S.wfe_max_nm);
        Dc = Rr.D;
        if isfinite(mprev) && (mprev-m)/max(abs(mprev),eps) < 1e-6, break; end
        mprev = m;
    end
    Dr = Dc;  Dr.ngrid = P.ngrid;
    descent_build(Pr, Dr, deck, 'verify',false);
    S1 = afocal4_score(P, deck, 'fields',P.Fsolve, 'nodes',P.solve.nodes_score, ...
                       'pupil',false);
    g = 100*(1 - S1.wfe_max_nm/S0.wfe_max_nm);
    fprintf('    FLOOR  WFE %9.1f nm  (%.1f %% below the start, %.0fx the 71 nm target)\n\n', ...
            S1.wfe_max_nm, g, S1.wfe_max_nm/71);
    rows(end+1) = struct('N',N,'wfe0',S0.wfe_max_nm,'wfe',S1.wfe_max_nm, ...
        'gain_pct',g,'nfev',nf,'xfl',xfl,'src',srcname,'deck',deck); %#ok<SAGROW>
    save(fullfile(here, sprintf('descent_%s.mat', wTag)), 'rows', '-v7.3');
end

fprintf('==== THE WAVEFRONT FLOOR ====\n');
fprintf('  %3s %14s %14s %9s %8s %6s\n','N','start nm','floor nm','gain %','x target','evals');
for i = 1:numel(rows)
    r = rows(i);
    fprintf('  %3d %14.1f %14.1f %9.1f %8.0f %6d\n', r.N, r.wfe0, r.wfe, ...
            r.gain_pct, r.wfe/71, r.nfev);
end
exit(0);
