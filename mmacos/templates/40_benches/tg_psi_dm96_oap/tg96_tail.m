function out = tg96_tail(varargin)
%TG96_TAIL  Re-tune the detector tail for the gauge (lens OR oap).  The
%   fieldlens tail (FL_F, FL_Kc, D_MASK_FL, DET_TRIM) was fit to L2's
%   aberrations; the OAP focuser is a DIFFERENT element, so the tail MUST be
%   re-run for optics='oap' (Dave's rule).  Minimizes the UNALIGNED NULL RMS
%   at reduced resolution (model 512, NGRID 193, flat DM) with the SAME rig
%   geometry (scale, BS AOI, DM leg, OAP folds) as the full run.  Writes
%   <tag>_tail.mat, which tg96_run reads at Stage B.
%
%   Usage:  tg96_tail('bench.optics','oap','tag','oap')
%           tg96_tail                        % defaults (lens, tag 'lens')

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

P = parse_params_(varargin{:});
s = P.dm(1).nact/56;  LAM = P.LAM;  QWP = P.QWP;  THETAS = P.THETAS;
% reduced-res tuning grid (geometry is full-scale; only diffraction sampling drops)
MODEL = 512;  NGRID = 193;  N_G = 256;  DX_G = 0.4;

% ---- rig geometry (real-scale Stage-A solve; matches tg96_run) --------
beam_r = P.clear.beam_r;  if isempty(beam_r), beam_r = s*P.bench.R_TO_AP; end
need = beam_r + [P.clear.HW_DM; P.clear.HW_REF; P.clear.HW_CAM] + P.clear.MARGIN;
AOI = P.bench.BS_AOI;  if isempty(AOI), AOI = ceil(max(asind(need/P.clear.LEG_CAP))/2); end
D_BS_TO = P.bench.D_BS_TO;
if isempty(D_BS_TO), D_BS_TO = ceil(max(need(1)/sind(2*AOI), s*250)/50)*50; end
oapargs = {};
if strcmp(P.bench.optics,'oap')
    need_off = beam_r + P.clear.HW_CAM + P.clear.MARGIN;
    a1 = P.oap.OAP1_AOI;  a2 = P.oap.OAP2_AOI;
    if isempty(a1), a1 = solve_fold_(s*P.bench.F1, need_off); end
    if isempty(a2), a2 = solve_fold_(s*P.bench.F2, need_off); end
    oapargs = {'OAP1_AOI',a1,'OAP2_AOI',a2,'OAP1_SIDE',P.oap.OAP1_SIDE,'OAP2_SIDE',P.oap.OAP2_SIDE};
    fprintf('TAIL geometry: optics oap, BS_AOI %d, DM leg %d, OAP folds %d/%d deg\n', AOI, D_BS_TO, a1, a2);
else
    fprintf('TAIL geometry: optics lens, BS_AOI %d, DM leg %d\n', AOI, D_BS_TO);
end

macos.init(MODEL);
macos.write_grid_file('tail_flat.txt', zeros(N_G));
b = P.bench;
seed = [s*b.FL_F, b.FL_Kc, s*b.D_MASK_FL, s*b.DET_TRIM];
q0 = [0, seed(2), seed(3), seed(4)];   % FL_F = seed(1)*exp(q1) keeps positive
C = struct('s',s,'AOI',AOI,'D_BS_TO',D_BS_TO,'NGRID',NGRID,'N_G',N_G,'DX_G',DX_G, ...
           'QWP',QWP,'THETAS',THETAS,'LAM',LAM,'seed',seed,'optics',b.optics, ...
           'oapargs',{oapargs},'bench',b);
r0 = null_rms(q0, C);
fprintf('TAIL SEED null: %.4f nm rms\n', r0);
[qb, rb] = fminsearch(@(q) null_rms(q, C), q0, ...
    optimset('MaxFunEvals',120,'MaxIter',120,'TolFun',1e-3,'TolX',1e-4,'Display','off'));
pb = [seed(1)*exp(qb(1)), qb(2), qb(3), qb(4)];
fprintf('TAIL WINNER (%s): FL_F %.4f FL_Kc %.5f D_MASK_FL %.4f DET_TRIM %.4f -> null %.4f nm (seed %.4f)\n', ...
        b.optics, pb(1), pb(2), pb(3), pb(4), rb, r0);
out = struct('FL_F',pb(1),'FL_Kc',pb(2),'D_MASK_FL',pb(3),'DET_TRIM',pb(4), ...
             'null_nm',rb,'seed_null_nm',r0,'opt_model',MODEL,'opt_ngrid',NGRID,'optics',b.optics);
save([P.tag '_tail.mat'], 'out');
fprintf('wrote %s_tail.mat\n', P.tag);
end

% ---------------------------------------------------------------------
function r = null_rms(q, C)
    persistent neval;  if isempty(neval), neval = 0; end
    s = C.s;  b = C.bench;  p = [C.seed(1)*exp(q(1)), q(2), q(3), q(4)];
    try
        G = macos.design.twyman_green('polarizing',true,'ngridpts',C.NGRID, ...
            'optics',C.optics, C.oapargs{:}, 'BS_AOI',C.AOI, ...
            'F1',s*b.F1,'F2',s*b.F2,'D_LENS',s*b.D_LENS,'R_BAFFLE',s*b.R_BAFFLE,'D_SB',s*b.D_SB, ...
            'BS_T',s*b.BS_T,'D_L1_BS',s*b.D_L1_BS,'D_BS_TO',C.D_BS_TO,'D_BS_CMP',s*b.D_BS_CMP, ...
            'R_TO_AP',s*b.R_TO_AP,'L1_Kr',s*b.L1_Kr,'L1_Kc',b.L1_Kc,'L2_Kr',-s*abs(b.L2_Kr),'L2_Kc',b.L2_Kc, ...
            'to_grid_file','tail_flat.txt','to_grid_n',C.N_G,'to_grid_dx',C.DX_G, ...
            'qwp_ret',C.QWP,'pol_in_deg',b.pol_in_deg,'qwp_test_deg',b.qwp_test_deg, ...
            'qwp_ref_deg',b.qwp_ref_deg,'out_qwp_deg',b.out_qwp_deg,'analyzer_deg',b.analyzer_deg, ...
            'tail_arch','fieldlens','FL_F',p(1),'FL_Kc',p(2),'FL_D',s*b.FL_D,'D_MASK_FL',p(3),'DET_TRIM',p(4));
        G.bt.emit('tail_test.in');  G.br.emit('tail_ref.in');
        AT = arm_desc('tail_test.in', G.bt, G.T, 0);
        AR = arm_desc('tail_ref.in',  G.br, G.R, 45);
        Sr = analyzer_basis(AR, C.QWP);  S0 = analyzer_basis(AT, C.QWP);
        I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
        if nnz(msk) < 500, r = 1e6; return; end
        pn = fourstep(S0, Sr, C.THETAS);
        hn = (pn - median(pn(msk))) * C.LAM/(4*pi) * 1e6;
        r = std(hn(msk));
    catch
        r = 1e6;
    end
    neval = neval + 1;
    fprintf('TAILEVAL %3d: FL_F %.3f Kc %.4f D_MASK %.3f TRIM %.3f -> null %.4f nm\n', ...
            neval, p(1), p(2), p(3), p(4), r);
end

function a = solve_fold_(F, need_off)
    for a = 5:44
        if F*abs(sind(180-2*a)) >= need_off, return; end
    end
    a = 45;
end

function P = parse_params_(varargin)
    if ~isempty(varargin) && isstruct(varargin{1})
        P = varargin{1};  rest = varargin(2:end);
    else
        P = tg96_params();  rest = varargin;
    end
    for k = 1:2:numel(rest)
        parts = strsplit(rest{k}, '.');
        P = setfield(P, parts{:}, rest{k+1});  %#ok<SFLD>
    end
end

% ==== PSI helpers (flat-DM subset, copied verbatim from tg96_tail.m) ==
function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx',rx,'b',b,'iPol',find(strcmp(nm,'PolIn'),1), ...
        'iQ',find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base',base_deg,'qwp_deg',base_deg,'oq_deg',0, ...
        'iRC',ix.iRC,'iOQ',ix.iOutQWP,'iAn',ix.iAnalyzer,'iDET',ix.iDET);
end
function a = lax(psi, deg)
    u1 = macos.design.Bench.perp(psi(:));  u2 = cross(psi(:), u1);
    a = cosd(deg)*u1 + sind(deg)*u2;  a = a(:).';
end
function load_arm(A, QWP, an_deg)
    macos.load_rx(A.rx);  b = A.b;
    macos.polarizer(A.iPol, 'axis', lax(b.E(A.iPol).psi, 45));
    qa = lax(b.E(A.iQ(1)).psi, A.qwp_deg);
    for j = 1:2, macos.waveplate(A.iQ(j), 'axis', qa, 'retardance', QWP); end
    macos.waveplate(A.iOQ, 'axis', lax(b.E(A.iOQ).psi, A.oq_deg), 'retardance', QWP);
    macos.polarizer(A.iAn, 'axis', lax(b.E(A.iAn).psi, an_deg));
    macos.polarization('on', 'Ex',[1/sqrt(2) 0], 'Ey',[1/sqrt(2) 0]);
    macos.vector_diffraction(true);
end
function E = arm_field(A, QWP, an_deg)
    load_arm(A, QWP, an_deg);
    E = cat(3, macos.complex_field(A.iDET,'plane',1), ...
               macos.complex_field(A.iDET,'plane',2), macos.complex_field(A.iDET,'plane',3));
end
function S = analyzer_basis(A, QWP)
    S = struct('A',arm_field(A,QWP,0),'C',arm_field(A,QWP,90),'B',[]);
    S.B = 2*arm_field(A,QWP,45) - S.A - S.C;
end
function E = synth(S, th), c=cosd(th); s=sind(th); E = c^2*S.A + c*s*S.B + s^2*S.C; end
function I = frame(Sx, Sr, th), I = sum(abs(synth(Sx,th)+synth(Sr,th)).^2, 3); end
function p = fourstep(Sx, Sr, th)
    p = atan2(frame(Sx,Sr,th(2))-frame(Sx,Sr,th(4)), frame(Sx,Sr,th(1))-frame(Sx,Sr,th(3)));
end
