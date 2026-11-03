function out = tg96_tail()
%TG96_TAIL  Re-tune the detector tail at the TG96 scale (Dave 2026-09-04).
%   The v1 fieldlens tail (l2_trade winner) was scaled geometrically to
%   the 96 mm rig; geometry scales, diffraction does not, and the result
%   is the ~9.1 nm static null (invariant to sampling and model size).
%   This optimizer re-opens the tail's four parameters -- FL_F, FL_Kc,
%   D_MASK_FL, DET_TRIM -- minimizing the UNALIGNED NULL RMS.  Runs at
%   reduced resolution (model 512, NGRID 193, flat DM so the grid cap is
%   irrelevant); the winner is verified at full scale by tg96.m, which
%   reads tg96_tail.mat when present.
%   Run:  cd <this dir>;  matlab -batch "tg96_tail"

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

s   = 96/56;
LAM = 6.328e-4;  QWP = 0.25;  THETAS = [0 45 90 135];
MODEL = 512;  NGRID = 193;  N_G = 256;  DX_G = 0.4;
AOI = 7;  D_BS_TO = 700;
macos.init(MODEL);
macos.write_grid_file('tail_flat.txt', zeros(N_G));

seed = [s*25.02100857, -2.11278288, s*6.277463741, s*1.085330067];
% parameterization: FL_F = seed(1)*exp(q1) keeps it positive; others direct
q0 = [0, seed(2), seed(3), seed(4)];
C = struct('s',s, 'AOI',AOI, 'D_BS_TO',D_BS_TO, 'NGRID',NGRID, 'N_G',N_G, ...
           'DX_G',DX_G, 'QWP',QWP, 'THETAS',THETAS, 'LAM',LAM, 'seed',seed);

r0 = null_rms(q0, C);
fprintf('TAIL SEED null: %.4f nm rms (scaled v1 tail)\n', r0);
[qb, rb] = fminsearch(@(q) null_rms(q, C), q0, optimset('MaxFunEvals',120, 'MaxIter',120, ...
                      'TolFun',1e-3, 'TolX',1e-4, 'Display','off'));
pb = [seed(1)*exp(qb(1)), qb(2), qb(3), qb(4)];
fprintf('TAIL WINNER: FL_F %.4f  FL_Kc %.5f  D_MASK_FL %.4f  DET_TRIM %.4f -> null %.4f nm (seed %.4f)\n', ...
        pb(1), pb(2), pb(3), pb(4), rb, r0);
out = struct('FL_F',pb(1), 'FL_Kc',pb(2), 'D_MASK_FL',pb(3), 'DET_TRIM',pb(4), ...
             'null_nm',rb, 'seed_null_nm',r0, 'opt_model',MODEL, 'opt_ngrid',NGRID);
save('tg96_tail.mat', 'out');
fprintf('wrote tg96_tail.mat\n');
end

function r = null_rms(q, C)
        persistent neval
        if isempty(neval), neval = 0; end
        s = C.s;  AOI = C.AOI;  D_BS_TO = C.D_BS_TO;  NGRID = C.NGRID;
        N_G = C.N_G;  DX_G = C.DX_G;  QWP = C.QWP;  THETAS = C.THETAS;
        LAM = C.LAM;  seed = C.seed;
        p = [seed(1)*exp(q(1)), q(2), q(3), q(4)];
        try
            G = macos.design.twyman_green('polarizing',true, 'ngridpts',NGRID, ...
                'BS_AOI',AOI, ...
                'F1',s*500, 'F2',s*250, 'D_LENS',s*60, 'R_BAFFLE',s*12.5, 'D_SB',s*250, ...
                'BS_T',s*1.5, 'D_L1_BS',s*150, 'D_BS_TO',D_BS_TO, 'D_BS_CMP',s*100, ...
                'R_TO_AP',s*30, 'L1_Kr',s*236.866, 'L1_Kc',-0.5829, ...
                'L2_Kr',-s*124.076, 'L2_Kc',-0.5826, ...
                'to_grid_file','tail_flat.txt', 'to_grid_n',N_G, 'to_grid_dx',DX_G, ...
                'qwp_ret',QWP, 'pol_in_deg',45, 'qwp_test_deg',0, 'qwp_ref_deg',45, ...
                'out_qwp_deg',0, 'analyzer_deg',0, ...
                'tail_arch','fieldlens', 'FL_F',p(1), 'FL_Kc',p(2), ...
                'FL_D',s*12, 'D_MASK_FL',p(3), 'DET_TRIM',p(4));
            G.bt.emit('tail_test.in');  G.br.emit('tail_ref.in');
            AT = arm_desc('tail_test.in', G.bt, G.T, 0);
            AR = arm_desc('tail_ref.in',  G.br, G.R, 45);
            Sr = analyzer_basis(AR, QWP);
            S0 = analyzer_basis(AT, QWP);
            I0 = frame(S0, Sr, 0);  msk = I0 > 0.1*max(I0(:));
            if nnz(msk) < 500, r = 1e6; return; end
            pn = fourstep(S0, Sr, THETAS);
            hn = (pn - median(pn(msk))) * LAM/(4*pi) * 1e6;
            r = std(hn(msk));
        catch
            r = 1e6;
        end
        neval = neval + 1;
        fprintf('TAILEVAL %3d: FL_F %.3f Kc %.4f D_MASK %.3f TRIM %.3f -> null %.4f nm\n', ...
                neval, p(1), p(2), p(3), p(4), r);
end

% ==== helpers (flat-DM subset, copied verbatim) ====================
function A = arm_desc(rx, b, ix, base_deg)
    nm = {b.E.name};
    A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
        'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
        'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, ...
        'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
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
               macos.complex_field(A.iDET,'plane',2), ...
               macos.complex_field(A.iDET,'plane',3));
end

function S = analyzer_basis(A, QWP)
    E0  = arm_field(A, QWP, 0);
    E45 = arm_field(A, QWP, 45);
    E90 = arm_field(A, QWP, 90);
    S = struct('A', E0, 'C', E90, 'B', 2*E45 - E0 - E90);
end

function E = synth(S, th)
    c = cosd(th);  s = sind(th);
    E = c^2*S.A + c*s*S.B + s^2*S.C;
end

function I = frame(Sx, Sr, th)
    I = sum(abs(synth(Sx,th) + synth(Sr,th)).^2, 3);
end

function p = fourstep(Sx, Sr, th)
    I1 = frame(Sx,Sr,th(1));  I2 = frame(Sx,Sr,th(2));
    I3 = frame(Sx,Sr,th(3));  I4 = frame(Sx,Sr,th(4));
    p  = atan2(I2-I4, I1-I3);
end
