function r4_loop_diag()
%R4_LOOP_DIAG  Pure-linear stability diagnosis of the RBCS mechanization.
%   No engine: the drift + measurement + estimator loop simulated on the
%   linear model alone, for three mechanizations.  Prints spectral radii
%   and 41-frame residuals.  The engine run diverged (19 nm residual on
%   3 nm drift); this isolates why and picks the stable law.
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    met = load(fullfile(here,'r3_met.mat'), 'dldx','dedx','dxdl','dxde');
    P = e2e6m_r2_params(struct());
    nb = 19;  nx = 6*nb;
    H  = [met.dldx(:,1:nx); met.dedx(:,1:nx)];
    K  = [met.dxdl, met.dxde];
    Ks = K(1:nx,:);
    sig = [1e-12*ones(size(met.dldx,1),1); 1e-9*ones(size(met.dedx,1),1)];

    % (a) the engine mechanization: u <- u - g*(Ks*m)
    T = Ks*H;
    ea = eig(eye(nx) - 0.5*T);
    fprintf('(a) MMSE slice:  max|eig(I-gKH)| = %.4f\n', max(abs(ea)));

    % (b) BLUE on the segment state: x_hat = (H''WH + eps)^-1 H''W m
    W = diag(1./sig.^2);
    A = H.'*W*H;
    epsr = 1e-6*max(diag(A));
    Kb = (A + epsr*eye(nx)) \ (H.'*W);
    eb = eig(eye(nx) - 0.5*(Kb*H));
    fprintf('(b) BLUE+ridge:  max|eig(I-gKH)| = %.4f\n', max(abs(eb)));

    % (c) BLUE + observable-subspace projection (SVD truncation)
    [U_,S_,V_] = svd(A);  sv = diag(S_);
    keep = sv > 1e-9*sv(1);
    Pk = V_(:,keep)*V_(:,keep).';
    ec = eig(eye(nx) - 0.5*Pk*(Kb*H));
    fprintf('(c) BLUE+proj:   max|eig(I-gKH)| = %.4f  (%d/%d modes kept)\n', ...
            max(abs(ec)), nnz(keep), nx);

    % 41-frame closed-loop sims on the linear model
    rng(P.ts.seed);
    X = drift_(P, nb);
    for lab = 'abc'
        switch lab
            case 'a', Kk = Ks;  Pj = eye(nx);
            case 'b', Kk = Kb;  Pj = eye(nx);
            case 'c', Kk = Kb;  Pj = Pk;
        end
        u = zeros(nx,1);  r_end = 0;
        for k = 1:size(X,2)
            m = H*(X(:,k)+u) + sig.*randn(numel(sig),1);
            u = u - 0.5*Pj*(Kk*m);
            r_end = X(:,k)+u;
        end
        fprintf('(%c) 41-frame linear loop: final |x+u| rms %.3g (drift %.3g)\n', ...
                lab, rms(r_end), rms(X(:,end)));
    end
end

function X = drift_(P, nb)
    nT = P.ts.frames;  t = (0:nT-1)*P.ts.dt;
    X = zeros(6*nb, nT);  w = zeros(6*nb, 1);
    dir = randn(6*nb, 1);  dir = dir/norm(dir);
    for k = 2:nT
        s = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.walk_rot;  if j >= 3, a = P.ts.walk_trans; end
            s(j+1:6:end) = a*randn(nb,1);
        end
        w = w + s;
        g = zeros(6*nb,1);
        for j = 0:5
            a = P.ts.drift_rot;  if j >= 3, a = P.ts.drift_trans; end
            g(j+1:6:end) = a;
        end
        X(:,k) = w + dir .* g .* (t(k)/100);
    end
end
