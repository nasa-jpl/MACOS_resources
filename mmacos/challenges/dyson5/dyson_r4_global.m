function Gs = dyson_r4_global(P, tag, opts)
%DYSON_R4_GLOBAL  Multi-start search over the meniscus corrector (R4), from R3.
%   Gs = dyson_r4_global(P, tag) takes the ladder's R3 solution from <tag>_s3.mat
%   and runs lsqnonlin from opts.nstart meniscus seeds spread over the
%   corrector's box -- vertex z in [block face + 20 mm, 0.6 m], thickness
%   2-40 mm, both face curvatures in [-8, 8] /m (either sign: a plate, a
%   meniscus either way round) -- with every other R4 variable free and the
%   same operands as dyson_ladder (10 x smile/keystone, 1 x rms spot,
%   clearance wall; 5 x 5 chain grid, straight-ruled grooves).  The best
%   merit is re-solved at the full grid, emitted with apertures, clearance-
%   checked and ENGINE-scored; every start's merit and solution is recorded
%   in <tag>_s3_r4global.txt so the landscape, not one minimum, is the
%   record.  Seeds: a stratified random design (rng 1 for reproducibility)
%   plus the bounded R4 of record as start 1.
    arguments
        P struct
        tag (1,:) char
        opts.nstart (1,1) double = 12
        opts.max_iter (1,1) double = 40
        opts.quiet (1,1) logical = true
    end
    s3 = load([tag '_s3.mat']);  L = s3.S;
    k3 = find(strncmp({L.rung.name}, 'R3 ', 3), 1, 'last');  k4 = find(strncmp({L.rung.name}, 'R4 ', 3), 1, 'last');
    P3 = L.rung(k3).P;  P4 = L.rung(k4).P;
    V = {'Rg_factor', 0.90, 1.10, 1;  'face_offset', 1e-4, 5e-3, 1e-3;  'block_Kc', -2, 2, 0.5; ...
         'asph4', -200, 200, 10;  'asph6', -2e5, 2e5, 1e4;  'block_dz', -0.05, 0.05, 1e-2;  'block_dy', -0.03, 0.03, 1e-2; ...
         'men_z', P3.block_r + 0.02, 0.60, 0.1;  'men_t', 0.002, 0.040, 0.01;  'men_ca', -8, 8, 1;  'men_cb', -8, 8, 1};
    sc = cell2mat(V(:,4));  lb = cell2mat(V(:,2))./sc;  ub = cell2mat(V(:,3))./sc;
    % a stratified (Latin-hypercube-style) design without the Statistics
    % Toolbox: each column is a random permutation of the strata plus a
    % uniform jitter inside the stratum; rng(1) for reproducibility
    rng(1);  nm = 4;  ns = opts.nstart - 1;  H = zeros(ns, nm);
    for c = 1:nm, H(:, c) = (randperm(ns)' - rand(ns, 1))/ns; end
    starts = cell(1, opts.nstart);  starts{1} = P4;
    for q = 2:opts.nstart
        Pq = P3;  Pq.men_z = V{8,2} + H(q-1,1)*(V{8,3} - V{8,2});  Pq.men_t = V{9,2} + H(q-1,2)*(V{9,3} - V{9,2});
        Pq.men_ca = V{10,2} + H(q-1,3)*(V{10,3} - V{10,2});  Pq.men_cb = V{11,2} + H(q-1,4)*(V{11,3} - V{11,2});
        starts{q} = Pq;
    end
    fid = fopen([tag '_s3_r4global.txt'], 'w');
    pr = @(varargin) dualprint_(fid, varargin{:});
    pr('dyson5 R4 global search (%s): %d starts from R3, lsqnonlin %d iterations each, 5x5 chain grid; merit = sum of squares\n', datestr(now, 'yyyy-mm-dd HH:MM'), opts.nstart, opts.max_iter);
    pr('%5s %10s %10s | %8s %8s %8s %8s | %8s %8s %8s %8s %8s %8s %8s\n', 'start', 'merit0', 'merit', 'men_z mm', 'men_t mm', 'c_a /m', 'c_b /m', 'Rg_f', 'face mm', 'Kc', 'A4', 'A6', 'dz mm', 'dy mm');
    o = optimoptions('lsqnonlin', 'Display', 'off', 'MaxIterations', opts.max_iter, 'FunctionTolerance', 1e-10, 'StepTolerance', 1e-8, 'FiniteDifferenceStepSize', 1e-4);
    res = struct('merit0', {}, 'merit', {}, 'P', {});
    for q = 1:opts.nstart
        Pq = starts{q};
        x0 = cellfun(@(n) get_(Pq, n), V(:,1)) ./ sc;  x0 = min(max(x0, lb), ub);
        f = @(x) resid_(set_(Pq, V, x), P);
        m0 = sum(f(x0).^2);
        try
            [x, rn] = lsqnonlin(f, x0(:), lb(:), ub(:), o);
        catch
            x = x0;  rn = m0;
        end
        Ps = set_(Pq, V, x);  res(q) = struct('merit0', m0, 'merit', rn, 'P', Ps);
        pr('%5d %10.4g %10.4g | %8.1f %8.1f %8.3f %8.3f | %8.4f %8.3f %8.3f %8.3g %8.3g %8.2f %8.2f\n', q, m0, rn, Ps.men_z*1e3, Ps.men_t*1e3, Ps.men_ca, Ps.men_cb, ...
            Ps.Rg_factor, Ps.face_offset*1e3, Ps.block_Kc, Ps.block_asph(1), Ps.block_asph(2), Ps.block_dz*1e3, Ps.block_dy*1e3);
    end
    [~, ib] = min([res.merit]);  Pb = res(ib).P;
    G = spectrometer_geom('dyson', Pb);
    Rc = spectrometer_score_chain(G, Pb, 'nx', P.score_nx, 'nlam', P.score_nlam, 'nring', 6);
    file = sprintf('%s_s3_r4global.in', tag);
    M = spectrometer_rx(G, file, 'ngridpts', P.ngridpts, 'name', [P.tag '_r4global'], 'apertures', true, 'margin', P.ap_margin_m);
    macos.init(P.model);  macos.load_rx(file);
    Re = spectrometer_score(G, M, Pb, 'nx', P.score_nx, 'nlam', P.score_nlam, 'quiet', true);
    Cl = spectrometer_clearance(G, P, 'quiet', true);
    pr('BEST start %d: merit %.4g; engine keystone %.4f smile %.4f CRF %.3f SRF %.3f EE %.3f; clearance min %+.2f mm (%s vs %s); deck %s\n', ...
        ib, res(ib).merit, Re.keystone_max, Re.smile_max, Re.crf_max, Re.srf_max, Re.ee_min, Cl.min_mm, Cl.table.leg{1}, Cl.table.body{1}, file);
    k = find(strncmp({L.rung.name}, 'R4 ', 3), 1, 'last');  Re4 = L.rung(k).engine;
    pr('R4 of record (bounded solve): merit %.4g; engine keystone %.4f smile %.4f CRF %.3f SRF %.3f EE %.3f\n', L.rung(k).merit, Re4.keystone_max, Re4.smile_max, Re4.crf_max, Re4.srf_max, Re4.ee_min);
    fclose(fid);
    Gs.res = res;  Gs.best = ib;  Gs.P = Pb;  Gs.chain = Rc;  Gs.engine = Re;  Gs.clearance = Cl;  Gs.file = file;
    save([tag '_s3_r4global.mat'], 'Gs', 'P');
end

function r = resid_(Pc, P)
    try
        G = spectrometer_geom('dyson', Pc);
    catch
        r = 1e3*ones(2*25*2 + 1, 1);  return
    end
    R = spectrometer_score_chain(G, Pc, 'nx', 5, 'nlam', 5, 'nring', 4);
    smile = R.V - R.V(3, :);  keystone = R.U - R.U(:, 3);
    bad = isnan(R.SU) | isnan(R.SV);  smile(bad) = 10;  keystone(bad) = 10;  SU = R.SU;  SV = R.SV;  SU(bad) = 10;  SV(bad) = 10;
    wall = max(0, P.ladder_clear_m - G.fpa.clear_to_slit)/P.ladder_clear_m*100;
    r = [P.ladder_w_dist*smile(:); P.ladder_w_dist*keystone(:); P.ladder_w_blur*SU(:); P.ladder_w_blur*SV(:); wall];
end

function v = get_(Pc, name)
    switch name
        case 'asph4', v = Pc.block_asph(1);
        case 'asph6', v = Pc.block_asph(2);
        otherwise,    v = Pc.(name);
    end
end

function Pc = set_(Pc, V, x)
    for i = 1:size(V, 1)
        val = x(i)*V{i,4};
        switch V{i,1}
            case 'asph4', Pc.block_asph(1) = val;
            case 'asph6', Pc.block_asph(2) = val;
            otherwise,    Pc.(V{i,1}) = val;
        end
    end
end

function dualprint_(fid, varargin)
    fprintf(1, varargin{:});  fprintf(fid, varargin{:});
end
