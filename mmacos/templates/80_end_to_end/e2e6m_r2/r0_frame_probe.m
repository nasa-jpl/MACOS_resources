function OUT = r0_frame_probe()
%R0_FRAME_PROBE  e2e6m round 2, R0.1: measure the slide-11 discrepancy.
%
%   Round 1's [1] check table (s5_report.txt) disagreed with the S4
%   Jacobian on every Seg19 DOF but piston.  Both paths call the SAME
%   macos.perturb(...,'frame','local') veneer, so this probe measures
%   where they actually diverge, on the round-1 deck, element 19:
%
%     [A] the six direct s5-style pokes -> engine dW maps + rms norms
%     [B] a fresh dw_dx_multi replay restricted to elt 19 (harvest
%         defaults: fex reset, central difference, delta 1e-8)
%     [C] the STORED S4 columns for elt 19, center field
%     [D] attribution scan: each engine dW map correlated against ALL
%         stored center-field columns -- a best match under a different
%         (iElt,dof) label means bookkeeping, not frames
%     [E] the triads: TElt 6x6, psi/vpt/rpt, and the angles of the
%         local axes to radial / azimuthal / normal / parent-axis
%
%   Writes r0_probe_report.txt + r0_probe.mat beside itself.
%   Read-only on ../e2e6m (round 1 is frozen).

    here = fileparts(mfilename('fullpath'));
    r1   = fullfile(here, '..', 'e2e6m');
    run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
    addpath(r1);
    P  = e2e6m_params(struct());
    rx = fullfile(r1, P.sn.rx);
    assert(isfile(rx), 'r0: %s not found', rx);
    S4 = load(fullfile(r1, 's4_sens.mat'), 'ox');
    ox = S4.ox;

    IE   = 19;                       % Seg19, the round-1 fingerprint row
    dofn = {'Rx','Ry','Rz','Tx','Ty','Tz'};
    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R0.1 -- frame probe, elt %d', IE);
    L = say_(L, 'deck %s', rx);
    L = say_(L, 'pokes: %g nrad / %g nm (the s5 check amplitudes)', ...
             P.ts.d_rot*1e9, P.ts.d_trans*1e9);

    % ---- the stored model, center field ---------------------------------
    ic   = find(strcmp(ox.field_names, 'C'), 1);
    assert(~isempty(ic), 'r0: no center field in the stored harvest');
    Wnom = ox.per_field_w_nom_2d{ic};
    mnom = finite_(Wnom);
    A0   = ox.per_field_dwdx{ic};
    idx  = find(mnom);
    assert(nnz(mnom) == size(A0,1), 'r0: stored mask/rows mismatch');

    % ---- [A] direct pokes (the s5 path, verbatim) ------------------------
    L = say_(L, '\n[A] direct macos.perturb(...,''local'') pokes:');
    macos.init(P.sn.model);
    n = macos.load_rx(rx);
    macos.opd_ref('chief');
    macos.trace(n);
    W0 = macos.opd();
    dW  = cell(1,6);  both = cell(1,6);
    n_eng = zeros(1,6);  amp = zeros(1,6);
    for j = 1:6
        a = P.ts.d_rot;  if j > 3, a = P.ts.d_trans; end
        amp(j) = a;
        d = zeros(6,1);  d(j) = a;
        macos.perturb(IE, 'rotation', d(1:3), 'translation', d(4:6), ...
                      'frame','local');
        macos.modify();  macos.trace(n);
        W1 = macos.opd();
        macos.perturb(IE, 'rotation', -d(1:3), 'translation', -d(4:6), ...
                      'frame','local');
        macos.modify();
        b  = mnom & finite_(W1);
        v1 = W1(b);  v1 = v1 - mean(v1);
        v0 = W0(b);  v0 = v0 - mean(v0);
        dW{j} = v1 - v0;  both{j} = b;
        n_eng(j) = rms_(dW{j});
        L = say_(L, '    %s : |engine| %.4g', dofn{j}, n_eng(j));
    end

    % ---- [C] stored columns for elt 19 ----------------------------------
    L = say_(L, '\n[C] stored S4 columns, elt %d, center field:', IE);
    n_sto = nan(1,6);  rel = nan(1,6);  q19 = nan(1,6);
    for j = 1:6
        q = find(ox.iElt == IE & ox.dof_idx == j-1 & ...
                 strcmp(ox.kind(:), 'RigidBody'), 1);
        if isempty(q), continue; end
        q19(j) = q;
        rows = find(ismember(idx, find(both{j})));
        col  = A0(rows, q) * amp(j);
        col  = col - mean(col);
        n_sto(j) = rms_(col);
        rel(j)   = rms_(dW{j} - col) / max(n_eng(j), realmin);
        L = say_(L, '    %s : |model| %.4g   rel.err vs engine %.3g', ...
                 dofn{j}, n_sto(j), rel(j));
    end

    % ---- [D] attribution scan -------------------------------------------
    % Which STORED column does each engine response actually match?
    % Correlation over the common rows, all columns at once; report the
    % top 3 with their labels and the implied scale.
    L = say_(L, '\n[D] attribution: engine dW vs ALL %d stored columns:', ...
             size(A0,2));
    top = cell(1,6);
    for j = 1:6
        rows = find(ismember(idx, find(both{j})));
        M  = A0(rows, :);
        M  = M - mean(M,1);
        w  = dW{j};
        cn = sqrt(sum(M.^2,1));  cn(cn==0) = realmin;
        cc = (w.' * M) ./ (norm(w) * cn);
        [~, ord] = sort(abs(cc), 'descend');
        top{j} = ord(1:3);
        for r = 1:3
            q = ord(r);
            sc = (M(:,q).' * w) / (cn(q)^2) / amp(j);   % engine per unit col
            L = say_(L, ['    %s -> col %3d = elt %2d %s (%s)   ' ...
                     '|corr| %.4f   scale %.4g'], ...
                     dofn{j}, q, ox.iElt(q), dofn{ox.dof_idx(q)+1}, ...
                     ox.kind{q}, abs(cc(q)), sc);
        end
    end

    % ---- [B] fresh single-element harvest replay -------------------------
    L = say_(L, '\n[B] fresh dw_dx_multi replay, elts=%d, harvest defaults:', IE);
    m  = macos.Session(P.sn.model);
    fov = deg2rad(P.tel.fov_arcmin/60);
    fr = macos.dw_dx_multi(m, char(rx), ...
            'field_x_rad', fov, 'field_y_rad', fov, ...
            'reset_xp_method', 'fex', ...
            'elts', IE, 'dofs', (0:5).');
    icf = find(strcmp(fr.field_names, 'C'), 1);
    Af  = fr.per_field_dwdx{icf};
    mf  = finite_(fr.per_field_w_nom_2d{icf});
    idf = find(mf);
    n_fre = nan(1,6);  cc_fs = nan(1,6);
    for j = 1:6
        q = find(fr.iElt == IE & fr.dof_idx == j-1, 1);
        if isempty(q), continue; end
        rowsf = find(ismember(idf, find(both{j} & mf)));
        rows0 = find(ismember(idx, find(both{j} & mf)));
        cf = Af(rowsf, q) * amp(j);   cf = cf - mean(cf);
        n_fre(j) = rms_(cf);
        if ~isnan(q19(j))
            c0 = A0(rows0, q19(j)) * amp(j);  c0 = c0 - mean(c0);
            cc_fs(j) = (cf.' * c0) / max(norm(cf)*norm(c0), realmin);
        end
        L = say_(L, '    %s : |fresh| %.4g   (engine %.4g, stored %.4g)   corr(fresh,stored) %+.4f', ...
                 dofn{j}, n_fre(j), n_eng(j), n_sto(j), cc_fs(j));
    end

    % ---- [E] the triads --------------------------------------------------
    L = say_(L, '\n[E] frames at elt %d:', IE);
    macos.init(P.sn.model);  macos.load_rx(rx);
    cs  = macos.get_elt_csys(IE);
    psi = macos.get_elt_psi(IE);
    vpt = macos.get_elt_vpt(IE);
    rpt = macos.get_elt_rpt(IE);
    L = say_(L, '    psi  %s', mat2str(psi(:).', 6));
    L = say_(L, '    vpt  %s', mat2str(vpt(:).', 6));
    L = say_(L, '    rpt  %s', mat2str(rpt(:).', 6));
    L = say_(L, '    TElt 6x6:');
    T = cs.csys(:,:,1);
    for r = 1:6
        L = say_(L, '      %s', sprintf('% .6f ', T(r,:)));
    end
    % geometry references: parent axis ~ global z of the PM group; the
    % segment's radial / azimuthal unit vectors from its vpt (PM vertex
    % is near the origin of the round-1 telescope frame)
    zax = [0;0;1];
    rad = [vpt(1); vpt(2); 0];  rad = rad / max(norm(rad), realmin);
    azi = cross(zax, rad);
    nrm = psi(:) / norm(psi);
    ang = @(u,v) acosd(min(1, abs((u(:).'*v(:)) / ...
                  max(norm(u)*norm(v), realmin))));
    R3 = T(1:3,1:3);
    L = say_(L, '    angles of TElt(1:3,1:3) ROWS to references (deg, unsigned):');
    for r = 1:3
        u = R3(r,:).';
        L = say_(L, ['      row %d: radial %6.2f  azimuthal %6.2f  ' ...
                 'normal %6.2f  parent-z %6.2f'], ...
                 r, ang(u,rad), ang(u,azi), ang(u,nrm), ang(u,zax));
    end
    L = say_(L, '    angles of TElt(1:3,1:3) COLUMNS likewise:');
    for r = 1:3
        u = R3(:,r);
        L = say_(L, ['      col %d: radial %6.2f  azimuthal %6.2f  ' ...
                 'normal %6.2f  parent-z %6.2f'], ...
                 r, ang(u,rad), ang(u,azi), ang(u,nrm), ang(u,zax));
    end

    L = say_(L, '\nR0.1 probe DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(here,'r0_probe_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('n_eng',n_eng, 'n_sto',n_sto, 'n_fre',n_fre, ...
                 'rel',rel, 'cc_fresh_stored',cc_fs, 'top',{top}, ...
                 'T',T, 'psi',psi, 'vpt',vpt, 'rpt',rpt, ...
                 'dW',{dW}, 'both',{both}, 'q19',q19, 'amp',amp, ...
                 'text',txt);
    save(fullfile(here,'r0_probe.mat'), 'OUT', '-v7.3');
end

function m = finite_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
