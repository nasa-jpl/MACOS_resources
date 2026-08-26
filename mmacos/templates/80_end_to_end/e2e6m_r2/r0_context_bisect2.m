function OUT = r0_context_bisect2()
%R0_CONTEXT_BISECT2  e2e6m R0.1c: locate the flip INSIDE the dw_dx path.
%
%   Round 1b showed every hand-rolled context gives the PISTON response
%   (rotation effectively about the parent vertex), yet dw_dx_multi's
%   columns carry the TILT-about-center.  This round closes in:
%     i: channel poke, but trace/opd at nElt-1 (dw_dx's wf_elt)
%     j: set_src_fov(nominal) + modify first, then channel poke (n)
%     k: macos.dw_dx single-field, elts=19 -- the [B] result minus the
%        multi supervisor
%     l: dw_dx's body replicated inline (load, channels via
%        rigid_body_channels, dwdx_for_current_source) -- if this gives
%        tilt while i/j give piston, strip until it flips.
%
%   Metric per context: |column| for Ry (tilt ~ 0.142 m/rad, piston
%   ~ 0.926 m/rad -- an order apart, no ambiguity).

    here = fileparts(mfilename('fullpath'));
    r1   = fullfile(here, '..', 'e2e6m');
    run(fullfile(here, '..', '..', '..', 'mmacos_setup.m'));
    addpath(r1);
    P  = e2e6m_params(struct());
    rx = fullfile(r1, P.sn.rx);
    IE = 19;
    L = {};  t0 = tic;
    L = say_(L, '==================== e2e6m R0.1c -- dw_dx path bisect, elt %d Ry', IE);
    L = say_(L, 'tilt-about-center ~ 0.142 m/rad; vertex-piston ~ 0.926 m/rad');

    % ---- i: wf at nElt-1 -------------------------------------------------
    m = macos.Session(P.sn.model);
    n = m.load_rx(rx);  we = n - 1;
    m.trace(we);  W0 = m.opd();
    ch = macos.channels.RigidBodyChannel(m, IE, 1);
    ch.apply(1e-9);  m.trace(we);  W1 = m.opd();  ch.restore();
    L = say_(L, '  i: channel poke, wf at nElt-1        : %.4g m/rad', dn_(W0,W1)/1e-9);

    % ---- j: set_src_fov(nominal)+modify first ---------------------------
    m = macos.Session(P.sn.model);
    n = m.load_rx(rx);
    nom = m.get_src_fov();
    m.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, ...
                  'zSrc', nom.zSrc);
    m.modify();
    m.trace(n);  W0 = m.opd();
    ch = macos.channels.RigidBodyChannel(m, IE, 1);
    ch.apply(1e-9);  m.trace(n);  W1 = m.opd();  ch.restore();
    L = say_(L, '  j: set_src_fov(nom)+modify, poke     : %.4g m/rad', dn_(W0,W1)/1e-9);

    % ---- k: macos.dw_dx single field ------------------------------------
    m = macos.Session(P.sn.model);
    o = macos.dw_dx(m, char(rx), 'elts', IE, 'dofs', (0:5).');
    qRy = find(o.iElt == IE & o.dof_idx == 1, 1);
    col = o.dwdx(:, qRy);  col = col - mean(col);
    L = say_(L, '  k: macos.dw_dx single-field column   : %.4g m/rad', rms_(col));

    % ---- l: dw_dx body inline -------------------------------------------
    m = macos.Session(P.sn.model);
    m.load_rx(char(rx));
    chans = macos.channels.rigid_body_channels(m, char(rx), ...
        'dofs', (0:5).', 'elts', IE, 'fp_mode', 'track', 'ep_elt', -1);
    n_elt = m.num_elt();  wf_elt = n_elt - 1;
    wf_func = @() lwf_(m, wf_elt);
    dwdx = macos.dwdx_for_current_source(chans, wf_func, 1e-8, ...
        'method', 'central');
    qRy2 = 0;
    for c = 1:numel(chans)
        if chans{c}.dof_idx == 1, qRy2 = c; break; end
    end
    cl = dwdx(:, qRy2);  cl = cl - mean(cl);
    L = say_(L, '  l: dw_dx body inline (central 1e-8)  : %.4g m/rad', rms_(cl));

    L = say_(L, '\nR0.1c DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(here,'r0_bisect2_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('text',txt);
    save(fullfile(here,'r0_bisect2.mat'), 'OUT');
end

function W = lwf_(m, wf_elt)
    m.trace(wf_elt);
    W = m.opd();
end
function r = dn_(W0, W1)
    b  = fin_(W0) & fin_(W1);
    v1 = W1(b) - mean(W1(b));  v0 = W0(b) - mean(W0(b));
    r  = rms_(v1 - v0);
end
function m = fin_(W)
    m = isfinite(W) & W ~= 0 & abs(W) < 1e30;
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
