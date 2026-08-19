function out = ctb_proper_compare(opts)
%CTB_PROPER_COMPARE  Per-leg PROPER cross-compare on the CTB deck (finding 3).
%   The compact-vs-full model agreement is NOT validation -- the arbiter is
%   a cross-compare against MATLAB PROPER on the CTB prescription's own
%   sampling (recipe validation-ladder item 1).  This runs the arbiter for
%   the NOVEL kernel the whole coronagraph chain rests on: the through-focus
%   FPM leg (feed sphere FPM-1 -> FPM focus), which macos executes as the
%   NF1 FarField sphere->plane and PROPER executes as prop_lens(R) +
%   prop_propagate(R) at MATCHED sampling (beam_ratio = 1, grid_extent =
%   N*dx_sphere -> PROPER pitch == macos pitch bit-for-bit).
%
%   Run this BEFORE interpreting the compact-vs-full suppression gap: it
%   tells you whether the CTB's specific geometry reproduces the diffraction
%   focus PROPER computes, independent of any model-vs-model comparison.
%
%   Reuses the PROPER-validated recipe from mmacos/tests/proper_compare
%   (proper_run_sphere_to_plane) -- same kernel, CTB deck + sampling.
%
%   Requires MATLAB PROPER on the path (~/dev/proper_matlab).  If PROPER is
%   absent the macos leg still runs and the PROPER column is skipped.
%
%   out = CTB_PROPER_COMPARE() uses the compact deck FPM leg.
%   Name-value: 'rx','FPM','model_size','outdir','visible'.
%
%   See also: proper_run_sphere_to_plane, ctb_coro_compare.
    arguments
        opts.rx         (1,:) char   = ''
        opts.FPM        (1,1) double = 17          % compact FPM station
        opts.model_size (1,1) double = 512
        opts.outdir     (1,:) char   = ''
        opts.visible    (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    if isempty(opts.rx),     opts.rx     = fullfile(here,'ctb_dcr.in'); end
    if isempty(opts.outdir), opts.outdir = here; end
    addpath(fullfile(here,'..','..','..','src'));
    assert(~isempty(getenv('MACOS_HOME')),'MACOS_HOME must be set.');

    N   = opts.model_size;
    FPM = opts.FPM;

    % --- macos leg: sphere (FPM-1) -> focus (FPM) -----------------------
    macos.init(N);
    macos.load_rx(opts.rx);
    cbm      = macos.cbm();
    lambda_m = macos.get_src_wvl() * cbm;
    macos.intensity(FPM);                                    % run the chain
    % raw dispatch (2-arg): the installed MEX predates the veneer's 'plane'
    % arg, so macos.complex_field (which passes 3 args) errors.  reset_trace=0.
    cf_sphere = mmacos('complex_field', double(FPM-1), 0);
    I_sphere  = macos.intensity(FPM-1, 'reset_trace', false);
    I_focus_m = macos.intensity(FPM,   'reset_trace', false);
    dx_sph_m  = abs(macos.dx_at(FPM-1));
    R_m       = abs(macos.get_elt_z(FPM-1)) * cbm;           % NF1 sphere zElt
    dx_f_m    = lambda_m * R_m / (N * dx_sph_m);             % deterministic

    fprintf('[proper] CTB FPM leg: sphere elt %d -> focus elt %d\n', FPM-1, FPM);
    fprintf('[proper]   lambda=%.3e m  R=%.4e m  dx_sph=%.4e m  dx_f=%.4e m\n', ...
        lambda_m, R_m, dx_sph_m, dx_f_m);

    % --- PROPER leg (if available): matched sampling --------------------
    have_proper = exist('prop_begin','file')==2 && exist('prop_lens','file')==2;
    I_focus_p = []; dx_f_p = NaN;
    if have_proper
        grid_extent = N * dx_sph_m;                          % match macos pitch
        bm = prop_begin(grid_extent, lambda_m, N, 1.0);
        cf = complex(cf_sphere);
        bm = prop_multiply(bm, abs(cf));
        opd = -angle(cf) * lambda_m / (2*pi);                % macos sign flip
        bm = prop_add_phase(bm, opd);
        bm = prop_define_entrance(bm);
        bm = prop_lens(bm, R_m);
        bm = prop_propagate(bm, R_m);
        [I_focus_p, dx_f_p] = prop_end(bm);
        fprintf('[proper]   PROPER dx_focal=%.4e m (macos dx_f=%.4e, ratio=%.4f)\n', ...
            dx_f_p, dx_f_m, dx_f_p/dx_f_m);
    else
        fprintf('[proper]   PROPER not on path -- macos leg only, comparison skipped.\n');
    end

    % --- metrics --------------------------------------------------------
    m = struct();
    if have_proper
        [m.corr, m.dcx, m.dcy, Am, Ap] = compare_focal_(I_focus_m, I_focus_p);
        fprintf('[proper]   peak-norm corr=%.6f  centroid dcx=%.3f dcy=%.3f px\n', ...
            m.corr, m.dcx, m.dcy);
    end

    % --- figure ---------------------------------------------------------
    vis = 'off'; if opts.visible, vis='on'; end
    fig = figure('Visible',vis,'Color','w','Position',[80 80 1200 420]);
    tl = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    title(tl, sprintf('CTB FPM through-focus leg -- macos vs PROPER (%s)', ...
        'peak-norm log I'), 'FontWeight','bold','Interpreter','none');
    w = 80;                                                  % crop half-width
    nexttile; show_focal_(I_focus_m, w); title('macos (NF1/NF2)');
    if have_proper
        nexttile; show_focal_(I_focus_p, w); title('PROPER (lens+prop)');
        nexttile;
        D = log10(max(norm_(I_focus_m),1e-12)) - log10(max(norm_(I_focus_p),1e-12));
        imagesc(crop_(D,w)); axis image off; colormap(gca,parula);
        clim([-2 2]); cb=colorbar; cb.Label.String='\Delta log_{10} I';
        title(sprintf('difference (corr=%.4f)', m.corr));
    else
        nexttile; text(0.1,0.5,'PROPER not available','Units','normalized');
        axis off;
        nexttile; axis off;
    end
    figpath = fullfile(opts.outdir,'ctb_proper_compare_fpm.png');
    exportgraphics(fig, figpath, 'Resolution',150);
    if ~opts.visible, close(fig); end
    fprintf('[proper] wrote %s\n', figpath);

    out = struct('rx',opts.rx,'FPM',FPM,'lambda_m',lambda_m,'R_m',R_m, ...
        'dx_sph_m',dx_sph_m,'dx_f_macos_m',dx_f_m,'dx_f_proper_m',dx_f_p, ...
        'I_focus_macos',I_focus_m,'I_focus_proper',I_focus_p, ...
        'have_proper',have_proper,'metrics',m,'figure',figpath);
end

% ---------------------------------------------------------------------
function [c, dcx, dcy, Am, Ap] = compare_focal_(Im, Ip)
    Am = norm_(Im); Ap = norm_(Ip);
    a = Am(:); b = Ap(:);
    c = (a-mean(a))'*(b-mean(b)) / (norm(a-mean(a))*norm(b-mean(b))+eps);
    [cxm,cym] = centroid_(Am); [cxp,cyp] = centroid_(Ap);
    dcx = cxm-cxp; dcy = cym-cyp;
end

function A = norm_(I), I=double(I); A = I / max(I(:)+eps); end

function [cx,cy] = centroid_(A)
    [xx,yy] = meshgrid(1:size(A,2),1:size(A,1));
    s = sum(A(:))+eps; cx = sum(xx(:).*A(:))/s; cy = sum(yy(:).*A(:))/s;
end

function show_focal_(I, w)
    L = log10(max(norm_(I),1e-12));
    imagesc(crop_(L,w)); axis image off; colormap(gca,parula); clim([-8 0]);
    cb=colorbar; cb.Label.String='log_{10} norm I';
end

function o = crop_(img, w)
    n = size(img,1); if 2*w>=n, o=img; return; end
    c = floor(n/2)+1; lo=max(c-w,1); hi=min(c+w,n); o=img(lo:hi,lo:hi);
end
