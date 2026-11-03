function OUT = r2_masks_fig()
%R2_MASKS_FIG  The MASK and the APODIZER as physical objects (items 5+6).
%
%   Two exhibit figures of the things themselves, with physical axes:
%
%     r2_fpm_mask.png   the focal-plane occulter: an opaque disk of
%        radius 2.8 lambda/D at the FPM focus, drawn in TRANSMISSION
%        with the axis in lambda/D and the physical radius (um) stated
%        -- the scales measured from the ENGINE at the FPM plane, not
%        assumed.
%     r2_apodizer.png   the pupil apodizer: left, the clear-pupil
%        prolate transmission profile (the mask as manufactured);
%        right, the same profile over the TRACED segmented pupil (what
%        the light actually sees, gaps and all), axes in mm at the
%        apodizer plane.
%
%   Scales come from the same engine queries the scoring chain uses
%   (s5's coro_setup_ recipe); the prolate is rebuilt with the SAME
%   parameters ctb_aplc scored with, so these figures show the scoring
%   masks, not lookalikes.
%
%   Needs r1_coro's r1_seg_prop.in.  Writes the two PNGs.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(struct());
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    rx = fullfile(P.outdir, 'r1_seg_prop.in');
    assert(isfile(rx), 'r2_masks_fig: run r1_coro first');

    N = P.co.model;
    macos.init(N);
    macos.load_rx(rx);
    e = elt_map_(rx);
    cbm = macos.cbm();
    lam = macos.get_src_wvl() * cbm;

    % ---- focal-plane scales at the FPM (the coro_setup_ recipe) ---------
    macos.intensity(e.FPM);
    Isph  = macos.intensity(e.FPM-1, 'reset_trace', false);
    dxsph = abs(macos.dx_at(e.FPM-1));
    R_fpm = abs(macos.get_elt_z(e.FPM-1)) * cbm;
    Dbeam = 2*beam_radius_(Isph, dxsph);
    dx_f       = lam * R_fpm / (N * dxsph);       % m per pixel at the FPM
    lamD_fpm_m = lam * R_fpm / Dbeam;             % m per lambda/D there
    r_occ_m    = P.co.r_occ_lamD * lamD_fpm_m;

    % ---- pupil scales at the apodizer -----------------------------------
    Iap  = macos.intensity(e.Apodizer);
    dxap = abs(macos.dx_at(e.Apodizer));          % m per pixel at the pupil
    r_apod_px = beam_radius_(Iap, 1);
    Phi  = ctb_apod_prolate(N, r_apod_px, P.co.r_occ_lamD, ...
                            'n_iter', P.co.prolate_iter);
    Mocc = 1 - ctb_mask_disk(N, dx_f, r_occ_m, 8);

    % ---- figure 1: the occulter -----------------------------------------
    span = 8;                                     % +- lambda/D shown
    npx  = round(span * lamD_fpm_m / dx_f);
    c    = N/2 + 1;  ix = c-npx:c+npx;
    ax_l = ((ix - c) * dx_f) / lamD_fpm_m;
    f = figure('Visible','off','Color','w','Position',[60 60 720 640]);
    ax = axes(f);
    imagesc(ax, ax_l, ax_l, Mocc(ix,ix).');  axis(ax,'image'); set(ax,'YDir','normal');
    colormap(ax, gray);  cb = colorbar(ax);  cb.Label.String = 'transmission';
    xlabel(ax,'\lambda/D');  ylabel(ax,'\lambda/D');
    title(ax, sprintf(['the focal-plane occulter (FPM): opaque disk, radius ' ...
        '%.1f \\lambda/D = %.1f \\mum\nat the FPM focus (R_{EP} %.3f m, ' ...
        '%.2f \\mum per \\lambda/D, %g nm)'], P.co.r_occ_lamD, r_occ_m*1e6, ...
        R_fpm, lamD_fpm_m*1e6, P.lambda_m*1e9));
    png1 = fullfile(P.outdir,'r2_fpm_mask.png');
    exportgraphics(f, png1, 'Resolution',150);  close(f);

    % ---- figure 2: the apodizer -----------------------------------------
    rpx  = ceil(1.15 * r_apod_px);
    ix   = c-rpx:c+rpx;
    axmm = ((ix - c) * dxap) * 1e3;               % mm at the pupil
    Ap   = Iap > 0.02*max(Iap(:));                % the traced (gapped) pupil
    f = figure('Visible','off','Color','w','Position',[60 60 1240 600]);
    t = tiledlayout(f, 1, 2, 'Padding','compact');
    ax1 = nexttile(t);
    imagesc(ax1, axmm, axmm, Phi(ix,ix).');  axis(ax1,'image'); set(ax1,'YDir','normal');
    colormap(ax1, gray);  cb = colorbar(ax1);  cb.Label.String = 'amplitude transmission';
    xlabel(ax1,'mm at the apodizer pupil');  ylabel(ax1,'mm');
    title(ax1, sprintf('the apodizer as manufactured:\nclear-pupil prolate, throughput %.3f', ...
          sum(Phi(:).^2 .* Ap(:)) / max(sum(Ap(:)),1)));
    ax2 = nexttile(t);
    imagesc(ax2, axmm, axmm, (Phi(ix,ix) .* Ap(ix,ix)).');  axis(ax2,'image'); set(ax2,'YDir','normal');
    colormap(ax2, gray);  cb = colorbar(ax2);  cb.Label.String = 'amplitude transmission';
    xlabel(ax2,'mm at the apodizer pupil');  ylabel(ax2,'mm');
    title(ax2, sprintf('what the light sees:\nprolate over the traced %d-segment pupil', 19));
    png2 = fullfile(P.outdir,'r2_apodizer.png');
    exportgraphics(f, png2, 'Resolution',150);  close(f);

    fid = fopen(fullfile(P.outdir,'r2_masks_report.txt'),'w');
    fprintf(fid, '==================== e2e6m R2 -- the masks as objects\n');
    fprintf(fid, 'occulter radius %.2f lambda/D = %.2f um at the FPM focus\n', ...
            P.co.r_occ_lamD, r_occ_m*1e6);
    fprintf(fid, 'focal scale %.3f um per lambda/D (R_EP %.4f m, %g nm)\n', ...
            lamD_fpm_m*1e6, R_fpm, P.lambda_m*1e9);
    fprintf(fid, 'apodizer radius %.1f px, pupil dx %.4f mm per px\n', ...
            r_apod_px, dxap*1e3);
    fprintf(fid, 'apodizer throughput over the traced aperture %.3f\n', ...
            sum(Phi(:).^2 .* Ap(:)) / max(sum(Ap(:)),1));
    fclose(fid);
    fprintf('r2_masks_fig: occulter %.1f um (%.1f lam/D), apodizer r %.1f px -> %s, %s\n', ...
            r_occ_m*1e6, P.co.r_occ_lamD, r_apod_px, png1, png2);
    OUT = struct('png_fpm',png1, 'png_apod',png2, 'r_occ_m',r_occ_m, ...
                 'lamD_fpm_m',lamD_fpm_m, 'dx_f',dx_f, 'dx_apod',dxap, ...
                 'r_apod_px',r_apod_px);
    save(fullfile(P.outdir,'r2_masks_run.mat'),'OUT');
end

% =========================================================================
function e = elt_map_(rx)
    nm = regexp(fileread(rx), '^\s*EltName=\s*(\S+)', 'tokens','lineanchors');
    nm = cellfun(@(c) c{1}, nm, 'UniformOutput', false);
    at = @(s) find(strcmp(nm, s), 1);
    e = struct('Apodizer',at('Apodizer'), 'FPM',at('FPM'), 'Lyot',at('Lyot'));
    f = fieldnames(e);
    for k = 1:numel(f)
        assert(~isempty(e.(f{k})), 'r2_masks_fig: %s not found in %s', f{k}, rx);
    end
end

function r = beam_radius_(I, dx)
    m = I > 0.02*max(I(:));
    [rr,cc] = find(m);
    if isempty(rr), r = 0;  return; end
    r = 0.5 * max(max(rr)-min(rr), max(cc)-min(cc)) * dx;
end
