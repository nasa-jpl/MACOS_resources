function cophi_figs(P, G, say, pngfile)
%COPHI_FIGS  The COPHI station figure in the gauge deck's format (tg96_run's
%   stations_ifo_ / zwfs_run's stations_fig_): two rows -- the flat DM and the
%   30 nm working surface (seed 7, the DM's full lattice, the same command as the
%   other gauges' station figures) -- by seven stations: the mirror command (nm);
%   the TEST arm's field at the camera; the LO's (placed on the test pixels by
%   camera position); two of the four heterodyne frames (buckets 1 and 2); the
%   recovered surface (the H reading, nm); and the recovered map minus the
%   ENGINE's own OPD change (pm) -- the gauge's error, not a model of it.  The
%   rms of each map is in its xlabel; one colorbar per row on the last panel; the
%   panels are the tool's own output (Dave's deck rule).  Noise-free.
C = cophi_ctx(P, G, @(varargin) []);
cfg = P.dm(1);  nact = cfg.nact;
dmap = @(A) dm_influence_map(P.grid.N_G, P.grid.DX_G, 'nact', nact, 'pitch', cfg.pitch, 'act', A);
rng(P.battery.seed_base);  A30 = P.battery.base_rms * randn(nact);      % over the WHOLE lattice, as the station figures
msk = C.mskT;  [mr, mc] = find(msk);  pw = 4;
pr = max(1, min(mr)-pw):min(size(msk,1), max(mr)+pw);  pc = max(1, min(mc)-pw):min(size(msk,2), max(mc)+pw);
crop = @(X) X(pr, pc);
W0 = C.opdf(zeros(P.grid.N_G));
f = figure('Color', 'w', 'Visible', 'off', 'Position', [0 0 2300 700]);
tl = tiledlayout(f, 2, 7, 'TileSpacing', 'compact', 'Padding', 'compact');
titles = {'mirror command', 'test arm at the camera', 'LO at the camera', 'frame 1 (bucket 1)', 'frame 2 (bucket 2)', 'recovered surface', 'recovered - engine'};
lab = {'flat DM', sprintf('%.0f nm working surface', P.battery.base_rms*1e6)};
for row = 1:2
    A = (row == 2) * A30;  M = dmap(A);
    E = C.fieldf(M);  [Fr, ~] = dmg_cophi_frames(E, C.Elo, struct('N', C.N_));
    h  = C.measf(M);                                         % mm, about the flat null
    ht = -(C.opdf(M) - W0)/2;                                % the engine (sign as the capture stage measured)
    d  = h - ht;  d(~msk) = NaN;  d = d - median(d(msk));
    h(~msk) = NaN;
    It = sum(abs(E).^2, 3);  Il = sum(abs(C.Elo).^2, 3);
    pan = {A*1e6, It, Il, Fr(:,:,1), Fr(:,:,2), h*1e6, d*1e9};
    xl = {sprintf('%.1f nm rms', 1e6*sqrt(mean(A(:).^2))), '', '', '', '', ...
          sprintf('read %.1f nm rms', 1e6*std(h(msk))), sprintf('read - engine %.3g pm rms', 1e9*std(d(msk)))};
    for k = 1:7
        ax = nexttile(tl);
        X = pan{k};  if k > 1, X = crop(X); end
        imagesc(ax, X);  axis(ax, 'image');  set(ax, 'XTick', [], 'YTick', []);
        if row == 1, title(ax, titles{k}, 'FontWeight', 'normal', 'FontSize', 13); end
        if k == 1, ylabel(ax, lab{row}, 'FontSize', 13); end
        xlabel(ax, xl{k}, 'FontSize', 12);
        if k == 7, colorbar(ax); end
    end
    say('  station figure, %s: read %.4g nm rms, read - engine %.4g pm rms over the test pupil\n', lab{row}, 1e6*std(h(msk)), 1e9*std(d(msk)));
end
exportgraphics(f, pngfile, 'Resolution', 110);  close(f);
say('  wrote %s\n', pngfile);
end
