function OUT = cf3d_stations()
%CF3D_STATIONS  Internal-performance graphics at the CF3d operating point,
%   following the ctb_coro_compare documentation model (INT at key planes,
%   station by station), two columns: DMs FLAT vs the CF3d DUG state.
%
%   Rebuilds the d=1.10 apl chain EXACTLY as cf3d_deepdig does, then
%   replays cf_chain's run_ walk twice with complex-field reads at the
%   stations run_ already stops at (no new stations -- the cf0 lesson:
%   extra read-and-continue stops perturb the field).  Panels:
%     1  pupil amplitude at the Apodizer plane, before masks
%     2  after the circular stop + prolate apodizer
%     3  FPM plane, log10 I (own peak) -- before the occulter
%     4  FPM plane after the occulter
%     5  Lyot plane before the stop -- the rejected gap/edge light
%     6  Lyot plane after the 0.90 stop
%     7  science plane, log10(I / bare peak) = CONTRAST, dark zone
%        annotated with the 3-15 lambda/D mean
%   Caches the station fields to cf3d_stations.mat and renders via
%   CF3D_STATIONS_FIG (both orientations: cf3d_stations.png tall,
%   cf3d_stations_wide.png 2x7 slide strip) + cf3d_dm_state.png (the
%   dug DM surfaces); appends measured station numbers to
%   cf3d_report.txt.  Re-render figures WITHOUT a re-capture:
%   cf3d_stations_fig('wide'|'tall'|'both').
%
%   See also CF3D_STATIONS_FIG, CF3D_DEEPDIG, CF_CHAIN, ctb_coro_compare.

    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(struct());
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));

    rep = fullfile(P.outdir, 'cf3d_report.txt');

    % ---- chain + DMs, verbatim from cf3d_deepdig -----------------------
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    prop_in = fullfile(P.outdir, 'r1_seg_d110_prop.in');
    dmrx    = fullfile(P.outdir, 'r1_seg_d110_dm.in');
    assert(isfile(prop_in) && isfile(dmrx), 'cf3d_stations: run cf3b first');
    Adm = ctb_dm_rx('rx_in', prop_in, 'rx_out', dmrx, ...
                    'dms', P.dm.names, 'ng', P.dm.ng);
    ch = cf_chain('rx', dmrx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, FC.apl.cfg{:});
    dm = cell(1, numel(Adm.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', Adm.ielt(k), 'ng', Adm.ng, ...
            'gdx_mm', Adm.gdx_mm(k), 'nact', P.dj.nact, ...
            'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
            'coupling', P.dj.coupling);
        dm{k}.clear();
    end
    dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));

    CK = load(fullfile(P.outdir, 'cf3d_run.mat'));
    a_dug  = CK.a;
    a_flat = cellfun(@(x) zeros(size(x)), a_dug, 'UniformOutput', false);

    S(1) = capture_(ch, dm, a_flat, 'DMs flat');
    S(2) = capture_(ch, dm, a_dug,  'CF3d dug state');
    pb = ch.peak_bare;
    czf = mean(abs(S(1).fpa(dz_idx)).^2) / pb;
    czd = mean(abs(S(2).fpa(dz_idx)).^2) / pb;

    % station fields cached so figures re-render without a re-capture
    N = ch.N;  c = ch.center_px;  tag = ch.tag;
    save(fullfile(here,'cf3d_stations.mat'), 'S','pb','czf','czd', ...
         'N','c','tag', '-v7.3');
    cf3d_stations_fig('both');

    % ---- the dug DM surfaces -------------------------------------------
    fig2 = figure('Position',[40 40 900 380], 'Color','w', 'Visible','off');
    tl2 = tiledlayout(fig2, 1, numel(a_dug), 'TileSpacing','compact');
    title(tl2, 'CF3d dug DM command state -- the 33 nm rms that buys three decades');
    for k = 1:numel(a_dug)
        ax = nexttile(tl2);
        imagesc(ax, 1e9*reshape(a_dug{k}, P.dj.nact, P.dj.nact));
        axis(ax,'image');  cb = colorbar(ax);  cb.Label.String = 'nm';
        title(ax, sprintf('DM%d (elt %d), rms %.1f nm', k, Adm.ielt(k), ...
              1e9*rms(a_dug{k}(a_dug{k}~=0))));
    end
    exportgraphics(fig2, fullfile(here,'cf3d_dm_state.png'), 'Resolution',130);

    % ---- report --------------------------------------------------------
    logf_(rep, '---- cf3d_stations (%s) ----', datestr(now,31)); %#ok<DATST>
    logf_(rep, 'DZ mean in-walk: flat %.3e | dug %.3e (report round 10: 1.133e-09)', czf, czd);
    for s = 1:2
        logf_(rep, '%-14s | Lyot-plane energy: pre-stop %.4e post-stop %.4e (frac kept %.3f)', ...
              S(s).name, sum(abs(S(s).lyot0(:)).^2), sum(abs(S(s).lyot1(:)).^2), ...
              sum(abs(S(s).lyot1(:)).^2)/sum(abs(S(s).lyot0(:)).^2));
    end
    logf_(rep, 'figures: cf3d_stations.png + cf3d_dm_state.png');
    OUT = struct('S',S, 'cz_flat',czf, 'cz_dug',czd, 'peak_bare',pb);
end

function C = capture_(ch, dm, a, name)
    % cf_chain run_ verbatim, with complex-field reads at existing stops
    for k = 1:numel(dm), dm{k}.apply(a{k}); end
    e = ch.elt;  masks = ch.masks;
    macos.intensity(e.DM1);
    macos.intensity(e.DM2, 'reset_trace', false);
    macos.intensity(e.Apodizer, 'reset_trace', false);
    C.name = name;
    C.pup = macos.complex_field(e.Apodizer, 'reset_trace', false);
    if ~isempty(masks.S), macos.apodize(e.Apodizer, masks.S); end
    if ~isempty(masks.A), macos.apodize(e.Apodizer, masks.A); end
    if ~isempty(masks.S) || ~isempty(masks.A)
        macos.intensity(e.Apodizer, 'reset_trace', false);
    end
    C.apod = macos.complex_field(e.Apodizer, 'reset_trace', false);
    macos.intensity(e.FPM, 'reset_trace', false);
    C.fpm0 = macos.complex_field(e.FPM, 'reset_trace', false);
    if ~isempty(masks.F)
        if isreal(masks.F), macos.apodize(e.FPM, masks.F);
        else,               macos.apodize_complex(e.FPM, masks.F);
        end
        macos.intensity(e.FPM, 'reset_trace', false);
    end
    C.fpm1 = macos.complex_field(e.FPM, 'reset_trace', false);
    macos.intensity(e.Lyot, 'reset_trace', false);
    C.lyot0 = macos.complex_field(e.Lyot, 'reset_trace', false);
    if ~isempty(masks.L)
        macos.apodize(e.Lyot, masks.L);
        macos.intensity(e.Lyot, 'reset_trace', false);
    end
    C.lyot1 = macos.complex_field(e.Lyot, 'reset_trace', false);
    C.fpa = macos.complex_field(ch.elt.FPA, 'reset_trace', false);
end

function logf_(rep, varargin)
    s = sprintf(varargin{:});
    fid = fopen(rep, 'a');  fprintf(fid, '%s\n', s);  fclose(fid);
    fprintf('%s\n', s);
end
