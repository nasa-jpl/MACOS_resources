function OUT = oi_story(over)
%OI_STORY  The whole offset-imager story, one parameterized call.
%
%   OUT = OI_STORY(OVER) runs the complete narrative arc at any
%   parameter set (OVER = overrides on OFFSET_IMAGER_PARAMS, the e2e2
%   single-source-of-truth pattern):
%
%     1  the five-stage ladder (OFFSET_IMAGER: S1 coaxial solve, S2
%        disaster map, S3 re-solve at the offset, S4 tilt/dec under the
%        constraint set, S5 Zernike departures),
%     2  counter-design (a): sphere+Zernike from the start (the sz
%        doctrine), same constraints, same budget,
%     3  counter-design (b): the S5 solve with power (mode 5) and
%        y-tilt (mode 3) released into the basis (the term-set probe),
%     4  a story summary -- the ladder table, the counter verdict
%        lines, and the DECK-ASSET MANIFEST (every figure the run
%        emitted, stage by stage, absolute paths) -- printed and saved
%        in OUT.story / <tag>_STORY.md beside the run artifacts.
%
%   A RE-DO WITH DIFFERENT PARAMETERS IS THIS ONE CALL.  Examples:
%
%     OUT = oi_story();                          % the rodgers3 instance
%     OUT = oi_story(struct('EPD_m',0.200, ...   % the t4-wide instance
%             'Fno',2.5,'box_deg',[10 10],'offset_deg',12, ...
%             'z_m1_m',1.0,'spacings_m',[-0.10 0 1.10], ...
%             'seed_R1_m',15,'clear_m',[0.010 0.005], ...
%             'tag','t4','outdir','<dir>'));
%
%   The rodgers3 CHALLENGE instance (deck-truth parameters read from
%   the .seq at runtime + Mike's constraint set + the PACKET) remains
%   challenges/rodgers3/run_t3.m -- a thin instance of this flow.
%
%   See also OFFSET_IMAGER, OFFSET_IMAGER_PARAMS, OI_ZERN_SEED, OI_SOLVE.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    addpath(here);

    % ================= 1. the five-stage ladder =============================
    OUT = offset_imager(over);
    P = OUT.P;
    tag = fullfile(P.outdir, P.tag);

    % ================= 2. counter (a): sphere+Zernike start =================
    fprintf('\n===== counter (a): sphere+Zernike from the start =====\n');
    Xa = OUT.s3.X;
    Xa.K = [0 0 0];  Xa.asph = zeros(3,3);
    Xa.fpa_refit = [0 0];
    Xa = oi_zern_seed(Xa, P);
    [Xa, ha] = oi_solve(Xa, P, 'S5', 'clear', true);
    [Xa, Ga] = oi_close(Xa, P);  Xa.fpa = oi_apply_fpa(Xa);  Ga.fpa = Xa.fpa;
    [~, mpa] = oi_map_fig(Xa, Ga, P, P.offset_deg, ...
        'counter (a): sphere+Zernike from the start', [tag '_ca_map.png']);
    OUT.counter_a = struct('X',Xa,'map',mpa,'hist',ha);

    % ================= 3. counter (b): released term set ====================
    fprintf('\n===== counter (b): S5 with power + y-tilt released =====\n');
    Xb = OUT.s4.X;
    Xb.fpa_refit = [0 0];
    Pb = P;  Pb.zern_modes = sort([3 5 P.zern_modes]);
    Xb = oi_zern_seed(Xb, Pb);
    [Xb, hb] = oi_solve(Xb, Pb, 'S5', 'clear', true);
    [Xb, Gb] = oi_close(Xb, Pb);  Xb.fpa = oi_apply_fpa(Xb);  Gb.fpa = Xb.fpa;
    [~, mpb] = oi_map_fig(Xb, Gb, Pb, Pb.offset_deg, ...
        'counter (b): power + y-tilt released', [tag '_cb_map.png']);
    OUT.counter_b = struct('X',Xb,'map',mpb,'hist',hb);

    % ================= 4. the story summary + deck-asset manifest ===========
    f = fopen([tag '_STORY.md'], 'w');
    cs = onCleanup(@() fclose(f));
    pr = @(varargin) cellfun(@(fid) fprintf(fid, varargin{:}), {1, f});
    pr('\n# %s -- the story in numbers\n\n', P.name);
    pr(['ladder (dense %dx%d strict-centroid map max, nm; clearance = ' ...
        'min oi_clear pair):\n\n'], P.map_n, P.map_n);
    pr('| stage | map max (nm) | clearance (mm) | gates |\n|---|---|---|---|\n');
    for k = 1:numel(OUT.ladder)
        sid = OUT.ladder(k).stage;
        gt = OUT.(sid).gates;
        pr('| %s | %.1f | %.1f | exit %s / clear %s |\n', sid, ...
           OUT.ladder(k).map_max_nm, gt.clear_min_m*1e3, ...
           pf_(gt.exit_pass), pf_(gt.clear_pass));
    end
    pr('| counter (a) sz-start | %.1f | -- | same constraint rows |\n', mpa.max_nm);
    pr('| counter (b) released | %.1f | -- | same constraint rows |\n', mpb.max_nm);
    pr('\ndeck assets (stage order):\n\n');
    for k = 1:numel(OUT.ladder)
        sid = OUT.ladder(k).stage;
        pr('- %s: `%s`, `%s`, `%s`\n', sid, OUT.(sid).fig_layout, ...
           regexprep(OUT.(sid).fig_layout, '_layout\.png$', '_fields.png'), ...
           OUT.(sid).fig_map);
    end
    pr('- counters: `%s_ca_map.png`, `%s_cb_map.png`\n', tag, tag);
    save([tag '_run.mat'], 'OUT');
    fprintf('\nsaved %s_STORY.md + updated %s_run.mat\n', tag, tag);
end

function s = pf_(p), if p, s = 'PASS'; else, s = 'FAIL'; end, end
