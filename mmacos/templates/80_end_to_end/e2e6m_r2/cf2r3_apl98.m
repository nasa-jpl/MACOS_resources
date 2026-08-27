function OUT = cf2r3_apl98(over)
%CF2R3_APL98  R3 review directive: the L=0.98 closed-loop confirmation.
%
%   CF3a found the apodized-Lyot leg's static FLAT across the Lyot dial
%   (free throughput) -- but static-only.  This measures the closed
%   loop at L=0.98: Jacobian + fixed-G + relin + lin-ach, the CF2
%   protocol exactly, on the apl config with r_lyot_frac = 0.98.  If
%   the floor holds near the L=0.90 campaign floor (1.081e-7), the
%   +19% relative throughput is measured-free and S4 adopts L=0.98;
%   if not, the dial has a closed-loop knee to find.
%
%   Writes cf2r3_report.txt / cf2r3_run.mat + the tag-separated
%   Jacobian caches (cf2_G_<tag>.mat, stamped + fingerprinted).
%
%   See also CF2_EFC, CF3A_LYOT, cf_efc_lib.

    arguments
        over struct = struct()
    end
    here = fileparts(mfilename('fullpath'));
    run(fullfile(here,'..','..','..','mmacos_setup.m'));
    P = e2e6m_r2_params(over);
    addpath(fullfile(here,'..','..','30_instruments','bench_ctb'));
    lib = cf_efc_lib();

    rx = fullfile(P.outdir, 'r1_seg_dm.in');
    A  = load(fullfile(P.outdir,'r1_dm_run.mat'));
    aug = A.OUT.aug;
    beam_d = 2 * 0.023771;
    C1 = load(fullfile(P.outdir,'cf1_run.mat'));
    FC = struct();
    for k = 1:numel(C1.OUT.F), FC.(C1.OUT.F(k).key) = C1.OUT.F(k); end
    cfg = FC.apl.cfg;
    for i = 1:2:numel(cfg)-1
        if strcmp(cfg{i}, 'r_lyot_frac'), cfg{i+1} = 0.98; end
    end

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF2-R3 -- apodized Lyot at L=0.98, closed loop (CF2 protocol)');
    B = load(fullfile(P.outdir,'cf2_apl_run.mat'));
    L = say_(L, 'the L=0.90 campaign floor: %.3e -> %.3e -> %.3e (thru %.3f)', ...
             B.res.c_static, B.res.c_fixed, B.res.c_relin, B.res.thru);

    ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
    L = say_(L, 'tag %s | thru %.3f (vs %.3f at L=0.90: %+.0f%% relative)', ...
             ch.tag, ch.thru, B.res.thru, 100*(ch.thru/B.res.thru - 1));

    dm = cell(1, numel(aug.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                       'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                       'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                       'coupling', P.dj.coupling);
        dm{k}.clear();
    end
    dz_idx = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
    a0 = cellfun(@(d) zeros(d.nact^2,1), dm, 'UniformOutput', false);

    [G0, ~] = lib.jacobian(ch, dm, a0, dz_idx, P, ...
        fullfile(P.outdir, sprintf('cf2_G_%s.mat', ch.tag)));
    [afix, con_fix, ~] = lib.efc(ch, dm, G0, a0, dz_idx, 15, logspace(-6,-2,5));
    L = say_(L, 'fixed-G: %.3e -> %.3e in %d iters', ...
             con_fix(1), con_fix(end), numel(con_fix)-1);

    [G1, ~] = lib.jacobian(ch, dm, afix, dz_idx, P, ...
        fullfile(P.outdir, sprintf('cf2_G_%s_r1.mat', ch.tag)));
    [arel, con_rel, ~] = lib.efc(ch, dm, G1, afix, dz_idx, 10, logspace(-6,-2,5));
    c_relin = con_rel(end);
    strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), arel);
    la1 = lib.linfloor(G1, P.cf.stroke_bound_nm);
    ach = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), arel(:).', 'UniformOutput', false).'));
    fa = floor_at_(la1, ach);
    L = say_(L, 'relin: %.3e -> %.3e | lin-ach %.3e at %.1f nm | strokes %.1f/%.1f nm', ...
             con_rel(1), c_relin, fa.floor, ach, strokes(1), strokes(2));

    if c_relin < 1.5 * B.res.c_relin
        L = say_(L, 'VERDICT: the floor HOLDS at L=0.98 (%.3e vs %.3e at L=0.90, %.2fx) --', ...
                 c_relin, B.res.c_relin, c_relin/B.res.c_relin);
        L = say_(L, '  the +%.0f%% relative throughput is measured-free; S4 adopts L=0.98.', ...
                 100*(ch.thru/B.res.thru - 1));
    else
        L = say_(L, 'VERDICT: the dial has a closed-loop KNEE (%.3e vs %.3e, %.2fx) --', ...
                 c_relin, B.res.c_relin, c_relin/B.res.c_relin);
        L = say_(L, '  probe one or two intermediate fractions; the operating point is the knee.');
    end

    lib.seta(dm, a0);
    L = say_(L, 'CF2-R3 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf2r3_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('tag',ch.tag, 'thru',ch.thru, 'c_static',con_fix(1), ...
        'c_fixed',con_fix(end), 'c_relin',c_relin, 'la_ach',fa, ...
        'strokes_nm',strokes, 'baseline',B.res, 'text',txt, ...
        'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf2r3_run.mat'),'OUT');
end

function fa = floor_at_(la, stroke_nm)
    ok = la.curve_stroke_nm <= stroke_nm;
    if ~any(ok), rk = 1; else, rk = find(ok, 1, 'last'); end
    fa = struct('floor', la.curve_con(rk), 'rank', rk, ...
                'stroke_nm', la.curve_stroke_nm(rk));
end
function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
