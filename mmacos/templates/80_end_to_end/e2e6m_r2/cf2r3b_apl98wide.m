function OUT = cf2r3b_apl98wide(over)
%CF2R3B_APL98WIDE  R3 follow-on: the L=0.98 run is a STALL, not a floor.
%
%   cf2r3_apl98's headline ("holds at 1.48x") is mechanically inside
%   the brief's 1.5x criterion but the run has the BLC stall signature
%   AND the opposite attribution: fixed-G stopped after ONE iteration,
%   relin gained nothing, strokes 3.3 nm -- while the relin G's
%   linear-achievable at that stroke is 1.79e-9, EIGHTY-NINE TIMES
%   below the achieved 1.595e-7.  That is the definition of
%   CONTROL-limited: the open Lyot passes more DM-visible light (real
%   added authority) and the default alpha ladder cannot take it.
%
%   The R1 treatment, then: wide ladder (13 alphas, 1e-7..1e-1), cap
%   40, from the CACHED forward G.  If it digs materially, a FRESH
%   relin G about the new commands (the cached _r1 was measured about
%   the stalled 3 nm state and its a0 assert would rightly refuse) and
%   a relin pass.  The S4 operating point rides on this.
%
%   See also CF2R3_APL98, CF2R1_BLC, cf_efc_lib.

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
    R3 = load(fullfile(P.outdir,'cf2r3_run.mat'));

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF2-R3b -- apl L=0.98 wide-ladder probe (the stall vs the 89x la gap)');
    L = say_(L, 'R3 stalled state: %.3e at %.1f/%.1f nm; its relin G'' la at that stroke %.3e', ...
             R3.OUT.c_relin, R3.OUT.strokes_nm(1), R3.OUT.strokes_nm(2), ...
             R3.OUT.la_ach.floor);

    ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, cfg{:});
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

    alphas = logspace(-7, -1, 13);
    [afix, cf, alph] = lib.efc(ch, dm, G0, a0, dz_idx, 40, alphas);
    L = say_(L, 'wide fixed-G: %.3e -> %.3e in %d iters (alphas %s)', ...
             cf(1), cf(end), numel(cf)-1, num2str(alph(1:min(8,end)), '%.0e '));

    dug = cf(end) < 0.8 * R3.OUT.c_relin;
    if dug
        r1c = fullfile(P.outdir, sprintf('cf2_G_%s_r1w.mat', ch.tag));
        [G1, ~] = lib.jacobian(ch, dm, afix, dz_idx, P, r1c);
        [arel, cr, ~] = lib.efc(ch, dm, G1, afix, dz_idx, 15, alphas);
        c_final = cr(end);
        strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), arel);
        la1 = lib.linfloor(G1, P.cf.stroke_bound_nm);
        ach = 1e9 * rms_(cell2mat(cellfun(@(x) x(x~=0), arel(:).', ...
                                          'UniformOutput', false).'));
        fa = floor_at_(la1, ach);
        L = say_(L, 'wide relin: %.3e -> %.3e | lin-ach %.3e @ %.1f nm | strokes %.1f/%.1f nm', ...
                 cr(1), c_final, fa.floor, ach, strokes(1), strokes(2));
    else
        c_final = cf(end);  strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), afix);
        fa = R3.OUT.la_ach;  arel = afix;                        %#ok<NASGU>
        L = say_(L, 'the wide ladder does NOT dig either: the L=0.98 stall is real.');
    end

    B90 = load(fullfile(P.outdir,'cf2_apl_run.mat'));
    if c_final < 1.5 * B90.res.c_relin
        L = say_(L, 'VERDICT: L=0.98 floor %.3e vs L=0.90''s %.3e (%.2fx) -- HOLDS; +19%% relative', ...
                 c_final, B90.res.c_relin, c_final/B90.res.c_relin);
        L = say_(L, '  throughput is measured-free (with the wide-ladder loop); S4 adopts L=0.98.');
    else
        L = say_(L, 'VERDICT: even wide, L=0.98 lands %.3e (%.2fx above L=0.90) -- the dial has a', ...
                 c_final, c_final/B90.res.c_relin);
        L = say_(L, '  closed-loop KNEE between 0.90 and 0.98; probe 0.94 next, operating point = knee.');
    end

    lib.seta(dm, a0);
    L = say_(L, 'CF2-R3b DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf2r3b_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('c_final',c_final, 'dug',dug, 'strokes_nm',strokes, ...
        'la_ach',fa, 'con_fixed',cf, 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf2r3b_run.mat'),'OUT');
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
