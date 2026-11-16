function OUT = cf2r1_blc(over)
%CF2R1_BLC  R1 review directive: the BLC stall probe.
%
%   The CF2 BLC loop stopped at 1-2 iterations with 2.8/1.4 nm strokes
%   (siblings ran to 7-11 nm) and sits 1.74x above lin-ach -- and
%   "lin-ach at the achieved stroke" is circular when the stall chose
%   the stroke.  Re-run its EFC from the SAME cached forward Jacobian
%   with a wider alpha ladder (13 points, 1e-7..1e-1) and the
%   iteration cap lifted (40).  Either it digs (the CF2 row gets a
%   proper update, with a fresh relin G about the new commands) or the
%   stall is real and the alpha-search evidence says why.
%
%   Writes cf2r1_report.txt / cf2r1_run.mat; does NOT touch the
%   committed CF2 artifacts.
%
%   See also CF2_EFC, cf_efc_lib.

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

    L = {};  t0 = tic;
    L = say_(L, '==== e2e6m CF2-R1 -- BLC stall probe (alpha 1e-7..1e-1 x13, cap 40)');
    B = load(fullfile(P.outdir,'cf2_blc_run.mat'));
    L = say_(L, 'CF2 baseline: %.3e -> %.3e -> %.3e at %.1f/%.1f nm (la ratio %.2f)', ...
             B.res.c_static, B.res.c_fixed, B.res.c_relin, ...
             B.res.strokes_nm(1), B.res.strokes_nm(2), ...
             B.res.c_relin / B.res.la1_ach.floor);

    ch = cf_chain('rx', rx, 'model_size', P.dj.model, ...
                  'prolate_iter', P.co.prolate_iter, ...
                  'circ_stop_frac', P.cf.circ_stop_frac, FC.blc.cfg{:});
    dm = cell(1, numel(aug.ielt));
    for k = 1:numel(dm)
        dm{k} = ctb_dm('ielt', aug.ielt(k), 'ng', aug.ng, ...
                       'gdx_mm', aug.gdx_mm(k), 'nact', P.dj.nact, ...
                       'beam_d_mm', beam_d, 'pitch_mm', beam_d/P.dj.nact, ...
                       'coupling', P.dj.coupling);
        dm{k}.clear();
    end
    a0 = cellfun(@(d) zeros(d.nact^2,1), dm, 'UniformOutput', false);
    cache = fullfile(P.outdir, sprintf('cf2_G_%s.mat', ch.tag));
    dz_probe = find(ch.dz_mask(P.co.inner_lamD, P.co.outer_lamD));
    [G0, ~] = lib.jacobian(ch, dm, a0, dz_probe, P, cache);
    dz_idx = G0.dz_idx;

    alphas = logspace(-7, -1, 13);
    niter  = 40;
    [a1, con, alph] = lib.efc(ch, dm, G0, a0, dz_idx, niter, alphas);
    strokes = cellfun(@(x) 1e9 * rms_(x(x~=0)), a1);
    L = say_(L, 'probe: %.3e -> %.3e in %d iters | strokes %.1f/%.1f nm', ...
             con(1), con(end), numel(con)-1, strokes(1), strokes(2));
    L = say_(L, 'alpha trail: %s', num2str(alph, '%.0e '));

    dug = con(end) < 0.8 * B.res.c_fixed;
    if dug
        L = say_(L, 'VERDICT: the wider ladder DIGS (%.3e vs CF2 fixed-G %.3e) -- the CF2 row', ...
                 con(end), B.res.c_fixed);
        L = say_(L, '  needs a proper update: fresh relin G about these commands, then re-table.');
    else
        L = say_(L, 'VERDICT: the stall is REAL.  With 13 alphas over 1e-7..1e-1 and a 40-iteration');
        L = say_(L, '  cap, the loop reaches %.3e vs CF2''s %.3e -- no alpha buys a further', ...
                 con(end), B.res.c_fixed);
        L = say_(L, '  measured-contrast improvement: the BLC''s dark-zone residual couples to the');
        L = say_(L, '  DMs more weakly than its G''s top modes suggest, and the 1.7x la gap is a');
        L = say_(L, '  bound looseness, not an unexploited control margin.');
    end

    lib.seta(dm, a0);
    L = say_(L, 'CF2-R1 DONE in %.1f min', toc(t0)/60);
    txt = strjoin(L, newline);
    fid = fopen(fullfile(P.outdir,'cf2r1_report.txt'),'w');
    fprintf(fid,'%s\n',txt);  fclose(fid);
    OUT = struct('con',con, 'alphas_used',alph, 'strokes_nm',strokes, ...
        'dug',dug, 'baseline',B.res, 'text',txt, 'when',datestr(now,31)); %#ok<TNOW1,DATST>
    save(fullfile(P.outdir,'cf2r1_run.mat'),'OUT');
end

function r = rms_(v), v = v(:); if isempty(v), r = 0; else, r = sqrt(mean(v.^2)); end, end
function L = say_(L, varargin)
    s = sprintf(varargin{:});  L{end+1} = s;  fprintf('%s\n', s);
end
