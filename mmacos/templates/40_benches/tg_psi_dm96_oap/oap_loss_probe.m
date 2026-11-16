function out = oap_loss_probe(varargin)
%OAP_LOSS_PROBE  WHERE and WHY does the reflective rig lose rays at a large
%   fold?  oap_fold_solve's map flags every point with OAP1_AOI >= 15 deg as
%   losing rays, and that flag is what disqualifies the 90-deg fold -- the one
%   fold that clears the input polarizer geometrically, because at AOI 45 deg
%   the source leg runs PARALLEL to the polarizer's plane and never crosses
%   it.  So whether the loss is real optics or a modelling artifact decides
%   the design.
%
%   For each fold angle it traces the test arm and reports, per element, the
%   number of rays and the engine's own per-ray status (elt_mod's RayStat_*
%   via macos.get_ray_status): Obscured / Miss / Bracket / MaxIter / Undef,
%   plus the element the first failure was recorded at.  A Miss at the element
%   AFTER the OAP is the signature of a bad conic root; an Obscured count is a
%   real aperture clipping the beam.
%
%   Usage:  out = oap_loss_probe                       % the default ladder
%           out = oap_loss_probe('A1',[5 15 25 35 45],'A2',9)

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir,'..','..','..','mmacos_setup.m'));
end
cd(exdir);
o = struct('A1',[5 10 15 20 25 30 35 40 45], 'A2', 9, 'sides',[1 1], ...
           'MODEL',512, 'NGRID',65, 'tag','loss', 'POL_IN','collimated', ...
           'D_RC_L2',[], 'SRC_AT_FOCUS',false);
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
P = tg96_params();  s = P.dm(1).nact/56;  b = P.bench;
outdir = fullfile(exdir,'runs',o.tag);  if ~exist(outdir,'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir,[o.tag '_loss.txt']),'w');
cleaner = onCleanup(@() fclose(rep));
say = @(varargin) say_(rep, varargin{:});
gf = fullfile(outdir,'loss_flat.txt');
macos.init(o.MODEL);  macos.write_grid_file(gf, zeros(256));

say('=== where the reflective rig loses rays vs the OAP1 fold ===\n');
say('sides %+d/%+d, input polarizer in the %s leg\n', ...
    o.sides(1), o.sides(2), o.POL_IN);
say('model %d, %d rays\n\n', o.MODEL, o.NGRID);
% the ladder: whichever of A1 / A2 is a vector is swept (both, pairwise, if
% both are vectors of the same length)
n1 = numel(o.A1);  n2 = numel(o.A2);  nk = max(n1, n2);
A1v = o.A1;  A2v = o.A2;
if n1 == 1, A1v = repmat(o.A1, 1, nk); end
if n2 == 1, A2v = repmat(o.A2, 1, nk); end
assert(numel(A1v) == nk && numel(A2v) == nk, 'oap_loss_probe: A1/A2 lengths');
say('%5s %5s %8s %8s | %9s %7s %8s %8s %7s | %s\n', 'AOI1','AOI2','nRay','lost', ...
    'obscured','miss','bracket','maxiter','undef','first failing element(s)');

R = struct('A1',{},'A2',{},'n',{},'lost',{},'cnt',{},'elt',{});
for k = 1:nk
    a1 = A1v(k);  o.A2 = A2v(k);
    drc_ = o.D_RC_L2;  if isempty(drc_), drc_ = b.D_RC_L2; end
    try
        G = macos.design.twyman_green('polarizing',b.polarizing,'ngridpts',o.NGRID, ...
            'optics','oap','SRC_AT_FOCUS',o.SRC_AT_FOCUS,'OAP1_AOI',a1,'OAP2_AOI',o.A2, ...
            'OAP1_SIDE',o.sides(1),'OAP2_SIDE',o.sides(2),'POL_IN',o.POL_IN, ...
            'BS_AOI',b.BS_AOI,'F1',s*b.F1,'F2',s*b.F2,'D_LENS',s*b.D_LENS, ...
            'R_BAFFLE',s*b.R_BAFFLE,'D_SB',s*b.D_SB,'BS_T',s*b.BS_T, ...
            'D_L1_BS',s*b.D_L1_BS,'D_BS_TO',700,'D_BS_CMP',s*b.D_BS_CMP, ...
            'R_TO_AP',s*b.R_TO_AP,'L1_Kr',s*b.L1_Kr,'L1_Kc',b.L1_Kc, ...
            'L2_Kr',-s*abs(b.L2_Kr),'L2_Kc',b.L2_Kc, ...
            'to_grid_file',gf,'to_grid_n',256,'to_grid_dx',s*0.28*384/256, ...
            'qwp_ret',b.qwp_ret,'pol_in_deg',b.pol_in_deg, ...
            'qwp_test_deg',b.qwp_test_deg,'qwp_ref_deg',b.qwp_ref_deg, ...
            'out_qwp_deg',b.out_qwp_deg,'analyzer_deg',b.analyzer_deg, ...
            'tail_arch',b.tail_arch,'FL_F',s*b.FL_F,'FL_Kc',b.FL_Kc,'FL_D',s*b.FL_D, ...
            'D_MASK_FL',s*b.D_MASK_FL,'DET_TRIM',s*b.DET_TRIM, ...
            'D_RECOMB',b.D_RECOMB,'D_RC_L2',drc_);
        dk = fullfile(outdir, sprintf('loss_a%02d.in', a1));
        G.bt.emit(dk);  macos.load_rx(dk);
        t = macos.trace();
        st = macos.get_ray_status(t.nRays);
        ri = macos.get_ray_info(t.nRays);
        ok = ri.ok_trace(:) & ri.ok_pass(:);
        cnt = zeros(1,5);
        for c = 1:5, cnt(c) = nnz(st.status(:) == c); end   % 1..5 = Obs/Miss/Brk/Max/Undef
        bad = st.fail_elt(st.status(:) > 0);
        u = unique(bad(:));  lab = '';
        for q = 1:min(4,numel(u))
            nm = '?';
            if u(q) >= 1 && u(q) <= numel(G.bt.E), nm = G.bt.E(u(q)).name; end
            lab = [lab sprintf('%d:%s(%d) ', u(q), nm, nnz(bad==u(q)))]; %#ok<AGROW>
        end
        say('%5d %5d %8d %8d | %9d %7d %8d %8d %7d | %s\n', a1, o.A2, t.nRays, nnz(~ok), ...
            cnt(1), cnt(2), cnt(3), cnt(4), cnt(5), lab);
        R(end+1) = struct('A1',a1,'A2',o.A2,'n',t.nRays,'lost',nnz(~ok),'cnt',cnt,'elt',{lab}); %#ok<AGROW>
    catch e
        say('%5d %5d %8s -- build/trace failed: %s\n', a1, o.A2, '--', e.message);
    end
end
say('\nstatus codes: 1 Obscured, 2 Miss, 3 Bracket, 4 MaxIter, 5 Undef (elt_mod RayStat_*)\n');
out = struct('R', R, 'o', o);
save(fullfile(outdir,[o.tag '_loss.mat']),'out');
say('wrote %s_loss.{txt,mat} in %s\n', o.tag, outdir);
end

function say_(fid, varargin)
fprintf(varargin{:});  fprintf(fid, varargin{:});
end
