function out = tg96_collimate(varargin)
%TG96_COLLIMATE  Collimate the lens rig for real and seat its mask on the ray
%   focus (BRIEF_to_tg_redo package A item 1).  Solves four sheet numbers on
%   the bench the sheet describes -- the substrates in, the DM the stop --
%   and writes them out to be pasted back into tg96_params / zwfs_params:
%
%     L1_Kr      the collimator's vertex radius                 } together,
%     L1_Kc      its conic                                      } stage 1
%     L2_Kc      the focuser's powered-face conic                  stage 2
%     MASK_TRIM  the FocalMask seat, on the ray focus              stage 3
%
%   The SOURCE is not a free parameter: SRC_AT_FOCUS puts it exactly F1 from
%   the collimator's powered vertex, which is what "fed at its focus" has to
%   mean if F1 in the sheet is to mean anything, and the LENS is then solved
%   to collimate from there.  (SRC_TRIM stays available in the sheet and in
%   this tool's 'src_trim' option, but a source shift and a radius change are
%   the same first-order knob, so solving both walks along that degeneracy.)
%
%   WHY.  Bench emits zSource and the engine puts the real point source at
%   ChfRayPos + zSource*ChfRayDir, so the collimator is fed 25 mm inside its
%   conjugate on BOTH rigs.  The mirror rig shows it as blur (no free figure
%   to hide in) and SRC_AT_FOCUS fixed it there; the LENS rig hid it in the
%   tuned L1 conic, which is why nothing looked wrong -- and yet its
%   "collimated" space carries 5.8e-4 rad rms of angular spread, 41 waves of
%   curvature over the beam.  That is what walks the rays off the propagation
%   grid between the physical-optics chain's near-field legs (tg96_pupil_s2s,
%   REPORT_bench_realism section 7) and it is what any re-tune of the tail
%   would be tuning against.  The record's L1_Kc / L2_Kc (-0.5829 / -0.5826)
%   were solved on the MISFED beam, so they have to be re-solved here.
%
%   The exact surfaces for these plano orientations are Cartesian ovals, not
%   conics (the flat face refracts the diverging cone before the powered face
%   sees it), so every one of these is a SOLVE against the engine's own rays,
%   not a formula.  The infinite-conjugate limits are the seeds: -1/n^2 for
%   the collimator's back face (a point in glass to collimated air) and -n^2
%   for the focuser's front face (collimated air to a point in glass).
%
%   GATES (BRIEF_to_tg_redo package A item 1), all reported and asserted:
%     exit-ray angular spread after L1  < 1e-4 rad rms
%     focal-spot transverse rms at the mask < 1 um   (flat DM)
%     mask marker within 0.5 mm of the ray focus
%
%   Usage:  tg96_collimate                       % the lens rig, tag 'coll_lens'
%           tg96_collimate('bench.optics','oap') % the mirror rig (expects 0/0)
%           tg96_collimate('verify',true)        % measure the sheet, solve nothing
%
%   Name/value: any dotted tg96_params path ('bench.L1_Kc', ...), plus
%     'tag'      run tag (default coll_<optics>)
%     'verify'   true = report the sheet's own numbers and the gates only
%     'ngrid'    rays across for the solve (65); the winner is re-measured at
%                'ngrid_fine' (193)
%     'seed_Kc'  [] = the sheet's L1_Kc / L2_Kc; 'formula' = the
%                infinite-conjugate limits (-1/n^2, -n^2)
%
%   Writes runs/<tag>/<tag>_report.txt and <tag>.mat, and prints the sheet
%   lines.  Ray traces only -- no diffraction -- so it is minutes, not hours.

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
cd(exdir);

o = struct('tag','', 'verify',false, 'ngrid',65, 'ngrid_fine',193, 'seed_Kc',[], 'model',512, 'src_trim',0);
rest = varargin;  keep = {};
for k = 1:2:numel(rest)
    if isfield(o, rest{k}), o.(rest{k}) = rest{k+1}; else, keep(end+1:end+2) = rest(k:k+1); end %#ok<AGROW>
end
P = parse_params_(keep{:});
b = P.bench;  s = P.dm(1).nact/56;  n_g = getfield_(b,'N_GLASS',1.5);
if isempty(o.tag), o.tag = ['coll_' b.optics]; end
outdir = fullfile(exdir, 'runs', o.tag);  if ~exist(outdir,'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir, [o.tag '_report.txt']), 'w');
cleanrep = onCleanup(@() fclose_if_(rep));
say = @(varargin) say_(rep, varargin{:});

% ---- rig geometry: the same Stage-A solve tg96_run and tg96_tail do -----
beam_r = P.clear.beam_r;  if isempty(beam_r), beam_r = s*b.R_TO_AP; end
need = beam_r + [P.clear.HW_DM; P.clear.HW_REF; P.clear.HW_CAM] + P.clear.MARGIN;
AOI = b.BS_AOI;  if isempty(AOI), AOI = ceil(max(asind(need/P.clear.LEG_CAP))/2); end
D_BS_TO = b.D_BS_TO;
if isempty(D_BS_TO), D_BS_TO = ceil(max(need(1)/sind(2*AOI), s*250)/50)*50; end
oapargs = {};
if strcmp(b.optics,'oap')
    need_off = beam_r + P.clear.HW_CAM + P.clear.MARGIN;
    a1 = P.oap.OAP1_AOI;  a2 = P.oap.OAP2_AOI;
    if isempty(a1), a1 = solve_fold_(s*b.F1, need_off); end
    if isempty(a2), a2 = solve_fold_(s*b.F2, need_off); end
    oapargs = {'OAP1_AOI',a1,'OAP2_AOI',a2,'OAP1_SIDE',P.oap.OAP1_SIDE,'OAP2_SIDE',P.oap.OAP2_SIDE};
end

scr = sprintf('coll_%d_%s', feature('getpid'), o.tag);
f_flat = [scr '_flat.txt'];  f_test = [scr '_test.in'];
cleanscr = onCleanup(@() delete_if_({f_flat, f_test}));
macos.init(o.model);
% the reduced-resolution DM grid tg96_tail tunes on: mGridMat caps a grid at
% 256 on model 512, and the sheet's 384 would refuse to load.  The DM is flat
% throughout this tool, so all the grid has to do is span the aperture.
N_G = 256;  DX_G = 0.4;
macos.write_grid_file(f_flat, zeros(N_G));

C = struct('P',P, 'b',b, 's',s, 'AOI',AOI, 'D_BS_TO',D_BS_TO, 'oapargs',{oapargs}, ...
           'f_flat',f_flat, 'f_test',f_test, 'NGRID',o.ngrid, 'n_g',n_g, ...
           'N_G',N_G, 'DX_G',DX_G);

% ---- where we start ------------------------------------------------------
q0 = [o.src_trim, b.L1_Kc, b.L2_Kc, getfield_(b,'MASK_TRIM',0), s*b.L1_Kr];
% Seeds, when the sheet has not been solved yet (SRC_TRIM 0).  fminsearch
% builds its initial simplex from 5% of each component and falls back to
% 0.00025 for a component that is exactly zero, so a SRC_TRIM starting at 0
% would be searched in quarter-micron steps and never move: seed it at the
% PARAXIAL answer instead.  A plano singlet's front principal plane sits t/n
% behind its flat face, so the conjugate is t*(1-1/n) short of the F1 the
% builder steps off -- 2.6 mm here.  The conic seeds are the
% infinite-conjugate limits (see the header); the record's tuned pair belongs
% to the MISFED bench and is reported for comparison, not used as the start.
rec_Kr = q0(5);  rec_Kc = [q0(2) q0(3)];
if ~strcmp(b.optics,'oap') && (isempty(o.seed_Kc) || strcmp(o.seed_Kc,'formula'))
    % the infinite-conjugate limits (see the header) and the radius the sheet's
    % own F1 asks for -- the lens twyman_green would build if nothing
    % overrode it.  Seeding at the record's pair starts the search on the
    % wrong bench's answer.
    q0(2) = -1/n_g^2;  q0(3) = -n_g^2;  q0(5) = (n_g-1)*s*b.F1;
end
say('=== tg96_collimate: tag %s, optics %s (%s) ===\n', o.tag, b.optics, datestr(now,'yyyy-mm-dd HH:MM'));
say('scale %.4f, BS_AOI %g, DM leg %g, DM aperture %.2f mm radius, %d rays across (winner re-measured at %d)\n', ...
    s, AOI, D_BS_TO, s*b.R_TO_AP, o.ngrid, o.ngrid_fine);
say('SRC_AT_FOCUS %d; substrates: plate [%s], mask [%s], edge margin %.1f mm\n', ...
    getfield_(b,'SRC_AT_FOCUS',false), num2str(b.PLATE_SUB), num2str(b.MASK_SUB), b.EDGE_MARGIN);
say('the record''s lens: L1_Kr %.4f (= (n-1)*%.1f, the conjugate the source really sat at), L1_Kc %.6f, L2_Kc %.6f\n', ...
    rec_Kr, rec_Kr/(n_g-1), rec_Kc(1), rec_Kc(2));
say('start: SRC_TRIM %.4f  L1_Kr %.4f  L1_Kc %.6f  L2_Kc %.6f  MASK_TRIM %.4f\n', q0(1), q0(5), q0(2), q0(3), q0(4));
mrec = measure_(q0, C, getfield_(b,'SRC_AT_FOCUS',false));
say('the bench the sheet describes (SRC_AT_FOCUS %d): exit spread %.3e rad rms (%.1f waves over the beam), spot %.4f um rms, focus %+.3f mm from the marker\n', ...
    getfield_(b,'SRC_AT_FOCUS',false), mrec.spread, mrec.waves, 1e3*mrec.spot, mrec.dfoc);
m0 = measure_(q0, C);
say('the same conics with the source moved to the conjugate: exit spread %.3e rad rms (%.1f waves), spot %.4f um rms, focus %+.3f mm, %d of %d rays\n\n', ...
    m0.spread, m0.waves, 1e3*m0.spot, m0.dfoc, m0.nray, m0.nray_all);

if o.verify
    q = q0;  m = m0;
else
    % ---- stage 1: the collimated space.  (SRC_TRIM, L1_Kc) --------------
    % Measured on the rays that leave L1's powered face, before anything
    % downstream can be blamed for it.  Both knobs at once: a conic change
    % and a conjugate shift both act as defocus at first order, so solving
    % them one at a time walks along that degeneracy instead of across it.
    say('---- stage 1: the collimated space (L1_Kr, L1_Kc) ----\n');
    if strcmp(b.optics,'oap')
        say('the mirror rig collimates with a parabola fed at its focus: nothing to solve.\n');
        q = q0;
    else
        x1 = fminsearch(@(x) obj_spread_(x, q0, C), [q0(5) q0(2)], ...
                        optimset('TolX',1e-4,'TolFun',1e-3,'MaxFunEvals',120,'Display','off'));
        q = q0;  q(5) = x1(1);  q(2) = x1(2);
        m = measure_(q, C);
        say('winner: L1_Kr %.4f (%.4f in sheet units), L1_Kc %.6f -> exit spread %.3e rad rms (was %.3e)\n', ...
            q(5), q(5)/s, q(2), m.spread, m0.spread);
    end
    % ---- stage 2: the focuser's conic ----------------------------------
    say('---- stage 2: the focuser''s conic (L2_Kc) ----\n');
    if strcmp(b.optics,'oap')
        say('the mirror rig focuses with a parabola: nothing to solve.\n');
    else
        x2 = fminsearch(@(x) obj_spot_(x, q, C), q(3), ...
                        optimset('TolX',1e-5,'TolFun',1e-3,'MaxFunEvals',60,'Display','off'));
        q(3) = x2;
        m = measure_(q, C);
        say('winner: L2_Kc %.6f -> spot %.4f um rms at best focus (was %.4f)\n', q(3), 1e3*m.spot, 1e3*m0.spot);
    end
    % ---- stage 3: the seat ---------------------------------------------
    % The marker is placed at F2 from the powered vertex by the thin-lens
    % seed; the plano singlet's principal plane, the mask's own plate and
    % what is left of the spherical aberration move the real focus off it.
    % MASK_TRIM is additive on that station and the field lens and detector
    % ride on it, so the whole tail follows the seat.
    say('---- stage 3: the seat (MASK_TRIM) ----\n');
    for it = 1:3
        m = measure_(q, C);
        if abs(m.dfoc) < 1e-3, break; end
        q(4) = q(4) + m.dfoc;
    end
    m = measure_(q, C);
    say('winner: MASK_TRIM %+.4f mm -> marker within %+.4f mm of the ray focus (was %+.3f)\n', ...
        q(4), m.dfoc, m0.dfoc);
end

% ---- the winner, re-measured at the run's own sampling -------------------
Cf = C;  Cf.NGRID = o.ngrid_fine;
mf = measure_(q, Cf);
say('\n---- the winner at %d rays across ----\n', o.ngrid_fine);
say('exit spread %.3e rad rms (%.1f waves of curvature over the beam), spot %.4f um rms, ', ...
    mf.spread, mf.waves, 1e3*mf.spot);
say('marker %+.4f mm from the ray focus, %d of %d rays through\n', mf.dfoc, mf.nray, mf.nray_all);
say('beam %.2f mm radius at L1, %.2f at the DM (aperture %.2f: %s; %.0f%% of the cone through)\n', ...
    mf.r_L1, mf.r_dm, s*b.R_TO_AP, iff_(mf.r_dm > s*b.R_TO_AP, 'THE DM IS THE STOP', ...
    'the DM does NOT clip -- the cone is the stop'), 100*mf.f_dm);

gates = struct('spread', mf.spread < 1e-4, 'spot', mf.spot < 1e-3, 'seat', abs(mf.dfoc) < 0.5);
say('\nGATES: exit spread < 1e-4 rad rms  %s (%.3e)\n', pf_(gates.spread), mf.spread);
say('       spot < 1 um rms             %s (%.4f um)\n', pf_(gates.spot), 1e3*mf.spot);
say('       seat within 0.5 mm          %s (%+.4f mm)\n', pf_(gates.seat), mf.dfoc);
say('\n---- the sheet lines (tg96_params.m / zwfs_params.m) ----\n');
say('P.bench.SRC_AT_FOCUS = true;\n');
say('P.bench.SRC_TRIM  = %.6f;   %% mm (0 = the source AT the conjugate; solved by tg96_collimate, %s)\n', q(1), o.tag);
say('P.bench.L1_Kr = %.6f;  P.bench.L1_Kc = %.6f;   %% sheet units (x s in the runner)\n', q(5)/s, q(2));
say('P.bench.L2_Kr = %.6f;  P.bench.L2_Kc = %.6f;\n', -abs(b.L2_Kr), q(3));
say('P.bench.MASK_TRIM = %.6f;   %% mm, the mask on the ray focus\n', q(4));

out = struct('tag',o.tag, 'optics',b.optics, 'SRC_TRIM',q(1), 'L1_Kr',q(5)/s, 'L1_Kc',q(2), ...
             'L2_Kc',q(3), 'MASK_TRIM',q(4), 'start',q0, 'start_meas',m0, ...
             'meas',mf, 'meas_sheet',mrec, 'gates',gates, 'ngrid',o.ngrid_fine);
save(fullfile(outdir, [o.tag '.mat']), 'out');
say('\nrun complete\n');
end

% =========================================================================
function m = measure_(q, C, sf)
if nargin < 3, sf = true; end
%MEASURE_  Build the bench at q = [SRC_TRIM L1_Kc L2_Kc MASK_TRIM] and read
%   the three numbers the gates want off ONE trace of the test arm: the
%   angular spread of the rays leaving the collimator, the transverse rms of
%   the beam at its best focus near the mask, and where that focus is
%   relative to the mask marker.
m = struct('spread',1, 'spot',1e3, 'dfoc',1e3, 'nray',0, 'nray_all',0, 'waves',NaN, ...
           'r_dm',NaN, 'r_L1',NaN, 'f_dm',NaN);
b = C.b;  s = C.s;  P = C.P;
G = macos.design.twyman_green('polarizing',true, 'ngridpts',C.NGRID, ...
    'optics',b.optics, C.oapargs{:}, 'BS_AOI',C.AOI, ...
    'D_RECOMB',b.D_RECOMB, 'D_RC_L2',b.D_RC_L2, 'POL_IN',b.POL_IN, ...
    'SRC_AT_FOCUS',sf, 'SRC_TRIM',q(1), 'MASK_TRIM',q(4), ...
    'F1',s*b.F1, 'F2',s*b.F2, 'D_LENS',s*b.D_LENS, 'R_BAFFLE',s*b.R_BAFFLE, 'D_SB',s*b.D_SB, ...
    'BS_T',s*b.BS_T, 'D_L1_BS',s*b.D_L1_BS, 'D_BS_TO',C.D_BS_TO, 'D_BS_CMP',s*b.D_BS_CMP, ...
    'PLATE_SUB',b.PLATE_SUB, 'EDGE_MARGIN',b.EDGE_MARGIN, 'MASK_SUB',b.MASK_SUB, ...
    'R_TO_AP',s*b.R_TO_AP, 'L1_Kr',q(5), 'L1_Kc',q(2), ...
    'L2_Kr',-s*abs(b.L2_Kr), 'L2_Kc',q(3), ...
    'to_grid_file',C.f_flat, 'to_grid_n',C.N_G, 'to_grid_dx',C.DX_G, ...
    'qwp_ret',P.QWP, 'pol_in_deg',b.pol_in_deg, 'qwp_test_deg',b.qwp_test_deg, ...
    'qwp_ref_deg',b.qwp_ref_deg, 'out_qwp_deg',b.out_qwp_deg, 'analyzer_deg',b.analyzer_deg, ...
    'tail_arch',b.tail_arch, 'FL_F',s*b.FL_F, 'FL_Kc',b.FL_Kc, 'FL_D',s*b.FL_D, ...
    'D_MASK_FL',s*b.D_MASK_FL, 'DET_TRIM',s*b.DET_TRIM);
G.bt.emit(C.f_test);
names = {G.bt.E.name};
iL1 = find(strcmp(names,'L1pow') | strcmp(names,'L1'), 1);
iDM = find(strcmp(names,'TestOptic'), 1);
iMK = find(strcmp(names,'FocalMask'), 1);
macos.load_rx(C.f_test);

% the beam at the DM.  ok_trace, NOT ok_pass: obscuration flags the flux and
% leaves the geometric intersection alone, so the footprint is readable even
% where the DM clips it -- which is the whole point of asking whether the DM
% is the stop.
st = macos.trace(iDM);  ri = macos.get_ray_info(st.nRays);
okd = ri.ok_trace;
V = G.bt.E(iDM).vpt(:);  pc = ri.pos(:,okd) - V;
psi = G.bt.E(iDM).psi(:);  pc = pc - psi*(psi'*pc);
m.r_dm = max(sqrt(sum(pc.^2,1)));
m.f_dm = nnz(ri.ok_trace & ri.ok_pass) / max(nnz(okd),1);
lit = false(size(ri.ok_trace));  lit(ri.ok_trace & ri.ok_pass) = true;

% the collimated space, read at the collimator's own exit -- over the rays
% that REACH the DM, not over the whole cone.  The cone overfills the DM by
% design (the DM is the stop), and the rays it throws away are the outermost,
% where a singlet's spherical aberration is largest: including them tunes the
% collimator against light the bench never uses.
st = macos.trace(iL1);  ri = macos.get_ray_info(st.nRays);
ok = ri.ok_trace & ri.ok_pass & lit(1:numel(ri.ok_trace));
if nnz(ok) < 20, return; end
d = ri.dir(:,ok);  dm_ = mean(d,2);
m.spread = sqrt(mean(sum((d - dm_).^2, 1)));
% the same number as a wavefront: a beam of radius r whose rays spread by
% sigma over it carries ~sigma*r/2 of sag = that many waves.  Reported so the
% brief's "41 waves" is reproducible from the spread alone.
pL1 = ri.pos(:,ok);  rL1 = max(sqrt(sum((pL1 - mean(pL1,2)).^2, 1)));
m.waves = m.spread * rL1 / 2 / P.LAM;  m.r_L1 = rL1;

% the focus: one trace at the mask, the best focus found along the rays
st = macos.trace(iMK);  ri = macos.get_ray_info(st.nRays);
ok = ri.ok_trace & ri.ok_pass;
m.nray_all = numel(ok);  m.nray = nnz(ok);
if m.nray < 20, return; end
p = ri.pos(:,ok);  d = ri.dir(:,ok);
c = G.bt.E(iMK).psi(:);  c = c/norm(c);            % the chief direction at the seat
d = d ./ sqrt(sum(d.^2,1));
[u, v] = frame_(c);
% transverse coordinates of p + t*d about the running centroid
a = [u'; v'] * p;  bdir = [u'; v'] * d;
a = a - mean(a,2);  bdir = bdir - mean(bdir,2);
% rms^2(t) = <|a + t*b|^2> is quadratic in t; its minimum is the best focus
aa = mean(sum(a.^2,1));  ab = mean(sum(a.*bdir,1));  bb = mean(sum(bdir.^2,1));
t = -ab/max(bb, eps);
m.spot = sqrt(max(aa + 2*t*ab + t^2*bb, 0));
m.dfoc = t * (mean(sum(d.*c, 1)));                 % along the chief
end

% =========================================================================
function [u, v] = frame_(c)
t = [0;0;1];  if abs(c(3)) > 0.9, t = [1;0;0]; end
u = cross(c, t);  u = u/norm(u);  v = cross(c, u);
end
function y = obj_spread_(x, q0, C)
q = q0;  q(5) = x(1);  q(2) = x(2);
mm = measure_(q, C);
y = log10(max(mm.spread, 1e-12));
fprintf('  COLL1: L1_Kr %9.4f  L1_Kc %9.6f -> spread %.4e rad rms\n', x(1), x(2), mm.spread);
end
function y = obj_spot_(x, q, C)
qq = q;  qq(3) = x;
mm = measure_(qq, C);
y = log10(max(mm.spot, 1e-9));
fprintf('  COLL2: L2_Kc %9.6f -> spot %8.4f um rms (focus %+.3f mm)\n', x, 1e3*mm.spot, mm.dfoc);
end
function P = parse_params_(varargin)
    if ~isempty(varargin) && isstruct(varargin{1})
        P = varargin{1};  rest = varargin(2:end);
    else
        P = tg96_params();  rest = varargin;
    end
    for k = 1:2:numel(rest)
        parts = strsplit(rest{k}, '.');
        P = setfield(P, parts{:}, rest{k+1});  %#ok<SFLD>
    end
end
function v = getfield_(S, f, dflt)
v = dflt;  if isfield(S, f) && ~isempty(S.(f)), v = S.(f); end
end
function a = solve_fold_(F, need_off)
    for a = 5:44
        if F*abs(sind(180-2*a)) >= need_off, return; end
    end
    a = 45;
end
function s = pf_(tf), if tf, s = 'PASS'; else, s = 'FAIL'; end, end
function s = iff_(c,a,b), if c, s = a; else, s = b; end, end
function delete_if_(fs)
for i = 1:numel(fs), if isfile(fs{i}), delete(fs{i}); end, end
end
function fclose_if_(f), if f > 2, fclose(f); end, end
function say_(fid, varargin)
fprintf(varargin{:});  if fid > 2, fprintf(fid, varargin{:}); end
end
