function S = dyson5_sens_run(which)
%DYSON5_SENS_RUN  Addendum 49 step 1: the tolerance ladder of a Dyson of record, alone and through its telescope's join.
%   S = dyson5_sens_run('3k')   -- the 3k Dyson (CaF2 240, size:F:240) and the R9 telescope (tma_longslit deck of record)
%   S = dyson5_sens_run('1k5')  -- the 1.5k Dyson (silica 130, size:D:130) and the 1.5k telescope of record
%   Writes dyson5_sens_<which>.txt / .mat here.  See design/src/spectrometer_sens.
here = fileparts(mfilename('fullpath'));  addpath(fullfile(here, '..', '..', 'design', 'src'));
switch which
    case '3k',  fam = 'F';  r_mm = 240;  tel = fullfile(here, 'tls_e2e_cprime_tel.in');
                tmpl = fullfile(here, 'dyson5_t5e_tA_EP_3k_m30_B1_e2e.in');  dys = 'size:F:240';
                sell = [0.5675888 0.4710914 3.8484723 0.0025264299876 0.010078332803 1200.5559729];   % CaF2 (engine table)
                ov = struct('tel_dyson', dys);
    case '1k5', fam = 'D';  r_mm = 130;  tel = fullfile(here, 'dyson5_tA_GM_1k5_bAs.in');
                tmpl = fullfile(here, 'dyson5_t5e_tA_EP_pz_m30_B1_e2e.in');  dys = 'size:D:130';
                sell = [0.6961663 0.4079426 0.8974794 0.004679148 0.01351206 97.934];                % Silica (Malitson, engine table)
                ov = struct('tel_dyson', dys, 'tel_npix_xt', 1500, 'tel5e_roll_deg', 180);          % the 1.5k record's join
    case '3kB', fam = 'F';  r_mm = 240;  tel = fullfile(here, 'tls_e2e_cprime_tel.in');  dys = 'size:F:240';
                tmpl = '';      % built below: the record's join template with Option B's Dyson blocks after its Slit
                sell = [0.5675888 0.4710914 3.8484723 0.0025264299876 0.010078332803 1200.5559729];   % CaF2 (engine table)
                ov = struct('tel_dyson', dys);
    otherwise, error('dyson5_sens_run: 3k | 1k5 | 3kB');
end
Z = load(fullfile(here, 'dyson5_size.mat'));  rr = Z.OUT.rows;
k = find(strcmp(string({rr.family}), fam) & abs([rr.r_mm] - r_mm) < 1e-9 & strcmp(string({rr.variant}), 'solve'), 1);
Pspec = rr(k).P;
if strcmp(which, '3kB')                       % Option B (dyson5_optionB('3k')): its solved parameter set
    B = load(fullfile(here, 'dyson5_optionB_3k.mat'));  Pspec = B.OUT.B.P;
end
G = spectrometer_geom('dyson', Pspec);  Pd = dyson5_params();  macos.init(Pd.model);
file = fullfile(here, sprintf('dyson5_sens_%s_nominal.in', which));
M = spectrometer_rx(G, file, 'ngridpts', Pd.ngridpts, 'name', ['dyson5_sens_' which], 'apertures', true, 'margin', 5e-3);
pp = spectrometer_sens_defaults(G, 'sellmeier', sell);
if strcmp(which, '3kB')                       % the join template: the record's, through its Slit block, then Option B's Dyson
    t0 = fileread(fullfile(here, 'dyson5_t5e_tA_EP_3k_m30_B1_e2e.in'));  st = regexp(t0, '(?m)^\s*iElt=', 'start');
    nm0 = cellfun(@(i) regexp(t0(i:min(i + 400, end)), 'EltName=\s*(\S+)', 'tokens', 'once'), num2cell(st), 'uni', 0);
    iS = find(cellfun(@(c) strcmp(c{1}, 'Slit'), nm0), 1);  st(end+1) = numel(t0) + 1;
    tb = fileread(file);  sb = regexp(tb, '(?m)^\s*iElt=', 'start');
    tmpl = fullfile(here, 'dyson5_sens_3kB_e2e_template.in');
    fid0 = fopen(tmpl, 'w');  fwrite(fid0, [t0(1:st(iS + 1) - 1) newline tb(sb(1):end)]);  fclose(fid0);
end
e2e = [];  if ~isempty(tel), e2e = struct('template', tmpl, 'tel_deck', tel, 'dyson5', ov); end
fid = fopen(fullfile(here, sprintf('dyson5_sens_%s.txt', which)), 'w');  pr = @(varargin) dual_(fid, varargin{:});
pr('dyson5 sens %s -- addendum 49 step 1: the tolerance ladder of the %s Dyson of record (%s), one perturbation at a time (%s)\n', ...
   which, which, dys, datestr(now, 'yyyy-mm-dd HH:MM'));
pr('UNITS: px of %.0f um; per-unit sensitivities in px per the row''s unit; e2e = through the join with the telescope of record\n', Pd.pixel_m*1e6);
pr('  (dyson5_t5f, CENTROID launch -- the slit-filled convention -- roll 0).  Compensated = at the compensator''s optimum, vs the\n');
pr('  nominal at ITS optimum.  focus = the detector along its normal; clock = the detector about its normal.\n\n');
if strcmp(which, '3kB'), e2e.template = tmpl; end
S = spectrometer_sens(G, M, Pspec, 'perts', pp, 'e2e', e2e);
S.which = which;  S.dyson = dys;  S.deck = file;
b = S.base;  pr('NOMINAL alone: smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px\n', b.smile, b.keystone, b.CRF, b.SRF, b.EE);
for c = fieldnames(S.base_comp)', a = S.base_comp.(c{1}).abs;  pr('  at the %s optimum: smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f\n', c{1}, a.smile, a.keystone, a.CRF, a.SRF, a.EE); end
if isfield(S, 'base_e2e'), b = S.base_e2e;  pr('NOMINAL e2e: smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px\n', b.smile, b.keystone, b.CRF, b.SRF, b.EE); end
pr('\n%-20s %9s %-12s | %9s %9s %8s %8s %8s | %9s %9s | %9s %9s | %8s\n', 'row', 'amount', 'unit', 'dsmile', 'dkeyst', 'dCRF', 'dSRF', 'dEE', ...
   'focus dCRF', 'focus dEE', 'clock dsm', 'clock dky', 'e2e dCRF');
for r = S.rows
    fc = r.comp.focus.resid;  ck = r.comp.clock.resid;  e2 = NaN;  if ~isempty(r.d_e2e), e2 = r.d_e2e.CRF; end
    pr('%-20s %9.3g %-12s | %+9.4f %+9.4f %+8.4f %+8.4f %+8.4f | %+9.4f %+9.4f | %+9.4f %+9.4f | %+8.4f\n', r.name, r.amount, r.unit, ...
       r.d.smile, r.d.keystone, r.d.CRF, r.d.SRF, r.d.EE, fc.CRF, fc.EE, ck.smile, ck.keystone, e2);
end
pr('\nE2E per row (d vs the e2e nominal, px):\n');
for r = S.rows
    if isempty(r.d_e2e), continue, end
    pr('  %-20s dsmile %+8.4f dkey %+8.4f dCRF %+8.4f dSRF %+8.4f dEE %+8.4f\n', r.name, r.d_e2e.smile, r.d_e2e.keystone, r.d_e2e.CRF, r.d_e2e.SRF, r.d_e2e.EE);
end
pr('\nLINEARITY (d at 2x / (2 d at 1x)):\n');
for r = S.rows, if isstruct(r.lin), pr('  %-20s smile %.3f keystone %.3f CRF %.3f SRF %.3f\n', r.name, r.lin.smile, r.lin.keystone, r.lin.CRF, r.lin.SRF); end, end
q = pp(strcmp({pp.name}, 'lens_index'));  [dn, lam] = spectrometer_sens_dn(q);
pr('\nLENS INDEX ROW: dn = %.2e at %.2f um (exact); over 0.38-2.5 um the Sellmeier scaling gives dn %.3e .. %.3e (its residual dispersion %.2e)\n', ...
   q.amount, q.lambda0_m*1e6, min(dn), max(dn), max(dn) - min(dn));
pr('XY CHECK: focus alone CRF %.4f EE %.4f; focus + x/y CRF %.4f EE %.4f (a detector translation leaves the relative metrics unchanged)\n', ...
   S.xy_check.focus.abs.CRF, S.xy_check.focus.abs.EE, S.xy_check.focus_xy.abs.CRF, S.xy_check.focus_xy.abs.EE);
fclose(fid);
save(fullfile(here, sprintf('dyson5_sens_%s.mat', which)), 'S');
end
function dual_(fid, varargin), fprintf(varargin{:});  fprintf(fid, varargin{:}); end
