function OUT = dyson5_optionB(which, opts)
%DYSON5_OPTIONB  Addendum 49 step 2: the paper's Option B (aspheric Dyson lens + CONIC grating) on a Dyson of record.
%   OUT = dyson5_optionB('3k') re-solves the 3k Dyson of record (size:F:240, CaF2 240; the record = rung R3 = the
%   paper's OPTION A, aspheric lens + spherical grating) with the grating's conic constant freed: dyson_ladder rung 7
%   (RB = R3 + grat_Kc), warm from the record, the size trade's own settings (family F: rung 3, CaF2, the full slit,
%   no bound overrides; max_iter P.env_max_iter).  Scores the result in the engine alone, writes the deck
%   dyson5_optionB_<which>.in and dyson5_optionB_<which>.{txt,mat}.  opts.max_iter overrides the budget.
arguments
    which (1,:) char {mustBeMember(which, {'3k', '1k5'})} = '3k'
    opts.max_iter (1,1) double = NaN
    opts.rung (1,1) double = 7          % 7 = RB (Option B); 3 = the CONTROL: R3 re-solved from the record, no grating conic
end
here = fileparts(mfilename('fullpath'));  addpath(fullfile(here, '..', '..', 'design', 'src'));
switch which
    case '3k',  fam = 'F';  r_mm = 240;  glass = 'CaF2';
    case '1k5', fam = 'D';  r_mm = 130;  glass = 'Silica';
end
Z = load(fullfile(here, 'dyson5_size.mat'));  rr = Z.OUT.rows;
k = find(strcmp(string({rr.family}), fam) & abs([rr.r_mm] - r_mm) < 1e-9 & strcmp(string({rr.variant}), 'solve'), 1);
rec = rr(k);  P0 = dyson5_params();
Q = P0;  Q.glass = glass;  Q.block_r_m = r_mm*1e-3;
if strcmp(which, '1k5'), Q.npix = [round(27e-3/Q.pixel_m), Q.npix(2)];  Q.slit_m = 27e-3; end   % family D: the 27 mm slit
mi = P0.env_max_iter;  if ~isnan(opts.max_iter), mi = opts.max_iter; end
sfx = '';  if opts.rung ~= 7, sfx = sprintf('_ctrlR%d', opts.rung); end
deck = fullfile(here, sprintf('dyson5_optionB_%s%s.in', which, sfx));
macos.init(P0.model);
L = dyson_ladder(Q, fullfile(here, sprintf('dyson5_optionB_%s%s', which, sfx)), 'rungs', opts.rung, 'seed', rec.P, 'deck', deck, ...
                 'nx', P0.ladder_nx, 'nlam', P0.ladder_nlam, 'w_dist', P0.ladder_w_dist, 'w_blur', P0.ladder_w_blur, ...
                 'clear_m', P0.ladder_clear_m, 'max_iter', mi, 'quiet', false);
r = L.rung(1);  Re = r.engine;
OUT = struct('which', which, 'record', rec, 'B', r, 'deck', deck);
fid = fopen(fullfile(here, sprintf('dyson5_optionB_%s%s.txt', which, sfx)), 'w');
pr = @(varargin) dual_(fid, varargin{:});
pr('dyson5 Option B %s -- the grating conic freed on the Dyson of record (dyson_ladder rung RB, warm from the record) (%s)\n', which, datestr(now, 'yyyy-mm-dd HH:MM'));
pr('  record (Option A, rung R3): smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px (alone, engine)\n', rec.smile, rec.keystone, rec.CRF, rec.SRF, rec.EE);
pr('  rung %d (%s):           smile %.4f keystone %.4f CRF %.3f SRF %.4f EE %.3f px (alone, engine)\n', opts.rung, tern_(opts.rung == 7, 'Option B', 'control '), Re.smile_max, Re.keystone_max, Re.crf_max, Re.srf_max, Re.ee_min);
gk = 0;  if isfield(r.P, 'grat_Kc'), gk = r.P.grat_Kc; end
pr('  grat_Kc %.4f; block_Kc %.4f, asph %s; Rg_factor %.5f; face_offset %.3f mm; block_dz/dy %.3f / %.3f mm; on bounds: %s\n', ...
   gk, r.P.block_Kc, mat2str(r.P.block_asph, 5), r.P.Rg_factor, r.P.face_offset*1e3, r.P.block_dz*1e3, r.P.block_dy*1e3, strjoin(r.on_bounds, ', '));
fclose(fid);
save(fullfile(here, sprintf('dyson5_optionB_%s%s.mat', which, sfx)), 'OUT');
end
function s = tern_(c, a, b), if c, s = a; else, s = b; end, end
function dual_(fid, varargin), fprintf(varargin{:});  fprintf(fid, varargin{:}); end
