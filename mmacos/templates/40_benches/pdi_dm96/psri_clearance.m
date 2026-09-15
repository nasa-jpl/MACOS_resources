function out = psri_clearance(varargin)
%PSRI_CLEARANCE  Does every part of the P/SRI bench clear every beam it is not
%   in?  Item 7 of the bench-realism round (BRIEF_to_psri_clearance.md).
%
%   Dave ruled the 7-deg TG96 node unbuildable -- at that splitter angle eight
%   of nine parts around the splitter sat inside a beam -- and the bench of
%   record is now 22.5 deg.  The P/SRI rig was never checked the same way.  Its
%   Mach-Zehnder pickoff and recombiner are at 45 deg, so the node may well be
%   clean, but "may well be" is not a number; and the REFERENCE arm's node (the
%   two relay lenses, the pinhole seat and the bottom fold) is the part no
%   clearance check has ever covered.
%
%   macos.design.psri_bench returns the same {bt, br, P} shape twyman_green
%   does, so dmg_bench_clearance measures it directly through its 'G' option --
%   no fork of the tool, and no bending the TG96 builder into a shape it is not
%   (the brief's three routes; this is the first one).
%
%   Usage:  out = psri_clearance                      % 22.5 deg, the ruled node
%           out = psri_clearance('BS_AOI',[7 22.5])   % the record's angle too
%
%   Options: 'BS_AOI' (deg, a vector runs each), 'MODEL' (512), 'NGRID' (65),
%   'MOUNT' (8), 'MARGIN' (25), 'BODY' (part stem -> physical body radius; the
%   default is the TG96 rule's own half-widths plus the P/SRI's fold mirrors),
%   'tag' (run tag; writes runs/<tag>/).

exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('macos.init'))
    run(fullfile(exdir, '..', '..', '..', 'mmacos_setup.m'));
end
addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
cd(exdir);
o = struct('BS_AOI', 22.5, 'MODEL', 512, 'NGRID', 65, 'MOUNT', 8, 'MARGIN', 25, ...
           'BODY', [], 'tag', 'psriclear');
for i = 1:2:numel(varargin), o.(varargin{i}) = varargin{i+1}; end
if isempty(o.BODY)
    % the source head and the camera package are not their clear apertures, and
    % the source is not an optic at all (its builder element is an Obscuring
    % baffle), so without this it is not in the table.  50 / 90 / 60 mm are the
    % TG96 Stage-A rule's own HW_CAM / HW_DM / HW_REF half-widths.
    o.BODY = struct('Baffle',50, 'Detector',50, 'TestOptic',90, 'M1',75, 'M3',75);
end
P0 = pdi_params();
outdir = fullfile(exdir, 'runs', o.tag);
if ~exist(outdir,'dir'), mkdir(outdir); end
rep = fopen(fullfile(outdir, [o.tag '_clear.txt']), 'w');
cleaner = onCleanup(@() fclose(rep));
say = @(varargin) say_(rep, varargin{:});

macos.init(o.MODEL);
gf = fullfile(exdir, 'zwfs_flat.txt');
if ~isfile(gf), macos.write_grid_file(gf, zeros(P0.grid.N_G)); end

say('=== P/SRI bench clearance: does every part clear every beam it is not in? ===\n');
say('macos.design.psri_bench through dmg_bench_clearance''s ''G'' option.\n');
say('model %d, %d rays; mount +%g mm; spec >= %g mm\n', o.MODEL, o.NGRID, o.MOUNT, o.MARGIN);
bn = fieldnames(o.BODY);
say('physical bodies (radius before the mount, mm): ');
for i = 1:numel(bn), say('%s %g  ', bn{i}, o.BODY.(bn{i})); end
say('\nthe solved reference optics of record (pdi_params): Lr1 = Lr2 conic %.4f, REF_TRIM %+.4f mm\n\n', ...
    P0.pdi.psri.LR1_Kc, P0.pdi.psri.REF_TRIM);

R = struct('aoi',{}, 'rows',{}, 'worst',{}, 'bind',{});
for q = 1:numel(o.BS_AOI)
    aoi = o.BS_AOI(q);
    say('--- front-end splitter at %g deg (the Mach-Zehnder pickoff and recombiner are at 45) ---\n', aoi);
    G = macos.design.psri_bench('BS_AOI', aoi, 'ngridpts', o.NGRID, ...
        'to_grid_file', gf, 'to_grid_n', P0.grid.N_G, 'to_grid_dx', P0.grid.DX_G, ...
        'LR1_Kc', P0.pdi.psri.LR1_Kc, 'LR2_Kc', P0.pdi.psri.LR2_Kc, ...
        'REF_TRIM', P0.pdi.psri.REF_TRIM);
    png = fullfile(outdir, sprintf('%s_bs%g.png', o.tag, aoi));
    cl = dmg_bench_clearance('G', G, 'MODEL', o.MODEL, 'NGRID', o.NGRID, ...
        'MOUNT', o.MOUNT, 'BODY', o.BODY, 'quiet', true, 'draw', png);
    r = cl.rows;  v = cell2mat(r(:,6));  vf = v;  vf(isnan(vf)) = inf;
    say('%-22s %-12s %-5s %7s %7s %9s  %s\n', 'element','type','arm','a+mnt','r_beam','clear mm','against');
    for i = 1:size(r,1)
        say('%-22s %-12s %-5s %7.1f %7.1f %9.1f  %s\n', r{i,1},r{i,2},r{i,3},r{i,4},r{i,5},r{i,6},r{i,7});
    end
    [w, k] = min(vf);
    say('worst %+.1f mm (%s) over %d parts; spec >= %g -- %s\n', w, r{k,1}, nnz(~isnan(v)), o.MARGIN, ...
        iff_(w >= o.MARGIN, 'PASS', 'FAIL'));
    say('wrote %s\n\n', png);
    R(end+1) = struct('aoi',aoi, 'rows',{r}, 'worst',w, 'bind',{r{k,1}}); %#ok<AGROW>
end
out = struct('R', R, 'o', o);
save(fullfile(outdir, [o.tag '_clear.mat']), 'out');
say('wrote %s_clear.{txt,mat} + the figures in %s\n', o.tag, outdir);
end

function s = iff_(c, a, b)
if c, s = a; else, s = b; end
end

function say_(fid, varargin)
fprintf(varargin{:});  fprintf(fid, varargin{:});
end
