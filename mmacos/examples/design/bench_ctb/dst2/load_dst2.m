% load_dst2.m
% ===================================================================
%  DST2 (DST2R) FAITHFUL PRESCRIPTION -- import + trace
% ===================================================================
%  Companion to the generic coronagraph testbed (../example_ctb.m).  This
%  brings in the ACTUAL DST2R design (Brandon's CODE V layout, via
%  cv2macos) so we can develop against a faithful reference in parallel
%  with the clean generic CTB.
%
%  Source of record:
%    raw/dst2v2_aox_v5_di_2025_05_07_fiximgtilt.seq  -- Brandon's CODE V
%       sequence (ascii; CODEV 2024.03).  8 OAPs (K=-1), f = |RDY|/2 =
%       [2500 1524 1143 1350 675 635 635 762] mm -- IDENTICAL to the
%       generic CTB F_OAP seeds.  Stop at dm1.  500 nm, metres.
%    raw/DST2V2_AOX_V5_stop_at_dm1_F0{1..5}.IN  -- cv2macos output of that
%       design (one Rx per field point), the "stop at dm1" variant that
%       matches the .seq.  (cv2macos is a CODE V macro; it was run on the
%       lens to produce these -- CODE V is not required here.)
%
%  Two fix-ups are needed before the current engine will trace the
%  2013-era cv2macos output (applied by normalize_dst2 below, NON-destruc-
%  tive -- raw/ is untouched, normalized files land alongside as *_norm.in):
%    1. PARSE: tighten spaced keywords ('ApType = Circle' -> 'ApType=
%       Circle') and add BaseUnits/WaveUnits (mm), which the current
%       msmacosio reader wants and the old converter omitted.
%    2. APERTURES: the OAP ApVec circles carry the CODE V off-axis DECENTER
%       (e.g. 'ApVec= 75 200 0' = radius 75, 200 mm off axis), which the
%       engine applies about the element VERTEX (the parent-parabola
%       vertex), far from the off-axis beam -> every ray blocked (ok=0).
%       Same off-axis-aperture frame issue documented for add_oap.  We
%       neutralize them (ApType=None): the beam is well inside the optics,
%       so the clear apertures are not the functional stop (dm1 is).
%
%  Run:   >> run('.../dst2/load_dst2.m')     (Requires MACOS_HOME.)
% ===================================================================

addpath('~/dev/MACOS_resources/mmacos/src');
exdir = fileparts(mfilename('fullpath'));
if isempty(exdir), exdir = pwd; end
assert(~isempty(getenv('MACOS_HOME')), 'MACOS_HOME must be set.');
macos.init(256);

fields = 1:5;
fprintf('DST2 (stop-at-dm1) faithful import -- %d field points\n', numel(fields));
res = struct('field',{},'rx',{},'nElt',{},'nRays',{},'ok',{},'rmsWFE',{});
for f = fields
    raw  = fullfile(exdir, 'raw', sprintf('DST2V2_AOX_V5_stop_at_dm1_F0%d.IN', f));
    norm = fullfile(exdir, sprintf('dst2_F0%d_norm.in', f));
    normalize_dst2(raw, norm);
    macos.load_rx(norm);
    nE = macos.num_elt();
    s  = macos.trace(nE);  info = macos.get_ray_info(s.nRays);
    ok = nnz(info.ok_trace(:) & info.ok_pass(:));
    fprintf('  F0%d: nElt=%d nRays=%d ok=%d  RMS WFE=%.4g waves\n', ...
        f, nE, s.nRays, ok, s.rmsWFE);   % WaveUnits=mm, Wavelen in mm -> waves
    res(end+1) = struct('field',f,'rx',norm,'nElt',nE,'nRays',s.nRays, ...
                        'ok',ok,'rmsWFE',s.rmsWFE);   %#ok<AGROW>
    assert(ok > 0, 'DST2 F0%d failed to trace (ok=0)', f);
end

% render the on-axis field (F01) as a sanity view
macos.load_rx(res(1).rx);  macos.modify();  macos.trace(macos.num_elt());
fig = macos.view_rx('show','beam','bundle','rings','nrings',3,'nspokes',12,'bodies','solid');
set(fig, 'Color','w', 'Position',[100 100 1500 1000]);  axis equal; grid on;
title('DST2 (faithful, stop-at-dm1, F01) -- beam through the relay');
print(fig, fullfile(exdir,'dst2_F01_view_rx.png'), '-dpng', '-r150');

save(fullfile(exdir,'dst2.mat'), 'res');
fprintf('DONE.  Normalized Rx: dst2_F0{1..5}_norm.in\n');

% ===================================================================
function normalize_dst2(src, dst)
%NORMALIZE_DST2  Make a 2013-era cv2macos .IN loadable + traceable by the
%   current engine.  Non-destructive: reads SRC, writes DST.  See header.
    lines = strsplit(fileread(src), '\n');
    o = {};
    for i = 1:numel(lines)
        ln = lines{i};
        % (1a) tighten spaced keywords: 'Word = ' -> 'Word= '
        ln = regexprep(ln, '([A-Za-z0-9])\s+=\s*', '$1= ');
        % (2) neutralize off-axis clear apertures (mis-framed about vertex)
        ln = regexprep(ln, 'ApType=\s*Circle', 'ApType= None');
        if ~isempty(regexp(ln, '^\s*ApVec=', 'once'))
            continue;   % drop the ApVec data line that paired with the circle
        end
        o{end+1} = ln;   %#ok<AGROW>
        % (1b) add BaseUnits/WaveUnits (mm) right after zSource
        if ~isempty(regexp(ln, 'zSource=', 'once'))
            o{end+1} = '        BaseUnits=  mm';   %#ok<AGROW>
            o{end+1} = '        WaveUnits=  mm';   %#ok<AGROW>
        end
    end
    fid = fopen(dst, 'w');  assert(fid > 0, 'cannot write %s', dst);
    fprintf(fid, '%s\n', o{:});  fclose(fid);
end
