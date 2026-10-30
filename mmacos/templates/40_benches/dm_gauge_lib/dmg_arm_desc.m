function A = dmg_arm_desc(rx, b, ix, base_deg)
%DMG_ARM_DESC  Describe one TG arm for dmg_ifo_gauge.
%   A = dmg_arm_desc(RX, BENCH, IDX, BASE_DEG) with BENCH = G.bt/G.br
%   and IDX = G.T/G.R from macos.design.twyman_green.
%   Extracted verbatim from tg96_s3/tg96_eprime @ 10cf593.
nm = {b.E.name};
A = struct('rx', rx, 'b', b, 'iPol', find(strcmp(nm,'PolIn'),1), ...
    'iQ', find(contains(nm,'QWP') & ~strcmp(nm,'OutQWP')), ...
    'base', base_deg, 'qwp_deg', base_deg, 'oq_deg', 0, 'iTO', [], ...
    'iRC', ix.iRC, 'iOQ', ix.iOutQWP, 'iAn', ix.iAnalyzer, 'iDET', ix.iDET);
if isfield(ix,'iTO'), A.iTO = ix.iTO; end
end
