function dmg_leg_draw(ax, E, arm, varargin)
%DMG_LEG_DRAW  Draw the LOADED + TRACED deck on ax with the deck's three legs in
%   the gauge decks' colour convention (Dave, 2026-10-05), the same on every rig:
%     red    source -> splitter -> reference flat and back to the splitter
%            (the reference arm; on the test deck the shared source leg)
%     blue   splitter -> DM (test optic) and back to the splitter
%     green  splitter -> recombination -> focuser -> mask seat -> camera
%   E is the arm's element struct array (names), ARM is 'test' | 'ref' | 'auto'
%   ('auto' = 'test' when an element is named TestOptic, else 'ref'); any further
%   name/value pairs go to macos.view_rx (e.g. 'bundle','rim','bodies','outline').
%   The legs are found by NAME: the splitter's faces are the elements whose names
%   start with 'BS' (BSrefl / BStxff / BSbinr ... on both rigs); the first is the
%   splitter's entry and the last its exit.  A deck with no 'BS' element is drawn
%   whole in green (a sensor-only leg).  Passive Reference planes are hidden.
%   Each leg is one macos.view_rx call over an element RANGE ('elts', [k0 k1]):
%   the optics of each range are drawn with it, the splitter at a boundary twice.
%
%   Why a shared helper: five producers (dmg_bench_clearance, tg96_run's render,
%   zwfs_vlayout, pdi_layout_fig, psri_layout_fig) each coloured whole arms by
%   hand, two colours on some, one on others, and the shared camera leg took
%   whichever arm was drawn last -- the deck's layouts disagreed with each other.
C = dmg_leg_colors();
nm = {E.name};
isbs = startsWith(nm, 'BS');
passive = find(strcmp({E.element}, 'Reference'));
nE = numel(E);
if ~any(isbs)
    macos.view_rx('ax', ax, 'ray_color', C.camera, 'title', '', 'labels', false, 'hide', passive, varargin{:});
    return
end
kin = find(isbs, 1, 'first');  kout = find(isbs, 1, 'last');
if strcmp(arm, 'auto')
    if any(strcmp(nm, 'TestOptic')), arm = 'test'; else, arm = 'ref'; end
end
switch arm
    case 'test', cmid = C.test;
    case 'ref',  cmid = C.reference;
    otherwise, error('dmg_leg_draw:arm', 'arm must be test | ref | auto');
end
legs = {[0 kin], C.reference; [kin kout], cmid; [kout nE], C.camera};
for i = 1:size(legs, 1)
    r = legs{i, 1};
    if r(2) <= r(1), continue; end
    macos.view_rx('ax', ax, 'ray_color', legs{i, 2}, 'title', '', 'labels', false, ...
                  'hide', passive, 'elts', r, varargin{:});
end
end
