function C = dmg_leg_colors()
%DMG_LEG_COLORS  The gauge decks' ray-colour convention (Dave, 2026-10-05), one
%   place: red = source -> splitter -> reference flat (and back), blue = splitter
%   -> DM and back, green = splitter -> camera.  Used by dmg_leg_draw and by the
%   figure titles/legends that name the legs.
C = struct('reference', [200 30 30]/255, 'test', [30 90 190]/255, 'camera', [0 140 60]/255, ...
           'legend', 'reference arm red, test arm (DM) blue, camera leg green');
end
