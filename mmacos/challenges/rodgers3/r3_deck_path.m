function p = r3_deck_path(G, dir)
%R3_DECK_PATH  The deck file of one build_r3 rung, resolved against its build.
%   p = r3_deck_path(G, dir): G is one rung of build_r3's D (D.r1 .. D.r5),
%   dir the directory that holds the r3_build*.mat it came from.  build_r3
%   stores G.deck as the bare file name, written beside the .mat
%   (2026-10-26, PLAN_CONSOLIDATION 1.2 (ttt)); a .mat from before then
%   holds an absolute path from the tree that built it, so only its file
%   name is kept.
%
%   See also: build_r3, rodgers3.
[~, n, e] = fileparts(G.deck);
p = fullfile(dir, [n e]);
end
