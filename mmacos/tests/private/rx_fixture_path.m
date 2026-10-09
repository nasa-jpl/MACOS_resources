function p = rx_fixture_path(name)
%RX_FIXTURE_PATH  Absolute path to a named Rx fixture.
%   Tests share the Rx corpus with pymacos under
%   MACOS_resources/pymacos/tests/Rx/.  Use this helper so the
%   physical layout can change without touching every test.
%
%   Example:
%       rx = rx_fixture_path('Rx_Cass_FarField.in');
arguments
    name (1,:) char
end
%   Fixtures that exist only for mmacos (no pymacos counterpart) live in
%   mmacos/tests/Rx/ and are found by the same call — the shared corpus is
%   searched first, the mmacos-local one second.
%   Roots are anchored to THIS clone first (repo-relative), with the
%   canonical ~/dev/MACOS_resources checkout as a fallback -- a
%   hard-coded absolute root made a test in one worktree read another
%   worktree's fixtures.
mm    = fileparts(fileparts(fileparts(mfilename('fullpath'))));  % mmacos
res   = fileparts(mm);                                           % repo root
roots = { ...
    fullfile(res, 'pymacos', 'tests', 'Rx'), ...
    fullfile(mm, 'tests', 'Rx'), ...
    fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'pymacos', 'tests', 'Rx')};
for i = 1:numel(roots)
    p = fullfile(roots{i}, name);
    if exist(p, 'file'), return; end
end
error('rx_fixture_path:notFound', ...
    'Rx fixture not found: %s (searched %s)', name, strjoin(roots, ', '));
end
