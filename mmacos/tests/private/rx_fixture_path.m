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
roots = { ...
    fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'pymacos', 'tests', 'Rx'), ...
    fullfile(fileparts(fileparts(mfilename('fullpath'))), 'Rx')};
for i = 1:numel(roots)
    p = fullfile(roots{i}, name);
    if exist(p, 'file'), return; end
end
error('rx_fixture_path:notFound', ...
    'Rx fixture not found: %s (searched %s)', name, strjoin(roots, ', '));
end
