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
root = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
    'pymacos', 'tests', 'Rx');
p = fullfile(root, name);
if ~exist(p, 'file')
    error('rx_fixture_path:notFound', 'Rx fixture not found: %s', p);
end
end
