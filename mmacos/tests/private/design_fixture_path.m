function p = design_fixture_path(name)
%DESIGN_FIXTURE_PATH  Absolute path to a shared optical-design fixture.
%   The telescope-design reference fixtures live one level up, shared by
%   mmacos and pymacos, at MACOS_resources/optical_design/fixtures/ (see
%   that dir's README + OPTICAL_DESIGN_AGENT_GUIDE.md).  Use this helper
%   so the physical layout can change without touching every test.
%
%   Example:
%       fx = jsondecode(fileread( ...
%               design_fixture_path('telescope_design_fixtures.json')));
    arguments
        name (1,:) char
    end
    root = fullfile(getenv('HOME'), 'dev', 'MACOS_resources', ...
        'optical_design', 'fixtures');
    p = fullfile(root, name);
    if ~exist(p, 'file')
        error('design_fixture_path:notFound', ...
            'optical-design fixture not found: %s', p);
    end
end
