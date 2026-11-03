function p = design_src_path()
%DESIGN_SRC_PATH  Absolute path to mmacos/design/src, anchored to THIS clone.
%   The design LIBRARY (design_report, zern_jacobian_solve,
%   field_zone_lmon, fold_station_report, ...) is not on the test path
%   by default -- run_mmacos_tests.sh adds src/ and design/runners/
%   only.  Tests that reach into it use this helper.
%
%   Repo-relative on purpose: a hard-coded ~/dev/MACOS_resources/... made
%   a test in one worktree exercise ANOTHER worktree's library.
    here = fileparts(mfilename('fullpath'));          % mmacos/tests/private
    p = fullfile(fileparts(fileparts(here)), 'design', 'src');
end
