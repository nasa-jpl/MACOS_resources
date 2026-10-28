function md = modes()
%MODES  Test stub for macos.modes: the session with CENTROID in force.
%   The bindings have no setter for the engine's CENTROID/CHIEFRAY mode
%   (only the CLI's CENTROID command sets it), so a test that needs
%   "the session's FEX mode is 0" puts this folder on the path with a
%   matlab.unittest.fixtures.PathFixture.  Every read is counted in
%   appdata 'stub_modes_reads', so the test can see that the code under
%   test consulted the session.  Used by tDesignTelescope.
n = getappdata(0, 'stub_modes_reads');
if isempty(n), n = 0; end
setappdata(0, 'stub_modes_reads', n + 1);
md = struct('pol', false, 'vecdif', false, 'centroid', true, ...
            'lneg', true, 'cir', false, 'in_force', {{'CENTROID'}});
end
