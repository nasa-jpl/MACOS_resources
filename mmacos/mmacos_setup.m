% mmacos_setup.m -- put the mmacos source + helper directories on the MATLAB path.
% =====================================================================
%  Run this ONCE per MATLAB session before using the +macos package, the
%  mmacos mex, or any script under examples/.  It anchors every path to its
%  OWN location, so it works wherever mmacos is installed -- no hard-coded
%  user paths, no fragile relative reaches from each example.
%
%  Dirs added (relative to this file):
%      src/                     the +macos package + the mmacos mex
%      design/runners/          the STAGE RUNNERS (run_met, ... -- the
%                               product pipeline; see design/runners/README.md)
%      sensitivities/           plot_opd_canvas / plot_dw_channels / save_dw_multi
%      examples/sensitivities/  mimg / m2v / v2m / pad display helpers
%
%  Cold start (mmacos root not yet on the path):
%      >> run('<path-to>/mmacos/mmacos_setup.m')
%  Once the root is reachable (or after the line above):
%      >> mmacos_setup
%
%  Add either form to your startup.m to make it automatic.
% =====================================================================

mmacos_root_ = fileparts(mfilename('fullpath'));
addpath(fullfile(mmacos_root_, 'src'));                       % +macos package + mmacos mex
addpath(fullfile(mmacos_root_, 'design', 'runners'));        % stage runners (the pipeline)
addpath(fullfile(mmacos_root_, 'sensitivities'));            % plot_*/save_dw_multi
addpath(fullfile(mmacos_root_, 'examples', 'sensitivities'));% mimg / m2v / v2m / pad
fprintf('mmacos: path set from %s\n', mmacos_root_);
clear mmacos_root_;
