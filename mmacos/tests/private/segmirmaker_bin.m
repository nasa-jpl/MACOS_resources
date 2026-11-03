function bin = segmirmaker_bin()
%SEGMIRMAKER_BIN  Path to the SegMirMaker executable, whichever tree built it.
%   bin = SEGMIRMAKER_BIN() searches the four build trees the project uses
%   and returns the first hit, or '' if none is built.  ifx is preferred
%   when both compilers are present (it is what the committed .presc
%   references were generated with).
%
%   Five test classes used to hard-code 'build_release_ifx' alone, so on a
%   gfortran-only box (macOS, and any Linux box built with makegfortran.sh)
%   every segmentation / MET class SKIPPED -- and a skip reads as green.
%   Use this helper instead of a literal build-tag path.
%
%   See also tSegMirMaker, tSegmentRx, tEdgeSensors, tMet, tRunMet,
%   tRunSegmentation.

here = fileparts(fileparts(mfilename('fullpath')));   % mmacos/tests
res_root = fileparts(fileparts(here));                % MACOS_resources
smmdir = fullfile(res_root, 'segmirmaker');
bin = '';
for tag = ["build_release_ifx", "build_release_gfortran", ...
           "build_debug_ifx", "build_debug_gfortran"]
    cand = fullfile(smmdir, tag, 'SegMirMaker');
    if isfile(cand), bin = cand; return; end
end
end
