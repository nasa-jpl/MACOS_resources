function h = macos_home_()
%MACOS_HOME_  The MACOS_HOME a nested `matlab -batch` needs for macos_param.txt: the parent's, or
%   the macos_f90 directory beside this checkout (a system() from MATLAB on macOS drops the parent
%   environment, so the nested engine init found no parameter file; CCMac's Mac gate 2026-10-19).
h = getenv('MACOS_HOME');
if isempty(h)
    res = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));   % MACOS_resources
    h = fullfile(fileparts(res), 'macos', 'macos_f90');
end
end
