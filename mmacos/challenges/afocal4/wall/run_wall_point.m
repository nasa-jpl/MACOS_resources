% RUN_WALL_POINT  One frontier point, one MATLAB process, driven by the
% environment -- so RUN_WALL_FLEET.SH can start the whole frontier in
% parallel and each point checkpoints itself (the AFOCAL4_BASIN2 'tag'
% pattern, and the save-resumable-workspaces rule: a re-solve here is an
% hours-long artifact and must survive the process that produced it).
%
%   WALL_TILT    extraction tilt, deg          (required)
%   WALL_UMIN    the wall's floor, mm          (0)
%   WALL_ON      1 = wall on, 0 = off          (1)
%   WALL_EVALS   evaluations per restart round (300)
%   WALL_ROUNDS  restart rounds                (3)
%   WALL_TAG     artifact suffix               (derived from tilt/umin)
%   WALL_STANDOFF  mm -- force the field-mirror standoff, skip the seeder
%   WALL_DOFS      comma list, e.g. "conic,front" -- overrides the DOF set
%   WALL_PUPILW    multiply the blur/breathing/wander merit weights by this
%
%   Batch:  MACOS_HOME=~/dev/macos/macos_f90 \
%           WALL_TILT=-9 WALL_UMIN=15 matlab -batch "run('run_wall_point.m')"

run('/home/dcr/dev/MACOS_res_dev/mmacos/mmacos_setup.m');
addpath(fileparts(mfilename('fullpath')));

wp_tilt   = str2double(getenv('WALL_TILT'));
wp_umin   = str2double(getenv('WALL_UMIN'));    if isnan(wp_umin),   wp_umin = 0;   end
wp_on     = str2double(getenv('WALL_ON'));      if isnan(wp_on),     wp_on = 1;     end
wp_evals  = str2double(getenv('WALL_EVALS'));   if isnan(wp_evals),  wp_evals = 300;  end
wp_rounds = str2double(getenv('WALL_ROUNDS'));  if isnan(wp_rounds), wp_rounds = 3;   end
wp_tag    = getenv('WALL_TAG');
wp_soff   = str2double(getenv('WALL_STANDOFF'));   % mm; NaN = use the seeder
wp_dofs   = getenv('WALL_DOFS');                   % comma list; '' = default
wp_pw     = str2double(getenv('WALL_PUPILW'));     % pupil weight multiplier

if ~isfinite(wp_tilt)
    error('run_wall_point:tilt','set WALL_TILT (deg).');
end

wp_args = {'tilt',wp_tilt, 'union_min',wp_umin/1e3, 'wall',wp_on ~= 0, ...
           'evals',wp_evals, 'rounds',wp_rounds, 'tag',wp_tag};
if isfinite(wp_soff), wp_args = [wp_args, {'seed_standoff', wp_soff/1e3}]; end
if ~isempty(wp_dofs), wp_args = [wp_args, {'dofs', strsplit(wp_dofs,',')}]; end
if isfinite(wp_pw),   wp_args = [wp_args, {'pupil_w', wp_pw}]; end
R = wall_point(wp_args{:}); %#ok<NASGU>
exit(0);
