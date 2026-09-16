function out = dmg_cam_line(rep, cam, dxd_mm, msk)
%DMG_CAM_LINE  The camera, on the pupil image THIS bench actually forms.
%   DMG_CAM_LINE(REP, CAM, DXD_MM, MSK) prints, and returns, what a named
%   camera gives on the measured pupil image: raw pixels across the pupil at
%   its pitch, the binning that lands nearest the modeled sampling, and -- if
%   CAM names one -- the snapshot polarization camera's sampling per analyzer
%   orientation.  The pupil diameter comes from the lit mask (equal-area
%   circle) times the measured detector pixel, so this is the TRACED image,
%   not a design intent (realism item 4: the model's NGRID is a sampling
%   FLOOR, not a sensor).
%
%   CAM fields: .name .pitch_um .bin, optionally .pol_name .pol_pitch_um.
%
%   TWO THINGS THIS GETS RIGHT THAT ARE EASY TO GET WRONG.
%   1. It reports the binning that MATCHES the model alongside the one the
%      sheet asked for.  Measured on the PSI rig, a 6.5 um sCMOS on its
%      7.8 mm pupil image gives 1197 raw px; binned 4 that is 299 against a
%      modeled 384, i.e. the sheet's binning UNDERSAMPLES the model by 22 %.
%      Binning 3 lands at 399.  Printing only the configured binning hides
%      that.
%   2. A micro-polarizer array's four orientations sit on a 2x2 superpixel,
%      so ACROSS A LINE each orientation is sampled every SECOND pixel --
%      N/2, not N/4.  N/4 is the area fraction, and using it for a linear
%      count understates the sampling by 2x.
%   REP may be a report FILE ID (dmg_say's convention, which zwfs_run uses) or
%   a printf-like FUNCTION HANDLE (tg96_run's `say`, which already has the file
%   id bound).  Accepting both is not politeness: dmg_say does
%   fprintf(rep,...), so handing it tg96_run's closure throws, and it would
%   throw inside a stage that runs an hour into a queued job.
if isempty(cam) || ~isstruct(cam), out = struct(); return; end
if isa(rep, 'function_handle')
    emit = rep;
elseif isempty(rep)
    emit = [];
else
    emit = @(varargin) dmg_say(rep, varargin{:});
end
d_px  = sqrt(4*nnz(msk)/pi);            % pupil diameter in modeled px
d_mm  = d_px * dxd_mm;
raw   = d_mm / (cam.pitch_um*1e-3);
binq  = max(1, round(raw/d_px));        % the binning that lands on the model
out = struct('d_mm',d_mm, 'd_px',d_px, 'raw_px',raw, 'bin_cfg',cam.bin, ...
             'bin_match',binq, 'binned_cfg',raw/cam.bin, 'binned_match',raw/binq);
if isempty(emit), return; end
emit(['camera: pupil image %.2f mm across (%.0f modeled px at %.1f um); ' ...
    '%s at %.2f um -> %.0f raw px across the pupil; binned %d = %.0f, ' ...
    'and binning %d = %.0f lands nearest the modeled %.0f\n'], ...
    d_mm, d_px, dxd_mm*1e3, cam.name, cam.pitch_um, raw, ...
    cam.bin, raw/cam.bin, binq, raw/binq, d_px);
if isfield(cam, 'pol_pitch_um') && ~isempty(cam.pol_pitch_um)
    npol = d_mm / (cam.pol_pitch_um*1e-3);
    emit(['  snapshot analyzer: %s at %.2f um -> %.0f px across the pupil, ' ...
        '%.0f per orientation (every 2nd pixel in each direction on the 2x2 ' ...
        'superpixel -- N/2 across a line, not N/4); four simultaneous frames, ' ...
        'no rotating stage\n'], cam.pol_name, cam.pol_pitch_um, npol, npol/2);
end
end
