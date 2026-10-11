function P = cophi_params()
%COPHI_PARAMS  The COPHI (heterodyne camera interferometer) campaign's sheet.
%   P = cophi_params returns tg96_params with the reading set to the heterodyne
%   form 'H' (dm_gauge_lib/dmg_cophi_*): the TG96 bench -- lens rig and mirror
%   rig, the collimated source, the splitter node, the imaging tail, the camera
%   -- with the REFERENCE FLAT'S SEAT taken by the LO arm (Dave's ruling 1,
%   BRIEF_cophi_campaign.md sec. 7: traced, as a drop-in).  Same DM truth, same
%   battery, same loop as the four gauges, so every table of the composite deck
%   gains one column.  Every P.cophi knob is documented here; the rest in
%   tg96_params.m.
%       P = cophi_params;  out = cophi_run(P);            % the record
%       cophi_run('cophi.cam', 'photonfocus', 'stages', {'bench','noise'})
%   Runs land in <this dir>/runs/<tag>/.  Gate: tests/tCophi.m (the algebra).
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('tg96_params'))                 % the code is SHARED, not copied
    addpath(fullfile(exdir, '..', 'tg_psi_dm96_oap'));
end
if isempty(which('dmg_cophi_frames'))
    addpath(fullfile(exdir, '..', 'dm_gauge_lib'));
end
P = tg96_params();
P.tag = 'cophi';
P.outdir = '';                                   % '' = <this dir>/runs/<tag>
P.readings = {'H'};                              % the heterodyne reading (beside the record's I when compared)
P.noise.readings = {'H'};
P.loop.readings  = {'H'};
P.loop.drifts    = {'walk', 'thermal', 'cam', 'lo'};   % + the LO-arm drift row (ruling 1)

% ---- the beat and the buckets (Feng's layout: two AOMs on a Rb standard) ---
P.cophi.f_beat    = 7.5;        % Hz, the AOM difference frequency (80 MHz vs 80 MHz + 7.5 Hz)
P.cophi.N         = 4;          % buckets per beat (4 at 30 fps = Feng's four-step; N > 4 = the LSQ backup)
P.cophi.frame_hz  = [];         % [] = N * f_beat (30 Hz); set to run the camera slower than the beat allows
P.cophi.jitter    = 0;          % rad of beat phase, trigger jitter rms per frame (same clock: ~0)
P.cophi.walk      = 0;          % rad per frame rms, DIFFERENTIAL fiber-phase walk within a scan (the new error term)
P.cophi.lo_ratio  = 1.0;        % LO power / test power at the camera (g = 2 sqrt(r)/(1+r); 1 = full contrast)
P.cophi.lo_drift  = struct('tilt_rad', 0, 'defocus_mm', 0);   % the LO arm's wavefront drift PER CYCLE (loop kind 'lo')

% ---- the cameras (Dave's ruling 3: several, read noise in EVERY comparison) --
% From NOTE_detectors_survey.md -- VERIFY against the vendor sheet before a number
% enters a slide (the survey is a web survey).  well = full well e-, read = e- rms
% per frame, rate = max full-frame fps with a TRUE global shutter, pitch um.
P.cophi.cameras = struct( ...
    'baseline',    struct('name', 'as proposed: 1k x 1k, 60 ke, 30 fps (read noise ASSUMED 5 e-)', 'npix', 1024, 'well', 60e3,  'read', 5,   'rate', 30,  'pitch_um', 6.5), ...
    'photonfocus', struct('name', 'Photonfocus MV1-D1024E-160-CL (A1024B, 200 ke, 220 e- read)',   'npix', 1024, 'well', 200e3, 'read', 220, 'rate', 150, 'pitch_um', 10.6), ...
    'pcoedge55',   struct('name', 'pco.edge 5.5 in TRUE global-shutter mode (30 ke, 2.2 e-)',      'npix', 2048, 'well', 30e3,  'read', 2.2, 'rate', 50,  'pitch_um', 6.5), ...
    'cred2',       struct('name', 'First Light C-RED 2 (InGaAs 640 x 512, for the 1.3 um row; read ASSUMED 30 e-)', 'npix', 640, 'well', 1.4e6, 'read', 30, 'rate', 600, 'pitch_um', 15));
% photons/s at full well (survey): baseline 1.9e12 -> 17 pm per 1 s reading on the deck's scaling;
% pco.edge GS 8.3e12 -> 8.0; Photonfocus 3.1e13 -> 4.1 (its 220 e- read noise then costs ~22 % in
% sigma at half well: (I0 + r^2)/I0 = 1.48 -- the reason read noise is in every row).
P.cophi.cam       = 'baseline';   % the record's camera; the noise stage runs every entry
P.cophi.fill      = 0.5;          % frame MEAN at this fraction of the well (fringe peak = fill (1 + g))

% ---- wavelength (ruling 2): run the record at 632.8; 1.3 um is a STAGE (InGaAs, refocused tail)
P.cophi.lam_rows  = [6.328e-4];   % mm; add 1.319e-3 for the SIM-MET-class row (needs 'cred2' + tail refocus)

% ---- the second sensor class (Dave's ruling 6, 2026-10-11: FULL actuator resolution, no
% low-order forms -- EFC probes are commanded at the full lattice).  A well-free (continuous
% or lock-in) reading at >= 1 element per actuator, ASSUMED to exist at 1:1 or better with the
% pitch; scored on cadence and photons per element per second, never on resolution.
% Candidates (survey NOTE_lockin_arrays_survey.md, verify): lock-in cameras (heliCam class),
% ToF demodulating sensors, 128 x 128 coherent (Lincoln-Lab-class) arrays.
P.format.lockin   = struct('sites', 128, 'f_beat_hz', 10e3, 'rate_hz', 1e3, ...   % sites per side; beat; I/Q output rate
                           'flux_e_per_s', 1e9, 'phasemeter_rad', 1e-3);          % photons per element per second; demod noise floor per sample
P.format.npix     = [];                   % camera formats below 1:1 are NOT run (ruling 6)
end
