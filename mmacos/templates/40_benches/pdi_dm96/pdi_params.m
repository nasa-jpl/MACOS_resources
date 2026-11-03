function P = pdi_params()
%PDI_PARAMS  The point-diffraction readings' parameter sheet: zwfs_params with
%   the P / PF record settings.  The point-diffraction interferometer readings
%   (P = stepped pinhole at the FocalMask, common path; PF = the P/SRI of Dube
%   et al. 2024, a waveguide reference with a photonic phase shifter) live in
%   the SAME runner as the Zernike sensor -- same bench, same DM truth, same
%   battery, same loop -- so this sheet is zwfs_params with the readings,
%   noise and loop lists set to the PDI record and the camera-drift loop kind
%   added.  Every P.pdi knob not listed below is documented in zwfs_params.m.
%       P = pdi_params;  out = pdi_run(P);                 % the record
%       pdi_run('pdi.DIA_LAMD', 1.0, 'stages', {'bench','battery','figs'})
%       ./pdi_batch.sh TAG "pdi_params, 'stages',{'bench','loop','figs'}"
%
%   Runs land in <this dir>/runs/<tag>/.  Records taken before 2026-09-13
%   (pdi193*, ploop193, pcam193*) are in ../zwfs_dm96/runs/ -- they are cited
%   by deck_pdi and were left in place; see README.md.
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end
if isempty(which('zwfs_params'))                 % the code is SHARED, not copied
    addpath(fullfile(exdir, '..', 'zwfs_dm96')); % zwfs_run, zwfs_params, zwfs_mask
end
P = zwfs_params();
P.tag = 'pdi';
P.outdir = '';                                   % '' = <this dir>/runs/<tag> (pdi_run sets it)
P.readings = {'S', 'V', 'P', 'PF'};              % the stepped and vector Zernike readings beside the two PDI forms
P.noise.readings = {'S', 'V', 'P', 'PF'};
P.loop.readings = {'S', 'V', 'P', 'PF'};
P.loop.drifts = {'walk', 'thermal', 'cam'};      % + the camera 1/f drift (P.loop.cam_walk)

% ---- knobs added with the gauge-deck slice (2026-09-13) -------------------
% They default in zwfs_run's parse_ (so an unedited zwfs_params sheet still
% runs every stage); this is where they are documented and set.

% THE DESCENT -- capturing the DM's initial figure (Dave 2026-09-13: "every
% configuration must show how it gets from 100-200 nm WFE to the hold
% regime").  With start_rms set, the loop stage adds a descent run per
% (reading, recal_list entry, photon level): the DM starts at a surface of
% this rms -- the set point's own random field, rescaled -- with the response
% matrix measured THERE, and the loop must bring it to the set point.
P.loop.start_rms   = [];      % mm rms of the starting surface ([] or 0 = off; 100e-6 = the record)
P.loop.recal_list  = [];      % the recal_every values compared in the descent ([] = [P.loop.recal_every])
P.loop.recal_every = 0;       % cycles between re-measurements of the response matrix ON the loop's
                              % current surface, through the instrument's own calibration (0 = never:
                              % the matrix measured once, at the start, is used throughout).  Each
                              % re-calibration costs battery.matrix_step^2 + 2 instrument states.
P.loop.reach = [10e-6 3e-9];  % mm: the levels whose first cycle the descent table reports (10 nm, 3 pm)

% WITHIN-MEASUREMENT DRIFT (V4).  The fraction of each cycle's drift
% increment that develops ACROSS one measurement's scan: frame j of nf is
% captured at (j-1)/(nf-1) of it.  A temporally stepped reading (S, P, PF)
% captures its frames one at a time and pays for this; a single-frame
% reading (L, I+) and a simultaneous pair (V) see one instant and do not.
% The DM/thermal analogue of P.loop.cam_intra, and the IFO's PZT-form drift
% term.  Cost: a stepped reading traces every frame separately.
P.loop.intra = 0;             % 0 = the DM is still while a scan is taken; 1 = the whole increment

% THE P/SRI's OWN REFERENCE ARM.
P.pdi.bench = 'zwfs';         % 'zwfs' = the ZWFS test arm with the reference SYNTHESIZED (the record
                              % through 2026-09-12: PF's reference is the recollimated LP01 mode);
                              % 'psri' = macos.design.psri_bench's TWO DECKS, the reference arm traced
                              % through its own pinhole (readings must be {'PF'}: the test arm's seat
                              % is empty on that bench, so no dimple reading and no common-path P)
P.pdi.ref_frozen = false;     % 'psri' bench only: trace the reference arm ONCE, on the flat, and
                              % reuse it.  The control that separates the real arm's SHAPE (which a
                              % frozen reference still carries) from its MOTION with the state
P.pdi.ref_walk = 0;           % rad per cycle rms: a random walk of the reference arm's phase relative
                              % to the test arm -- the NON-COMMON-PATH term, which no common-path
                              % reading has.  Applied to PF only, in the loop stage
P.pdi.ref_seed = 0;           % its stream (0 = dmg_loop's default, opt.seed + 2)

% macos.design.psri_bench overrides for P.pdi.bench 'psri'.  The three solved
% values are psri_layout_fig's (the reference lens's conic and the pinhole
% seat's trim to the TRUE focus, found by a scan on the traced bench -- the
% add_lens seed is 0.31 mm rms of ray blur at F/2.9, and the F/2.9
% diffraction focus is ~20 um deep, so neither is guessable).  Re-solve with
% psri_layout_fig; the front end and tail come from P.bench.
P.pdi.psri = struct('LR1_Kc', -0.5784, 'LR2_Kc', -0.5784, 'REF_TRIM', 1.0960);
end
