function p = iodc2010_params()
%IODC2010_PARAMS  Defaults for the IODC 2010 "green lens" challenge runner.
%   p = iodc2010_params() returns the documented knob struct; edit a field
%   or pass name/value pairs to iodc2010_run.  Every number a report prints
%   traces to one of these or to the prescription.
%
%   The problem (Juergens, Proc. SPIE 7652, 76522I, Sec. 2.1-2.2):
%     lambda_nm        532      monochromatic, doubled YAG
%     glass            N-BK7    Schott, the only glass
%     efl_mm           100      |EFL| = 100.00 +- 0.01 mm, paraxial
%     efl_tol_mm       0.01
%     rms_limit        0.070    waves RMS, chief-ray reference, tilt & focus kept
%     dist_limit       5.0      percent, |100 (yR - yP)/yP|, at the image plane used
%   Scoring (ours; README "The metric"):
%     nGridpts         97       rays across the paraxial entrance-pupil diameter
%                               (7081 in the disc).  The RMS is a lattice sum, so
%                               it depends on the grid: the f/10 singlet on axis
%                               reads 0.09095 / 0.09330 / 0.09439 / 0.09436 waves
%                               at 33 / 65 / 97 / 127 across (probe 2026-10-30);
%                               97 is converged to 4e-4 relative and costs 0.02 s
%                               per field.  The 2006 evaluator quotes CODE V with
%                               512 rays across the pupil.
%     field_scan_deg   the semi-fields scanned in order; the scan stops at the
%                      first failing field and bisects back to theta_tol_deg
%     theta_tol_deg    0.01     the semi-field limit's resolution
%     semidia_default  the clear semi-diameter written when a lens does not
%                      give its own and the footprint sizer is off (mm)
%     size_apertures   true: size every clear aperture to the beam footprint
%                      at the scan's largest PASSING field (the "no mounting
%                      material beyond the clear aperture" loophole), then
%                      check the edge thickness there
%     src_standoff_mm  how far upstream of the lens the source plane sits
%     model_size       128
%     outdir           where reports, decks and figures go (this directory)
p = struct();
p.lambda_nm       = 532;
p.glass           = 'N-BK7';
p.efl_mm          = 100;
p.efl_tol_mm      = 0.01;
p.rms_limit       = 0.070;
p.dist_limit      = 5.0;
p.nGridpts        = 97;
p.field_scan_deg  = [0 0.5 1 2 3 4 5 7.5 10 12.5 15 17.5 20 25 30 35 40 45 50 55 60];
p.theta_tol_deg   = 0.01;
p.semidia_default = 60;
p.size_apertures  = true;
p.aperture_margin = 1e-6;      % relative margin on a footprint-sized aperture
p.src_standoff_mm = 100;
p.model_size      = 128;
p.outdir          = fileparts(mfilename('fullpath'));
end
