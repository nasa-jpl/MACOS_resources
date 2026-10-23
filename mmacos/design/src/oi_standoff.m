function s = oi_standoff(EPD_m)
%OI_STANDOFF  Source-plane standoff ahead of the entrance-pupil point, m.
%
%   The offset_imager chain launches its ray grid on a plane NORMAL TO
%   THE CHIEF, standing off the entrance-pupil construction point.  That
%   standoff was a hard-coded 0.75 m at six sites (oi_score, oi_clear,
%   oi_close x2, oi_solve, oi_layout_fig) -- correct for the 75-200 mm
%   instances the template was built on, and wrong the moment the
%   aperture is large, because the launch plane is TILTED by the field
%   offset: its rim sits +-(EPD/2)*sin(offset) in z.  At EPD 6 m and a
%   22.5 deg offset that is 1.15 m, so the rim of the grid starts BEHIND
%   M1's vertex and the engine reports EVERY ray a surface miss -- an
%   opaque "candidate would not trace" that looks like a bad design.
%   Found on the first 6 m instance (e2e6m, 2026-08-24).
%
%   The standoff scales with the aperture and NEVER falls below the
%   legacy value, so every committed instance (EPD <= 0.5 m) is
%   bit-identical: max(0.75, 1.5*EPD).  1.5x the diameter clears the
%   tilted rim at any offset up to 90 deg with 3x margin.
%
%   See also OI_SCORE, OI_CLEAR, OI_CLOSE, OI_SOLVE, OI_LAYOUT_FIG.
    arguments
        EPD_m (1,1) double {mustBePositive}
    end
    s = max(0.75, 1.5*EPD_m);
end
