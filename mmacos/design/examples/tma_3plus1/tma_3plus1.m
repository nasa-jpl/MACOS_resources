% tma_3plus1.m  (mmacos/design/examples/ -- coronagraph front-end: 3 mirrors + 1)
% =====================================================================
%  3+1: sz_tma's three mirrors form the science focus; a 4th FIELD MIRROR
%  (M4, just past that focus) relays the system pupil to a downstream,
%  accessible plane and -- the point -- can FLATTEN it (null the pupil
%  defocus + astigmatism that the 3-mirror system leaves: +1.67 mm
%  defocus + 1.77 mm astig, measured by macos.pupil_quality).  A surface
%  at an IMAGE does pupil work with ~no image effect (field mirror); this
%  is the lever.  M4 conic first (sphere+FF if a conic can't flatten it).
%
%  THIS FIRST PASS: geometry + see WHERE the pupil lands (location +
%  diameter) and its baseline quality (M4 unoptimized).  [Future: spec a
%  target pupil DIAMETER to fit a DM/Lyot.]  M4 flattening = next step.
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% sz_tma base (D, lambda, the three base spheres + folds)
D=8.0; LAM=1e-6; MODEL=256; GRIDN=41;
R   = [51.534, 8.871, 3.0];      TBET=[22.0, 28.0];   TILT=[-7.2, 8.46, 12.0];

% --- M4 field-mirror knobs (M3 back-focus is ~3 m; put M4 just past it) -
% The builder makes the LAST mirror 'derive' its FP distance, so M4
% reimages the M3 focus to FP2 (a relay).  M4 near the focus = field
% mirror (pupil work, ~no image effect).  f4 sets the relay conjugate:
% s ~ 2 m past focus, f4=1.5 m -> FP2 ~6 m past M4.
M4_SPACE = 5.0;     % M3 -> M4 (m): ~2 m past the ~3 m M3 focus
M4_R     = -3.0;    % M4 radius (m), concave (f4=1.5 m)
M4_TILT  = 10.0;    % fold M4 out (deg, about x)

fprintf('=== 3+1 coronagraph front-end (sz_tma + M4 field mirror) ===\n');

t = macos.design.Telescope('family','TMA','aperture_diameter_m',D, ...
        'model_size',MODEL,'wavelength_m',LAM,'grid_npts',GRIDN);
t.set_base_sphere(true);
t.add_mirror('M1','radius_m',R(1),'spacing_after_m',TBET(1),'tilt_deg',TILT(1));
t.add_mirror('M2','radius_m',R(2),'spacing_after_m',TBET(2),'tilt_deg',TILT(2),'convex',true);
t.add_mirror('M3','radius_m',R(3),'spacing_after_m',M4_SPACE,'tilt_deg',TILT(3));
t.add_mirror('M4','radius_m',M4_R,'spacing_after','derive','tilt_deg',M4_TILT);
t.add_focal_plane('FP2');
t.build();  t.describe();
nE = numel(t.spec.elt);
fprintf('\nbuilt %d elements (M1 M2 M3 M4 + FP2)\n', nE);

s = macos.trace(nE);
fprintf('trace: nRays=%d  rmsWFE=%.3e (base spheres, image uncorrected)\n', ...
        s.nRays, s.rmsWFE);

rep = t.check_clipping('noload',true,'quiet',true);
fprintf('clearance: %d/%d optics clear -> %s\n', sum([rep.ok]), numel(rep), ...
        ternary(all([rep.ok]),'UNOBSCURED','OBSCURED'));

% --- the point: WHERE does the pupil land, and how good/big is it? ----
t.add_pupil(nE);                     % FEX-place an EP slot at nElt-1
nE = numel(t.spec.elt);
try
    pq = macos.pupil_quality(nE-1);
    fprintf('\n--- 3+1 EXIT PUPIL (after M4) ---\n');
    fprintf('  lands at [%.3f %.3f %.3f] m,  diameter %.1f mm\n', ...
            pq.vertex, pq.diameter*1e3);
    fprintf('  defocus %+.3f  astig [%+.3f %+.3f] mm  (3-mirror alone: 1.67 / 1.77)\n', ...
            pq.defocus*1e3, pq.astig*1e3);
catch ME
    fprintf('\npupil_quality failed: %s\n', ME.message);
end

try
    f1 = t.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f1, fullfile(exdir,'tma_3plus1_layout.png'));
    fprintf('layout: tma_3plus1_layout.png\n');
catch ME, fprintf('layout skipped (%s)\n', ME.message); end

function s=ternary(c,a,b), if c, s=a; else, s=b; end, end
