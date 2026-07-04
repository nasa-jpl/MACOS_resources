% tma_centered.m  (mmacos/design/examples/ -- an adaptable design study)
% =====================================================================
%  CENTERED (OBSCURED) THREE-MIRROR ANASTIGMAT -- the j18 / early-JWST
%  design family -- and WHY it out-fields the unobscured section.
% =====================================================================
%  The two BASIC conic-TMA design families share one parent geometry:
%
%   CENTERED (this example): rotational symmetry intact.  Three conics
%     null spherical + coma + astigmatism ABOUT THE AXIS, so the
%     residual field astigmatism is high-order and the corrected field
%     is wide.  The price is the central obscuration (M2 in the beam)
%     -- fine for IR imaging (JWST), costly for coronagraphy.
%
%   UNOBSCURED SECTION (see ../tma_offaxis): an eccentric-pupil
%     off-axis section of the same parent clears the beam, but BREAKS
%     the rotational symmetry -- inducing field-dependent (binodal)
%     astigmatism that a fixed mirror Zernike cannot null (its
%     magnitude/orientation vary over the field).  The field wall
%     arrives sooner; wfe_field_diag shows the astig signature.
%
%  This example builds the CENTERED family on the j18mono geometry
%  (6.6 m f/20 convex-secondary Korsch), balances it over a CIRCULAR
%  field (macos.design.field_ring -- a square grid's corners over-ask
%  by sqrt(2)), diagnoses the per-field aberration ladder, then runs
%  the SECTION arm of the same recipe and prints the A/B -- the
%  quantitative answer to "why does j18 do better?".
%
%  Run:  >> run('.../design/examples/tma_centered/tma_centered.m')
% =====================================================================
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/src'));
addpath(fullfile(getenv('HOME'),'dev/MACOS_resources/mmacos/design/src'));
exdir = fileparts(mfilename('fullpath'));  if isempty(exdir), exdir = pwd; end

% ====================  USER DESIGN CHOICES  ==========================
D          = 6.605;                          % aperture (m) -- j18mono
R          = [15.879722 1.778913 3.016227];  % |radii| M1/M2(convex)/M3
TBET       = [7.169041556 7.965313479];      % spacings (m)
LAM        = 1.0e-6;                         % score at 1 um (j18's own
                                             % band is 2.3 um: /2.3)
FIELD_RAD  = 2.5;                            % circular-field RADIUS (arcmin)
                                             % (= 5' diameter)
DIFFRACTION_LIMIT = 0.071;                   % Marechal, waves RMS
% =====================================================================

optF = macos.design.field_ring(FIELD_RAD, 'units','arcmin');

fprintf('====================================================================\n');
fprintf(' Centered (obscured) TMA -- j18 family | D=%.2f m | %g''-dia circular field\n', ...
        D, 2*FIELD_RAD);
fprintf('====================================================================\n');

%% -- [1] the CENTERED family through the shared recipe ---------------
[tc, rc] = tma_conic_recipe('D',D,'R',R,'spacings',TBET,'lambda',LAM, ...
                            'section',false,'fields',optF);
fprintf('\n[1] CENTERED: axial %.4f | worst ring field %.4f waves -> %s\n', ...
        rc.wfe_axial, rc.wfe_worst, ...
        ternary(rc.wfe_worst < DIFFRACTION_LIMIT,'DIFFRACTION-LIMITED','residual'));

%% -- [2] per-field aberration ladder (what limits the field?) --------
dc = wfe_field_diag(tc, optF);
fprintf(['[2] ladder: the centered family''s residual survives the astig fit\n' ...
         '    only at high order -- symmetric anastigmat behaviour.\n']);

%% -- [3] the obscuration price ---------------------------------------
scan = tc.realize_apertures('fields',[0 0; optF],'margin',0.05,'quiet',true);
rep  = tc.check_clipping('noload',true,'quiet',true);
nblk = sum(~[rep.ok]);
fprintf('[3] clearance: %d/%d optics clear -- the CENTERED price is the\n', ...
        sum([rep.ok]), numel(rep));
fprintf('    central obscuration (M2 shadows the pupil).\n');

%% -- [4] the A/B: the SECTION arm of the same recipe -----------------
[ts, rs] = tma_conic_recipe('D',D,'R',R,'spacings',TBET,'lambda',LAM, ...
                            'section',true,'fields',optF);
ds = wfe_field_diag(ts, optF, 'quiet',true);
fprintf('\n[4] SECTION (unobscured, decenter %.2f D): worst ring field %.4f waves\n', ...
        rs.decenter/D, rs.wfe_worst);

fprintf('\n================== A/B: %g''-dia circular field @ %g um ==================\n', ...
        2*FIELD_RAD, LAM*1e6);
fprintf('                       worst    -tilt   -focus  -astig    (waves)\n');
fprintf(' CENTERED (obscured)  %7.4f  %7.4f %7.4f %7.4f\n', rc.wfe_worst, ...
        max(dc.rms_tilt), max(dc.rms_focus), max(dc.rms_astig));
fprintf(' SECTION (unobscured) %7.4f  %7.4f %7.4f %7.4f\n', rs.wfe_worst, ...
        max(ds.rms_tilt), max(ds.rms_focus), max(ds.rms_astig));
fprintf([' Read: the section''s wavefront collapses once ASTIG is removed --\n' ...
         ' the unobscured cut buys its clear pupil with induced field\n' ...
         ' astigmatism.  j18''s remaining edge is its 2.3 um yardstick\n' ...
         ' (divide the waves by 2.3).\n']);
fprintf('==========================================================================\n');

%% -- [5] deliverable: the centered design ----------------------------
tc.add_pupil(numel(tc.spec.elt));
rxfile = fullfile(exdir,'tma_centered.in');  matfile = fullfile(exdir,'tma_centered.mat');
tc.save(rxfile);  tc.save_spec(matfile);
fprintf('[5] saved: %s\n           + %s\n', rxfile, matfile);

try
    f1 = tc.view_field_map(scan,'kind','contour');
    saveas(f1, fullfile(exdir,'tma_centered_wfe.png'));
    f2 = tc.view_orthoviews({'YZ','XZ'},'nrays',9);
    saveas(f2, fullfile(exdir,'tma_centered_layout.png'));
    fprintf('    figures: tma_centered_wfe.png + tma_centered_layout.png\n');
catch ME, fprintf('    figures skipped (%s)\n', ME.message); end

function s = ternary(c, a, b), if c, s = a; else, s = b; end, end
