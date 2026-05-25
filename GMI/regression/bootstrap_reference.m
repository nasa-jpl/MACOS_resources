% bootstrap_reference.m -- one-shot to (re)generate the committed
% reference .mat files in ./reference/.
%
% Runs the same setup as each test and saves the OPDnom / dOPD
% arrays as the ground-truth references.  Use ONLY after an
% intentional behavior change; inspect the resulting diff before
% committing.  Invoked via:
%
%   ./run_regression.sh --bootstrap

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'tests'));
addpath(fullfile(here, 'lib'));
addpath(fileparts(here));

% Same Rx copy logic as regression_main.m.
optiix_local = fullfile(here, 'Rx', 'optiixonaxisz1_v4_pmsm_met.in');
optiix_canon = fullfile(fileparts(here), 'optiixonaxisz1_v4_pmsm_met.in');
if ~isfile(optiix_local) && isfile(optiix_canon)
    copyfile(optiix_canon, optiix_local);
end
cd(fullfile(here, 'Rx'));

ref_dir = fullfile(here, 'reference');
if ~isfolder(ref_dir), mkdir(ref_dir); end

% --- Optiix: nominal + Z4 response on Elt 4 ---
fprintf('\n[bootstrap] Optiix nominal\n');
[p, prb, pz, pg, IZ, IG] = init_optiix();
clear mex;
[~, ~, OPDnom] = call_GMI(prb, pz, pg, 0,0,0, p.pimg, IZ, IG, p);
save(fullfile(ref_dir, 'nominal_optiix.mat'), 'OPDnom');
fprintf('  -> reference/nominal_optiix.mat  (OPDnom RMS=%.3e)\n', ...
        sqrt(mean(OPDnom(OPDnom~=0).^2)));

% Optiix Z4 response: temporarily skipped -- the Zernike-channel
% apply path SIGSEGVs the MATLAB process on the release-candidate
% build.  Re-enable when the mex is fixed.
if false
    fprintf('[bootstrap] Optiix Z4 response on Elt 4\n');
    p.zernSrf = [4];
    pz_loc = zeros(length(p.zernSrf) * p.mzern, 1);
    pz_loc(1) = 1.0d-8;
    clear mex;
    [~, ~, OPDnom2] = call_GMI(prb, zeros(size(pz_loc)), pg, 0,0,0, p.pimg, IZ, IG, p);
    [~, ~, OPDpert] = call_GMI(prb, pz_loc, pg, 0,0,0, p.pimg, IZ, IG, p);
    dOPD = OPDpert - OPDnom2;
    save(fullfile(ref_dir, 'zern_response_optiix.mat'), 'dOPD');
    fprintf('  -> reference/zern_response_optiix.mat  (max|dOPD|=%.3e)\n', ...
            max(abs(dOPD(:))));
end

% --- e5hex1: nominal only (Z4 response disabled, see above) ---
fprintf('\n[bootstrap] e5hex1 nominal\n');
[p, prb, pz, pg, IZ, IG] = init_e5hex1();
clear mex;
[~, ~, OPDnom] = call_GMI(prb, pz, pg, 0,0,0, p.pimg, IZ, IG, p);
save(fullfile(ref_dir, 'nominal_e5hex1.mat'), 'OPDnom');
fprintf('  -> reference/nominal_e5hex1.mat  (OPDnom RMS=%.3e)\n', ...
        sqrt(mean(OPDnom(OPDnom~=0).^2)));

if false
    fprintf('[bootstrap] e5hex1 Z4 response on Elt 8\n');
    p.zernSrf = [8];
    pz_loc = zeros(length(p.zernSrf) * p.mzern, 1);
    pz_loc(1) = 1.0d-8;
    clear mex;
    [~, ~, OPDnom2] = call_GMI(prb, zeros(size(pz_loc)), pg, 0,0,0, p.pimg, IZ, IG, p);
    [~, ~, OPDpert] = call_GMI(prb, pz_loc, pg, 0,0,0, p.pimg, IZ, IG, p);
    dOPD = OPDpert - OPDnom2;
    save(fullfile(ref_dir, 'zern_response_e5hex1.mat'), 'dOPD');
    fprintf('  -> reference/zern_response_e5hex1.mat  (max|dOPD|=%.3e)\n', ...
            max(abs(dOPD(:))));
end

fprintf('\n[bootstrap] done.  Inspect reference/*.mat, then commit.\n');
quit force;
