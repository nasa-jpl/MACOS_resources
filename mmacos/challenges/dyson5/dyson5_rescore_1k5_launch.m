function T = dyson5_rescore_1k5_launch()
%DYSON5_RESCORE_1K5_LAUNCH  The 1.5k module of record re-scored end to end under BOTH t5f launches (CC 2026-10-08):
%   telescope dyson5_tA_GM_1k5_bAs.in on the 130 mm silica Dyson of record (size:D:130, the record's own join template
%   dyson5_t5e_tA_EP_pz_m30_B1_e2e.in, the 1500-px strip), 'chief' (the record's point-source launch) and 'centroid' (the
%   slit-filled proxy), roll 0 and 180.  Writes dyson5_rescore_1k5_launch.txt / .mat.
here = fileparts(mfilename('fullpath'));  addpath(fullfile(here, '..', '..', 'design', 'src'));
old = cd(here);  back = onCleanup(@() cd(old));
T = struct('launch', {}, 'roll', {}, 'smile', {}, 'keystone', {}, 'CRF', {}, 'SRF', {}, 'EE', {}, 'admits', {});
for lm = {'centroid', 'chief'}
    for r = [0 180]
        Pd = dyson5_params(struct('tel_dyson', 'size:D:130', 'tel_npix_xt', 1500, 'tel5e_roll_deg', r, 'tel5f_launch', lm{1}, ...
             'tel5f_deck', 'dyson5_tA_GM_1k5_bAs.in', 'tel5f_e2e_template', 'dyson5_t5e_tA_EP_pz_m30_B1_e2e.in', ...
             'tel5f_suffix', sprintf('_1k5rec_%s_roll%03d', lm{1}, r)));
        S5 = dyson5_t5f(Pd, 'dyson5');  R = S5.e2e;
        T(end+1) = struct('launch', lm{1}, 'roll', r, 'smile', R.smile_max, 'keystone', R.keystone_max, 'CRF', R.crf_max, ...
                          'SRF', R.srf_max, 'EE', R.ee_min, 'admits', min(R.pass_frac(:)));   %#ok<AGROW>
    end
end
fid = fopen('dyson5_rescore_1k5_launch.txt', 'w');
fprintf(fid, 'dyson5 1.5k module of record end to end, both launches (%s)\n', datestr(now, 'yyyy-mm-dd HH:MM'));
fprintf(fid, '  telescope dyson5_tA_GM_1k5_bAs.in, Dyson size:D:130 (silica 130, 27 mm slit), 1500 px strip\n');
fprintf(fid, '  %-9s %5s | %8s %9s %7s %7s %6s %7s\n', 'launch', 'roll', 'smile', 'keystone', 'CRF', 'SRF', 'EE', 'admits');
for t = T, fprintf(fid, '  %-9s %5d | %8.4f %9.4f %7.3f %7.4f %6.3f %7.3f\n', t.launch, t.roll, t.smile, t.keystone, t.CRF, t.SRF, t.EE, t.admits); end
fclose(fid);  type('dyson5_rescore_1k5_launch.txt');
save('dyson5_rescore_1k5_launch.mat', 'T');
end
