function vh_diag(outfile)
%VH_DIAG  Side-by-side engine-vs-analytic diagnostic, written to a file.
%   Deliberately writes to a FILE: MATLAB's batch stdout is captured as a
%   rolling tail here, and the interesting rows scroll off.

    if nargin < 1
        outfile = fullfile(tempdir, 'vh_diag.txt');
    end
    d = vh_data();
    macos.init(128);

    work = [tempname '_vhdiag'];  mkdir(work);
    cl = onCleanup(@() rmdir(work, 's'));

    fid = fopen(outfile, 'w');
    cf  = onCleanup(@() fclose(fid));

    lam_nm = 632.8;  lam_mm = lam_nm*1e-6;
    nAl = 1.45;  kAl = 7.54;  dAl = 2.0e-4;

    % ---- 0. p-hat convention, MEASURED on a case with a known answer ----
    % The Bench emits perfect-conductor mirrors (IndRef=1, Extinc=1e22).
    % For a PEC the reflection is polarization-neutral, so in a FIXED
    % transverse frame r_s = r_p exactly and the ratio is +1; in the
    % RAY-FOLLOWING frame (p_r = s x k_out, which flips relative to p_i at
    % normal incidence) it is -1.  Whichever the engine reports here is the
    % frame our measured rho lives in, and therefore the bridge that must
    % be applied before comparing retardance with the publication.
    fprintf(fid, '=== 0. p-hat convention probe (uncoated = perfect conductor) ===\n');
    fprintf(fid, '%8s %14s %14s\n', 'AOI', 're(rho)', 'im(rho)');
    for th = [2 10 45 80]
        m = vh_measure(work, lam_mm, th, [], 41);
        r = median(real(m.rho)) + 1i*median(imag(m.rho));
        fprintf(fid, '%8.1f %14.6f %14.3e\n', th, real(r), imag(r));
    end
    fprintf(fid, ['  rho = -1 => ray-following p-hat (bridge = pi)\n' ...
                  '  rho = +1 => fixed transverse p-hat (bridge = 0)\n\n']);

    % ---- 1. engine vs analytic ------------------------------------------
    stacks = { ...
        'bare Al 200nm',     [complex(nAl,-kAl), dAl]; ...
        'Al2O3 4.12nm / Al', [complex(1.60,0), d.d_oxide_nm*1e-6; complex(nAl,-kAl), dAl]; ...
        'MgF2 110nm / Al',   [complex(1.38,0), 1.1e-4;            complex(nAl,-kAl), dAl] };

    aoi = [2 6 10 20 30 45 60 70];

    fprintf(fid, '=== 1. engine vs analytic, lambda = %.1f nm ===\n', lam_nm);
    fprintf(fid, 'analytic = vh_thinfilm (Macleod char. matrix), per-ray cos(theta)\n');
    fprintf(fid, 'retardance RAW on both sides (no bridge applied); the offset is the bridge\n\n');

    for i = 1:size(stacks,1)
        L = stacks{i,2};
        fprintf(fid, '--- %s ---\n', stacks{i,1});
        fprintf(fid, '%5s %14s %14s %11s | %12s %12s %11s | %10s %8s\n', ...
            'AOI', 'D_eng', 'D_ana', 'dD', ...
            'ret_eng_deg', 'ret_ana_deg', 'offset_deg', 'consist', 'aoi_act');
        for j = 1:numel(aoi)
            m = vh_measure(work, lam_mm, aoi(j), L, 41);
            [rp, rs] = vh_thinfilm(L, complex(1.52,0), m.cthi, lam_mm);
            Rp = abs(rp).^2;  Rs = abs(rs).^2;
            Da  = (Rs-Rp)./(Rs+Rp);
            dla = angle(rp) - angle(rs);

            rho = m.rho;
            De  = (abs(rho).^2 - 1)./(abs(rho).^2 + 1);
            dle = -angle(rho);                    % RAW, no bridge

            fprintf(fid, '%5.1f %14.6e %14.6e %11.2e | %12.6f %12.6f %11.4f | %10.2e %8.3f\n', ...
                aoi(j), median(De), median(Da), max(abs(De-Da)), ...
                rad2deg(median(dle)), rad2deg(median(dla)), ...
                rad2deg(median(wrap_(dle-dla))), m.consistency, ...
                rad2deg(acos(median(m.cthi))));
        end
        fprintf(fid, '\n');
    end

    % ---- 2. cross-pol driver --------------------------------------------
    % eps = (r_s - r_p)/(r_s + r_p) in the FIXED frame -- the scalar that
    % drives cross-polarization in an on-axis rotationally symmetric train.
    % Computed from the analytic (unambiguous frame) and, for the engine,
    % from rho with the measured bridge applied by section 0.
    fprintf(fid, '=== 2. cross-pol driver |eps| = |(rs-rp)/(rs+rp)|, fixed frame ===\n');
    fprintf(fid, '%5s', 'AOI');
    for i = 1:size(stacks,1), fprintf(fid, ' %26s', stacks{i,1}); end
    fprintf(fid, '\n');
    for j = 1:numel(aoi)
        fprintf(fid, '%5.1f', aoi(j));
        for i = 1:size(stacks,1)
            m = vh_measure(work, lam_mm, aoi(j), stacks{i,2}, 41);
            [rp, rs] = vh_thinfilm(stacks{i,2}, complex(1.52,0), m.cthi, lam_mm);
            fprintf(fid, ' %12.4e(a)', median(abs((rs-rp)./(rs+rp))));
            % engine, fixed frame: rho_fixed = rho * exp(i*bridge)
            rf = m.rho;
            fprintf(fid, '%12.4e(e)', median(abs((rf-1)./(rf+1))));
        end
        fprintf(fid, '\n');
    end

    fprintf('vh_diag: wrote %s\n', outfile);
end

function a = wrap_(a)
    a = mod(a + pi, 2*pi) - pi;
end
