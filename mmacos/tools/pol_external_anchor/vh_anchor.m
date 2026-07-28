function out = vh_anchor(opts)
%VH_ANCHOR  External anchor for the protected-metal polarization machinery.
%
%   Run:  matlab -batch "mmacos_setup; addpath('tools/pol_external_anchor'); vh_anchor"
%
%   PURPOSE.  Everything the polarization work had reported for a COATED
%   mirror was model-relative: the thin-film recursion was gated against
%   our own analytics -- tJonesPupil's Fresnel gate (an optically thick
%   SINGLE layer, i.e. a bare interface) and tPolRadiometric (the Abeles
%   matrix in TRANSMISSION).  Neither exercises a real DIELECTRIC-ON-METAL
%   stack, which is what a protected mirror is and what the Phase-2c "151x"
%   claim rests on.  This harness drives the engine with a PUBLISHED
%   configuration -- their indices, their film thickness, their
%   wavelengths, their incidence angles -- and compares curve-on-curve
%   against the publication's own model.
%
%   TWO COMPARISONS, reported separately, because they answer different
%   questions:
%
%     (a) MACHINERY CHECK.  Engine vs publication, at the publication's own
%         inputs.  Any disagreement here is OURS.  Tolerance stated from
%         the publication's error bars (+-0.01 per normalized Mueller
%         element, their Sec. 2).
%
%     (b) CONTEXT CHECK.  Our 632.8 nm / 110 nm-MgF2 configuration against
%         the nearest published configurations.  NEVER a gate: the nearest
%         published protected-Al work uses a different overcoat at
%         different wavelengths, so a numerical difference is expected and
%         is not evidence about the engine.
%
%   See README.md for the design and macos/REVIEW_POL_EXTERNAL_2026-07-28.md
%   for the findings.  Gate: tests/tPolExternal.m.

    arguments
        opts.aoi      (1,:) double = [6 20 30 45 60 70]
        opts.ngridpts (1,1) double = 41
        opts.model    (1,1) double = 128
        opts.outfile  (1,:) char   = ''
    end

    d = vh_data();
    macos.init(opts.model);

    work = [tempname '_vhanchor'];  mkdir(work);
    cl = onCleanup(@() rmdir(work, 's'));

    if isempty(opts.outfile)
        fid = 1;
    else
        fid = fopen(opts.outfile, 'w');
        cf = onCleanup(@() fclose(fid));   %#ok<NASGU>
    end

    fprintf(fid, 'External anchor: %s\n', d.source);
    fprintf(fid, 'captured %s   doi:%s\n\n', d.captured, d.doi);

    % ---------------------------------------------------------------------
    % (a) machinery check
    % ---------------------------------------------------------------------
    fprintf(fid, '=== (a) MACHINERY: engine vs publication, at their inputs ===\n');
    fprintf(fid, '%6s %6s %14s %14s %11s %11s\n', ...
            'lam', 'AOI', 'D_eng', 'D_ana', 'dD', 'dRet_M');

    nL = numel(d.lambda_nm);
    res = struct('lambda_nm', {}, 'aoi', {}, 'D_eng', {}, 'D_ana', {}, ...
                 'ret_eng', {}, 'ret_ana', {}, 'dev_D', {}, 'dev_ret_M', {});
    worstD = 0;  worstR = 0;

    for iL = 1:nL
        lam_nm = d.lambda_nm(iL);
        lam_mm = lam_nm * 1e-6;
        L = [complex(d.nf(iL),   0),           d.d_oxide_nm * 1e-6; ...
             complex(d.n_al(iL), -d.k_al(iL)), d.d_al_nm    * 1e-6];

        na = numel(opts.aoi);
        De_ = nan(1,na); Da_ = nan(1,na);
        Re_ = nan(1,na); Ra_ = nan(1,na);
        dD_ = nan(1,na); dM_ = nan(1,na);

        for iA = 1:na
            m = vh_measure(work, lam_mm, opts.aoi(iA), L, opts.ngridpts);

            % analytic at each ray's OWN incidence cosine
            [rp, rs] = vh_thinfilm(L, complex(1.52,0), m.cthi, lam_mm);
            Rp = abs(rp).^2;  Rs = abs(rs).^2;
            Da = (Rs - Rp)./(Rs + Rp);
            dla = angle(rp) - angle(rs);

            % engine.  The p-hat bridge is ZERO -- measured on the
            % perfect-conductor case (tPolExternal pins it), not assumed
            % from the ray-following doctrine, which would have added a
            % spurious pi.
            rho = m.rho;
            De  = (abs(rho).^2 - 1)./(abs(rho).^2 + 1);
            dle = -angle(rho);

            % retardance deviation in NORMALIZED MUELLER units: the lower
            % 2x2 block carries 2 sqrt(Rp Rs)/(Rp+Rs) * {cos,sin}
            amp = 2*sqrt(Rp.*Rs)./(Rp + Rs);

            De_(iA) = median(De);   Da_(iA) = median(Da);
            Re_(iA) = median(dle);  Ra_(iA) = median(dla);
            dD_(iA) = max(abs(De - Da));
            dM_(iA) = max(abs(amp .* wrap_(dle - dla)));

            fprintf(fid, '%6d %6.1f %14.6e %14.6e %11.2e %11.2e\n', ...
                    lam_nm, opts.aoi(iA), De_(iA), Da_(iA), dD_(iA), dM_(iA));
        end

        res(iL).lambda_nm = lam_nm;   res(iL).aoi     = opts.aoi;
        res(iL).D_eng     = De_;      res(iL).D_ana   = Da_;
        res(iL).ret_eng   = Re_;      res(iL).ret_ana = Ra_;
        res(iL).dev_D     = dD_;      res(iL).dev_ret_M = dM_;
        worstD = max(worstD, max(dD_));
        worstR = max(worstR, max(dM_));
    end

    out.machinery = res;
    out.worst_D   = worstD;
    out.worst_ret = worstR;
    out.accuracy  = d.mueller_accuracy;
    out.data      = d;

    fprintf(fid, '\n  worst over %d wavelengths x %d angles:\n', nL, numel(opts.aoi));
    fprintf(fid, '    diattenuation ([1,2] element) : %.3e\n', worstD);
    fprintf(fid, '    retardance block              : %.3e\n', worstR);
    fprintf(fid, '    publication accuracy          : %.3e\n', d.mueller_accuracy);
    fprintf(fid, '    inside the published bar by   : %.0fx\n\n', ...
            d.mueller_accuracy / max(worstD, worstR));

    % ---------------------------------------------------------------------
    % (b) context check
    % ---------------------------------------------------------------------
    fprintf(fid, '=== (b) CONTEXT: our MgF2/Al vs the nearest published stacks ===\n');
    lam_mm = 632.8e-6;
    nAl = 1.45;  kAl = 7.54;  dAl = 2.0e-4;
    stacks = { ...
        'bare Al 200nm',     [complex(nAl,-kAl), dAl]; ...
        'Al2O3 4.12nm / Al', [complex(1.60,0), d.d_oxide_nm*1e-6; complex(nAl,-kAl), dAl]; ...
        'MgF2 110nm / Al',   [complex(1.38,0), 1.1e-4;            complex(nAl,-kAl), dAl] };

    fprintf(fid, '%5s', 'AOI');
    for i = 1:size(stacks,1), fprintf(fid, ' %24s', stacks{i,1}); end
    fprintf(fid, '\n%5s', '');
    for i = 1:size(stacks,1), fprintf(fid, ' %10s %13s', 'D', '|eps|'); end
    fprintf(fid, '\n');

    ctx = struct('label', {}, 'aoi', {}, 'D', {}, 'ret', {}, 'eps', {});
    for i = 1:size(stacks,1)
        ctx(i).label = stacks{i,1};  ctx(i).aoi = opts.aoi;
        ctx(i).D = nan(1,numel(opts.aoi));
        ctx(i).ret = nan(1,numel(opts.aoi));
        ctx(i).eps = nan(1,numel(opts.aoi));
    end
    for j = 1:numel(opts.aoi)
        fprintf(fid, '%5.1f', opts.aoi(j));
        for i = 1:size(stacks,1)
            m = vh_measure(work, lam_mm, opts.aoi(j), stacks{i,2}, opts.ngridpts);
            r = m.rho;
            ctx(i).D(j)   = median((abs(r).^2 - 1)./(abs(r).^2 + 1));
            ctx(i).ret(j) = median(-angle(r));
            % eps = (r_s - r_p)/(r_s + r_p): the scalar that drives
            % cross-polarization in an on-axis rotationally symmetric train
            ctx(i).eps(j) = median(abs((r - 1)./(r + 1)));
            fprintf(fid, ' %10.3e %13.5e', ctx(i).D(j), ctx(i).eps(j));
        end
        fprintf(fid, '\n');
    end
    out.context = ctx;

    fprintf(fid, ['\n  |eps| is the cross-pol driver.  MgF2 REDUCES it at small AOI\n' ...
                  '  and costs more only above the crossover near 35 deg -- the\n' ...
                  '  direction the UV-coronagraph literature reports qualitatively.\n']);
end

function a = wrap_(a)
    a = mod(a + pi, 2*pi) - pi;
end
