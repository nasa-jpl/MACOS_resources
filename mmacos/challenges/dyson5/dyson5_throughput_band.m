function S = dyson5_throughput_band(opts)
%DYSON5_THROUGHPUT_BAND  Throughput of the Dyson's air-glass crossings per band, and the grating's scalar blaze.
%   S = dyson5_throughput_band()   both modules (3k CaF2 240 mm, 1.5k silica 130 mm); record dyson5_throughput_band.txt
%
%   BRIEF_to_dyson5 addendum 49 step 3 (Jim 2026-10-07: "the systems are photon starved at the long wavelength end;
%   they tailor the grating efficiency to compensate; you want the AR to help where you need the photons the most").
%   Per band (380-700, 700-1300, 1300-2500 nm) the mean over wavelength of the product of the crossings' power
%   transmittances (macos.design.thinfilm_rt, Abeles, normal incidence), the block's index from its Sellmeier
%   equation (Malitson: fused silica 1965, CaF2 1963) so the uncoated and the coated cases both carry the band's
%   dispersion; the coating materials held at their 1 um indices (MgF2 1.384, Al2O3 1.63; stated).
%   The crossings: the slit face (air -> block), the block's convex face out to the grating's air gap and back in,
%   the block's exit face (-> air) = four on the block; the detector's order-sorting filter (fused silica, 2 mm, in
%   air: Jim, it cannot be bonded to CaF2) = two more.  Six, as the record's route 1 counts them.
%   Coatings: none; a single quarter-wave MgF2 centred at 2.2 um (Jim's "help where the photons are"); a two-layer
%   quarter-quarter MgF2 / Al2O3 centred at 2.2 um; the same two-layer centred at 1.0 um (the record's route 2).
%   The grating: the SCALAR blaze estimate eta(lambda) = sinc^2(lambda_B/lambda - 1) in first order, for blaze
%   wavelengths 1.0, 1.4 and 1.8 um -- an estimate, not a vector calculation; it says where the photons go, not
%   the absolute efficiency at the short end.  No detector QE, no slit diffraction loss (the record's 0.3-1.8 %).
    arguments
        opts.bands (:,2) double = [380 700; 700 1300; 1300 2500]      % nm
        opts.nlam (1,1) double = 61                                    % samples per band
        opts.lam0_ar_um (1,:) double = [2.2 1.0]                       % AR design wavelengths
        opts.blaze_um (1,:) double = [1.0 1.4 1.8]
        opts.n_mgf2 (1,1) double = 1.384
        opts.n_al2o3 (1,1) double = 1.63
        opts.tag (1,:) char = 'dyson5_throughput_band'
        opts.quiet (1,1) logical = false
    end
    here = fileparts(mfilename('fullpath'));
    mods = struct('name', {'3k, CaF2 240 mm', '1.5k, silica 130 mm'}, 'block', {@n_caf2, @n_silica});
    nB = size(opts.bands, 1);
    cases = {'uncoated', sprintf('MgF2 quarter wave at %.1f um', opts.lam0_ar_um(1)), ...
             sprintf('MgF2 / Al2O3 quarter-quarter at %.1f um', opts.lam0_ar_um(1)), ...
             sprintf('MgF2 / Al2O3 quarter-quarter at %.1f um', opts.lam0_ar_um(end))};
    T = nan(numel(mods), numel(cases), nB);  Tfull = nan(numel(mods), numel(cases));
    lam_all = [];  Tl = cell(numel(mods), numel(cases));
    for im = 1:numel(mods)
        for ic = 1:numel(cases)
            tl = [];  ll = [];
            for ib = 1:nB
                lam = linspace(opts.bands(ib, 1), opts.bands(ib, 2), opts.nlam)*1e-3;   % um
                t = ones(size(lam));
                for k = 1:numel(lam)
                    nb = mods(im).block(lam(k));  nw = n_silica(lam(k));
                    % four block crossings (air<->block) + two filter crossings (air<->silica), each AR'd alike
                    t(k) = crossing_(ic, lam(k), nb, opts)^4 * crossing_(ic, lam(k), nw, opts)^2;
                end
                T(im, ic, ib) = mean(t);  tl = [tl t];  ll = [ll lam]; %#ok<AGROW>
            end
            Tfull(im, ic) = mean(tl);  Tl{im, ic} = tl;  lam_all = ll;
        end
    end
    % the grating's scalar blaze, per band
    B = nan(numel(opts.blaze_um), nB);
    for jb = 1:numel(opts.blaze_um)
        for ib = 1:nB
            lam = linspace(opts.bands(ib, 1), opts.bands(ib, 2), opts.nlam)*1e-3;
            x = opts.blaze_um(jb)./lam - 1;  eta = (sin(pi*x)./(pi*x)).^2;  eta(x == 0) = 1;
            B(jb, ib) = mean(eta);
        end
    end
    S = struct('bands_nm', opts.bands, 'cases', {cases}, 'modules', {{mods.name}}, 'T', T, 'T_band_mean', Tfull, ...
               'blaze_um', opts.blaze_um, 'blaze_eta', B, 'lam_um', lam_all, 'T_lambda', {Tl});
    % ---- record
    out = fullfile(here, [opts.tag '.txt']);  fid = fopen(out, 'w');
    pr = @(varargin) fprintf(fid, varargin{:});
    pr('dyson5 throughput by band (%s) -- addendum 49 step 3\n', datestr(now, 'yyyy-mm-dd HH:MM'));
    pr('CONVENTIONS: six air-glass crossings per module (four on the block: slit face in, convex face out to the grating gap and back in, exit face;\n');
    pr('  two on the detector''s order-sorting filter, fused silica in air); power transmittance per crossing by macos.design.thinfilm_rt (Abeles, normal\n');
    pr('  incidence); block index by Sellmeier (Malitson: silica 1965, CaF2 1963) at every wavelength; coating indices held at MgF2 %.3f, Al2O3 %.2f;\n', opts.n_mgf2, opts.n_al2o3);
    pr('  each entry = the mean over %d wavelengths in the band of the product of the six transmittances.  No detector QE, no slit diffraction loss.\n', opts.nlam);
    pr('  Grating: SCALAR blaze estimate eta = sinc^2(lambda_B/lambda - 1), first order -- where the photons go, not an absolute efficiency.\n\n');
    for im = 1:numel(mods)
        pr('MODULE %s\n', mods(im).name);
        pr('  %-44s', 'coating');  for ib = 1:nB, pr('  %4d-%4d nm', opts.bands(ib, :)); end;  pr('   380-2500 mean\n');
        for ic = 1:numel(cases)
            pr('  %-44s', cases{ic});  for ib = 1:nB, pr('  %12.3f', T(im, ic, ib)); end;  pr('   %.3f\n', Tfull(im, ic));
        end
        pr('\n');
    end
    pr('GRATING, scalar blaze estimate (first order)\n');
    pr('  %-44s', 'blaze wavelength');  for ib = 1:nB, pr('  %4d-%4d nm', opts.bands(ib, :)); end;  pr('\n');
    for jb = 1:numel(opts.blaze_um)
        pr('  %-44s', sprintf('%.1f um', opts.blaze_um(jb)));  for ib = 1:nB, pr('  %12.3f', B(jb, ib)); end;  pr('\n');
    end
    fclose(fid);
    save(fullfile(here, [opts.tag '.mat']), 'S');
    if ~opts.quiet, type(out); end
end

function t = crossing_(ic, lam_um, ns, opts)
%CROSSING_  power transmittance of one air -> substrate crossing (the reverse is the same by reciprocity at normal incidence)
    switch ic
        case 1, layers = zeros(0, 2);
        case 2, layers = [opts.n_mgf2, opts.lam0_ar_um(1)/(4*opts.n_mgf2)];
        case 3, layers = [opts.n_mgf2, opts.lam0_ar_um(1)/(4*opts.n_mgf2); opts.n_al2o3, opts.lam0_ar_um(1)/(4*opts.n_al2o3)];
        case 4, layers = [opts.n_mgf2, opts.lam0_ar_um(end)/(4*opts.n_mgf2); opts.n_al2o3, opts.lam0_ar_um(end)/(4*opts.n_al2o3)];
    end
    o = macos.design.thinfilm_rt(layers, 1.0, ns, 0, lam_um);
    t = real(o.Ts);
end

function n = n_silica(lam)
% Malitson 1965, fused silica, lam in um
    L2 = lam.^2;
    n = sqrt(1 + 0.6961663*L2./(L2 - 0.0684043^2) + 0.4079426*L2./(L2 - 0.1162414^2) + 0.8974794*L2./(L2 - 9.896161^2));
end

function n = n_caf2(lam)
% Malitson 1963, CaF2, lam in um
    L2 = lam.^2;
    n = sqrt(1 + 0.5675888*L2./(L2 - 0.050263605^2) + 0.4710914*L2./(L2 - 0.1003909^2) + 3.8484723*L2./(L2 - 34.649040^2));
end
