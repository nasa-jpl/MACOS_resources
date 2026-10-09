function vh_cass_probe(outfile)
%VH_CASS_PROBE  Does the Phase-2c "151x" follow from the engine's own
%   reflection coefficients?
%
%   The external anchor (vh_diag section 1) shows the coated-branch
%   machinery is exact: the engine reproduces a published protected-metal
%   model to ~1e-14 in diattenuation and 0.0000 deg in retardance, over
%   2-70 deg and three stack types.  That makes the 2c coating ladder
%   checkable INTERNALLY -- the per-mirror Fresnel coefficients are now
%   anchored, so the cross-polarized power they imply can be predicted and
%   compared with what macos.pol_contrast_floor reported.
%
%   PREDICTION.  For an on-axis rotationally symmetric train the mirrors'
%   s/p axes share one azimuth, so their Jones matrices commute into
%
%       J(rho,phi) = R(phi) diag(A,B) R(-phi),   A = prod(r_s), B = prod(r_p)
%                  = c0 * I  +  c2 * [cos2phi  sin2phi ;
%                                     sin2phi -cos2phi]
%       c0 = (A+B)/2,  c2 = (A-B)/2,  eps_tot = c2/c0
%
%   The cross-polarized amplitude is set entirely by eps_tot, so the cross
%   POWER ratio between two coatings is |eps_tot|^2 between them.  c0 and
%   c2 are read straight off the measured Jones pupil:
%       c0 = (J11+J22)/2,   c2 = (J11-J22)/2 (=c2 cos2phi) and J12 (=c2 sin2phi)
%   No external number enters the prediction -- it is the engine checked
%   against itself, with the anchor supplying the licence to trust the
%   coefficients.

    if nargin < 1, outfile = fullfile(tempdir, 'vh_cass.txt'); end
    macos.init(256);
    fid = fopen(outfile, 'w');  cf = onCleanup(@() fclose(fid));

    % rx_fixture_path lives in tests/private and is not reachable from a
    % tool directory, so mirror its search order here: the shared corpus
    % under pymacos first, the mmacos-local Rx/ second.
    here  = fileparts(mfilename('fullpath'));
    roots = {fullfile(getenv('HOME'), 'dev', 'MACOS_resources', 'pymacos', 'tests', 'Rx'), ...
             fullfile(here, '..', '..', 'tests', 'Rx')};
    rx = '';
    for i = 1:numel(roots)
        c = fullfile(roots{i}, 'Rx_Cass_FarField.in');
        if exist(c, 'file'), rx = c; break; end
    end
    assert(~isempty(rx), 'Rx_Cass_FarField.in not found');
    Prim = 2;  Sec = 3;  Pup = 5;  Det = 6;
    nAl  = 1.45;  kAl = 7.54;  thkAl = 2.0e-7;    % BaseUnits = m here
    nMg  = 1.38;  thkMg = 1.1e-7;

    % ---- 1. NOT MEASURED HERE: the per-element incidence angle ----------
    % Two routes were tried and both are unusable through this binding, so
    % no AOI number is reported rather than a wrong one:
    %   (a) ray_field's .nx/.ny/.nz is the ELEMENT AXIS normal broadcast to
    %       the grid (its own header says so), which on a CURVED mirror is
    %       not the local surface normal.  Tell: the primary and the
    %       secondary come back with IDENTICAL angle statistics.
    %   (b) the deviation between successive elements' ray directions,
    %       AOI = (180 - angle(k_in,k_out))/2, returns exactly 90 deg
    %       everywhere -- i.e. ray_field's direction cosines are not
    %       per-element in the way that construction needs.
    % Fortunately nothing below depends on it: the cross-power ratio this
    % probe is about is flat over the whole plausible near-normal range
    % (6.34 at 1.3 deg, 6.35 at 3.7 deg, 6.36 at 5.1 deg from the
    % independent analytic), and sections 2 and 3 measure the engine
    % directly.  Getting a trustworthy per-element AOI needs a route this
    % binding does not currently expose.
    macos.load_rx(rx);

    % ---- 2. eps_tot and the cross fraction from the Jones pupil ----------
    cfgs = {'uncoated (PEC)', []; ...
            'bare Al 200nm',  {nAl, kAl, thkAl}; ...
            'MgF2/Al',        {[nMg nAl], [0 kAl], [thkMg thkAl]}};

    fprintf(fid, '=== 2. eps_tot and cross fraction from macos.jones_pupil ===\n');
    fprintf(fid, '%18s %14s %14s %16s %14s\n', ...
            'config', 'mean|eps_tot|', 'cross frac', 'cross/bareAl', 'pred power');
    cf_ = nan(1,3);  ep_ = nan(1,3);
    for i = 1:3
        macos.load_rx(rx);
        if ~isempty(cfgs{i,2})
            v = cfgs{i,2};
            macos.coating(Prim, 'index', v{1}, 'extinc', v{2}, 'thickness', v{3});
            macos.coating(Sec,  'index', v{1}, 'extinc', v{2}, 'thickness', v{3});
        end
        macos.polarization('on', 'Ex', [1 0], 'Ey', [0 0]);
        jp = macos.jones_pupil(Det);
        m  = jp.mask;
        J  = jp.J;
        J11 = J(:,:,1,1); J12 = J(:,:,1,2);
        J21 = J(:,:,2,1); J22 = J(:,:,2,2);
        J11 = J11(m); J12 = J12(m); J21 = J21(m); J22 = J22(m);

        c0 = (J11 + J22)/2;
        c2 = hypot(abs((J11 - J22)/2), abs((J12 + J21)/2));
        ep_(i) = mean(abs(c2 ./ c0));

        % cross fraction relative to the pupil-MEAN output state, x input
        outx = J11;  outy = J21;
        vx = mean(outx);  vy = mean(outy);
        nv = hypot(abs(vx), abs(vy));
        co = (conj(vx)*outx + conj(vy)*outy)/nv;
        tot = abs(outx).^2 + abs(outy).^2;
        cf_(i) = sum(tot - abs(co).^2) / sum(tot);

        fprintf(fid, '%18s %14.6e %14.6e %16.4f %14.4f\n', cfgs{i,1}, ...
                ep_(i), cf_(i), cf_(i)/cf_(2), (ep_(i)/ep_(2))^2);
    end
    fprintf(fid, ['\n  "pred power" = (|eps_tot|/|eps_tot,bareAl|)^2 = the cross-power\n' ...
                  '  ratio the engine''s OWN coefficients imply.\n\n']);

    % ---- 3. what pol_contrast_floor reports ------------------------------
    fprintf(fid, '=== 3. macos.pol_contrast_floor sweep (the 2c ladder) ===\n');
    macos.load_rx(rx);
    mir = [Prim Sec];
    al   = struct('elt', num2cell(mir), 'index', nAl, 'extinc', kAl, ...
                  'thickness', thkAl, 'label', 'bare Al');
    mg   = struct('elt', num2cell(mir), 'index', [nMg nAl], 'extinc', [0 kAl], ...
                  'thickness', [thkMg thkAl], 'label', 'MgF2 / Al');
    o = macos.pol_contrast_floor(Pup, Det, 'input', 'x', ...
                                 'dark_zone', [10 40], 'coatings', {al, mg});
    fprintf(fid, '  uncoated dark-zone cross mean = %.6e\n', ...
            o.floor.dark_zone.cross.mean);
    for i = 1:numel(o.sweep)
        fprintf(fid, '  %-12s d_cross_rel = %10.4f   dark-zone cross mean = %.6e\n', ...
                o.sweep(i).label, o.sweep(i).d_cross_rel, ...
                o.sweep(i).floor.dark_zone.cross.mean);
    end
    fprintf(fid, '\n  measured MgF2/bareAl cross-power ratio = %.4f\n', ...
            o.sweep(2).d_cross_rel / o.sweep(1).d_cross_rel);
    fprintf(fid, '  predicted from the Jones pupil          = %.4f\n', ...
            (ep_(3)/ep_(2))^2);

    fprintf('vh_cass_probe: wrote %s\n', outfile);
end
