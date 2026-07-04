function [t, res] = tma_conic_recipe(opts)
%TMA_CONIC_RECIPE  The staged conic-TMA design recipe, both families.
%   [t,res] = tma_conic_recipe('section',false, ...) builds and optimizes
%   a convex-secondary Korsch TMA (j18mono geometry by default) through
%   the validated stage sequence, for EITHER basic design family:
%
%     CENTERED  ('section',false)  the classical on-axis anastigmat --
%       rotational symmetry intact, so three conics null spherical, coma
%       AND astigmatism about the axis and the residual field astig is
%       high-order (the j18 / early-JWST configuration).  Price: the
%       central obscuration (M2 in the beam).
%
%     SECTION   ('section',true)   the unobscured eccentric-pupil
%       off-axis section of the same parent.  Price: the section breaks
%       rotational symmetry and induces FIELD-DEPENDENT (binodal)
%       astigmatism the balance can only partially trade -- the field
%       wall arrives sooner (diagnose with wfe_field_diag).
%
%   Stages: build -> on-axis conic optimize -> [set_offaxis('all') +
%   axial radius+conic refigure]* -> multi-field balance (tip/tilt/dy +
%   radius+conic) over the given field set.        (* section only)
%
%   Name-value (defaults = the j18mono-geometry 6.6 m f/20):
%     'D'          aperture diameter, m           (6.605)
%     'R'          |radii| [M1 M2 M3], m          (j18mono)
%     'spacings'   [M1->M2, M2->M3], m            (j18mono)
%     'lambda'     wavelength, m                  (1e-6)
%     'model'      MACOS model size               (256)
%     'grid_npts'  source grid                    (41)
%     'section'    unobscured off-axis section    (false)
%     'fields'     Kx2 (thx,thy) OFF-axis balance set, rad -- e.g.
%                  macos.design.field_ring(2.5,'units','arcmin') for a
%                  5'-diameter circular field    (ring at 1.0')
%     'max_iters'  CALIB cap per optimize         (150)
%
%   Returns the Telescope t (optimized, ready for wfe_field_diag /
%   realize_apertures / add_pupil / save) and res:
%     .wfe_axial   on-axis RMS WFE after the axial stages (waves)
%     .wfe_worst   worst balanced field (waves)
%     .wfe_fields  per-field balanced WFE (waves)
%     .decenter    section decenter, m (0 for centered)
%     .conics      final [K1 K2 K3]
%
%   See also wfe_field_diag, macos.design.field_ring,
%   macos.design.Telescope.
    arguments
        opts.D         (1,1) double = 6.605
        opts.R         (1,3) double = [15.879722 1.778913 3.016227]
        opts.spacings  (1,2) double = [7.169041556 7.965313479]
        opts.lambda    (1,1) double = 1.0e-6
        opts.model     (1,1) double = 256
        opts.grid_npts (1,1) double = 41
        opts.section   (1,1) logical = false
        opts.fields    (:,2) double = macos.design.field_ring(1.0,'units','arcmin')
        opts.max_iters (1,1) double = 150
    end
    lam = opts.lambda;
    t = macos.design.Telescope('family','TMA','aperture_diameter_m',opts.D, ...
            'model_size',opts.model,'wavelength_m',lam, ...
            'grid_npts',opts.grid_npts);
    t.add_mirror('M1','radius_m',opts.R(1),'spacing_after_m',opts.spacings(1));
    t.add_mirror('M2','radius_m',opts.R(2),'spacing_after_m',opts.spacings(2), ...
                 'convex',true);
    t.add_mirror('M3','radius_m',opts.R(3),'spacing_after','derive');
    t.add_focal_plane('FP');
    t.build();

    % on-axis conic optimize (small field keeps the solve well-posed)
    t.optimize('fields_arcmin',[0.5 1.0],'dofs',[0 0 0 0 0 0 0 1], ...
               'max_iters',opts.max_iters);

    dy = 0;
    if opts.section
        dy = t.set_offaxis('all');
        t.optimize('fields_arcmin',[],'dofs',[0 0 0 0 0 0 1 1], ...
                   'max_iters',opts.max_iters);
        t.set_offaxis('none');
    end
    nE = numel(t.spec.elt);
    macos.trace(nE);
    W = macos.opd();  v = W(isfinite(W) & W ~= 0);
    wfe_ax = std(v)/lam;

    % multi-field balance over the given field set
    rb = t.optimize('fields',opts.fields,'dofs',[1 1 0 0 1 0 1 1], ...
                    'max_iters',opts.max_iters);
    if opts.section, t.set_offaxis('none'); end

    res = struct('wfe_axial',wfe_ax, ...
                 'wfe_worst',max(rb.wfe_after)/lam, ...
                 'wfe_fields',rb.wfe_after/lam, ...
                 'decenter',dy, ...
                 'conics',[t.spec.elt(1).Kc t.spec.elt(2).Kc t.spec.elt(3).Kc]);
end
