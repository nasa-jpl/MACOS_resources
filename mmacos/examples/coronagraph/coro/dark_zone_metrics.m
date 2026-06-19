function m = dark_zone_metrics(I, peak_unaberrated, lam_over_D_px, ...
                               inner_lamD, outer_lamD, region)
%DARK_ZONE_METRICS  Per-pixel dark-zone contrast statistics over a region.
%   M = DARK_ZONE_METRICS(I, PEAK_UNAB, LAMD_PX, INNER, OUTER) returns a
%   struct of scoring metrics over the annular dark zone
%   [INNER, OUTER] lambda/D of a focal-plane intensity I, each
%   Strehl-normalised to PEAK_UNAB (the no-mask on-axis peak):
%
%     m.mean   - mean per-pixel contrast in the region (standard
%                "dark-zone contrast")
%     m.peak   - max  (brightest pixel = worst contrast in the zone)
%     m.floor  - min  (darkest pixel  = best contrast achieved)
%     m.median - median per-pixel contrast
%     m.energy - sum of per-pixel contrast over the region (total
%                residual starlight energy, in contrast units)
%     m.n_pix  - number of pixels in the region
%
%   These are the SELECTABLE coronagraph optimization objectives: an
%   outer/inner loop minimises one of m.mean / m.peak / m.energy / ...
%   depending on the design goal (uniform dark hole vs worst-case
%   suppression vs total leaked flux).  dm_merit() picks one by name.
%
%   REGION (name-value, optional) selects the dark-zone GEOMETRY, which
%   is set by the control architecture (NOT a free choice):
%     'annulus'  (default)  full 360-deg annulus  -- needs 2 DMs
%     'right'/'left'/'top'/'bottom'  one-sided (half-plane) D-shape
%                           -- the fair region for a 1-DM system, which
%                              digs DEEPER contrast over a SMALLER area
%     'sector', [a0 a1]     angular sector in degrees (CCW from +x)
%   e.g. DARK_ZONE_METRICS(I,p,ld,7,10,'side','right')
%        DARK_ZONE_METRICS(I,p,ld,7,10,'sector',[ -30 30 ])
%
%   Per-pixel stats (not azimuthally pre-averaged ring means) are the
%   fundamental quantity; radial_contrast.m is the azimuthal average of
%   the same field and is what plots/curves use.
    arguments
        I                (:,:) double
        peak_unaberrated (1,1) double
        lam_over_D_px    (1,1) double
        inner_lamD       (1,1) double
        outer_lamD       (1,1) double
        region.side   (1,:) char {mustBeMember(region.side, ...
            {'annulus','right','left','top','bottom'})} = 'annulus'
        region.sector (1,2) double = [-180 180]
    end

    [ny, nx] = size(I);
    cy = (ny - 1) / 2.0;   % 0-based array centre (matches radial_profile)
    cx = (nx - 1) / 2.0;
    [xx, yy] = meshgrid(0:nx-1, 0:ny-1);
    dx_ = xx - cx;  dy_ = yy - cy;
    r_lamD = hypot(dy_, dx_) / lam_over_D_px;

    ann = (r_lamD >= inner_lamD) & (r_lamD <= outer_lamD);

    % One-sided half-plane restriction (image row index increases
    % downward, so 'top' is dy_ < 0).
    switch region.side
        case 'right',  ann = ann & (dx_ >  0);
        case 'left',   ann = ann & (dx_ <  0);
        case 'top',    ann = ann & (dy_ <  0);
        case 'bottom', ann = ann & (dy_ >  0);
        % 'annulus' -> no half-plane cut
    end

    % Optional angular sector (degrees CCW from +x; +y is up, so use
    % -dy_ to put angle in the conventional math sense).
    if ~isequal(region.sector, [-180 180])
        ang = atan2d(-dy_, dx_);
        a0 = region.sector(1);  a1 = region.sector(2);
        ann = ann & (ang >= a0 & ang <= a1);
    end

    vals = I(ann) / peak_unaberrated;

    m = struct();
    if isempty(vals)
        [m.mean, m.peak, m.floor, m.median, m.energy] = deal(NaN);
        m.n_pix = 0;
        return
    end
    m.mean   = mean(vals);
    m.peak   = max(vals);
    m.floor  = min(vals);
    m.median = median(vals);
    m.energy = sum(vals);
    m.n_pix  = numel(vals);
end
