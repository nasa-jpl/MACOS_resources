function lz = field_zone_lmon(t, elts, F, opts)
%FIELD_ZONE_LMON  Per-mirror field-zone Zernike normalization radii.
%   lz = field_zone_lmon(t, elts, F) measures, for each mirror in ELTS,
%   the FIELD-ZONE radius: the radius of the pooled ray-footprint cloud
%   over the field set F -- the beam footprint PLUS the chief-ray field
%   walk.  This is the lMon the sphere+Zernike solve doctrine fixes ONCE
%   before the first freeform solve:
%     * body-radius lMon  -> ill-conditioned metre-scale canceling
%       coefficient pairs (the shipped-3M pathology);
%     * center-footprint lMon -> stalls FIELD solves (the solved basis
%       is not normalized over where the off-axis beams land);
%     * field-zone lMon   -> the basis covers exactly the patch the
%       multi-field merit exercises.
%   Pass the SAME lz to every optimize_freeform stage -- coefficients
%   are only meaningful on the radius they were solved on.
%
%   t     macos.design.Telescope, built (its Rx loaded in-session).
%   elts  element indices of the mirrors to measure.
%   F     (nf x 2) [thx thy] field offsets in RADIANS, as
%         Telescope.trace_at_field takes.  The on-axis point is ALWAYS
%         included; pass zeros(0,2) for footprint-only.
%   Name-value: 'margin' (default 1.05) scales the measured radius.
%
%   Returns lz (1 x numel(elts)) in metres, ready for
%   optimize_freeform(..., 'lmon', lz).
%
%   See also: zern_jacobian_solve, design_report,
%   macos.design.Telescope/optimize_freeform.
    arguments
        t
        elts (1,:) double {mustBeInteger, mustBePositive}
        F (:,2) double = zeros(0,2)
        opts.margin (1,1) double {mustBePositive} = 1.05
    end
    fields = [0 0; F];
    P = cell(1, numel(elts));
    for i = 1:size(fields,1)
        if any(abs(fields(i,:)) > 1e-15), t.trace_at_field(fields(i,:));
        else,                             t.trace_at_field([]);        end
        for j = 1:numel(elts)
            s = macos.trace(elts(j));
            b = macos.get_ray_info(s.nRays);
            m = logical(b.ok_pass) & logical(b.ok_trace);
            P{j} = [P{j}, b.pos(:, m)];
        end
    end
    t.trace_at_field([]);                        % restore the nominal field
    lz = zeros(1, numel(elts));
    for j = 1:numel(elts)
        if isempty(P{j})
            error('macos:design:field_zone_lmon:norays', ...
                'no surviving rays at elt %d over the field set', elts(j));
        end
        c = mean(P{j}, 2);
        lz(j) = opts.margin * max(vecnorm(P{j} - c));
    end
end
