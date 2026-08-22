function fp = optic_footprints(m, rx, opts)
%OPTIC_FOOTPRINTS  Trace all fields x all configs; per-optic footprint geometry.
%   fp = optic_footprints(M, RX) loads RX in session M and traces it over a
%   field set (and, optionally, a set of CONFIGURATIONS), recording for every
%   element the UNION ray footprint across all those traces.  It gives the
%   figure-frame geometry the prescription alone cannot (Dave 2026-08-21):
%   the vertex (VptElt) is NOT the optic centre in general, RptElt is only a
%   convention, and lMon is a real footprint radius, not a deck constant.  So
%   to place and size a figure channel on an optic (e.g. promoting SM/TM to
%   FreeForm), TRACE and measure:
%
%     * WHERE the beam sits  -- the footprint CENTROID (a figure-frame centre);
%     * HOW BIG it is        -- the footprint RADIUS (an lMon / grid span);
%
%   measured over EVERY field and configuration, because a field scan and a
%   zoom/compensation move both walk the beam across each optic, so the union
%   footprint is larger (and differently centred) than any single trace.
%
%   (For WHICH elements carry a meaningful sensitivity -- e.g. dropping an
%   obscured virtual reference -- do NOT threshold pass-fraction: on a
%   segmented aperture every segment catches only ~1/nseg of the full beam.
%   Use the zero-norm channel flag on the harvested Jacobian instead --
%   design/src/flag_zero_norm_channels.)
%
%   OPTIONS
%     'fields'   N x 2 [dx dy] field offsets (rad).  Default: the stock
%                centre + 4 corners at 'fov_rad'.
%     'fov_rad'  half-field for the default 5-field set (rad).  Required
%                unless 'fields' is given.
%     'configs'  1xNc configuration struct array (macos.design.configs_
%                from_table shape); [] = nominal only.  Applied with the
%                public perturb + its exact inverse (footprint measurement
%                does not need the Jacobian-grade config_axis assert).
%     'stop_elt' set the aperture stop at this element before tracing.
%     'ngridpts' ray-grid sampling override.
%
%   RETURNS struct fp with, per element e = 1..nElt:
%     fp.nElt          element count
%     fp.type{e}       element type name
%     fp.n_max(e)      max passed-ray count over (field,config)
%     fp.centroid(:,e) 3x1 global centroid of the union footprint
%     fp.radius(e)     max distance from centroid over the union footprint
%   plus convenience maps for the promoter:
%     fp.centroid_map  containers.Map(iElt -> 3x1 centroid) over lit elts
%     fp.radius_map    containers.Map(iElt -> radius) over lit elts
%
%   See also: macos.ray_hist, macos.get_elt_info, run_sensitivities,
%             macos.design.promote_segments_freeform, flag_zero_norm_channels.

arguments
    m
    rx (1,:) char
    opts.fields double = []
    opts.fov_rad double = []
    opts.configs = []
    opts.stop_elt double = []
    opts.ngridpts double = []
end
m.load_rx(rx);
if ~isempty(opts.stop_elt), m.stop(int32(opts.stop_elt)); end
if ~isempty(opts.ngridpts), m.set_src_sampling(opts.ngridpts); end
m.modify();

nElt = m.num_elt();
nom  = m.get_src_fov();

% field set: explicit N x 2, or the stock centre + 4 corners
F = opts.fields;
if isempty(F)
    assert(~isempty(opts.fov_rad), ...
        'optic_footprints: pass ''fields'' or ''fov_rad''');
    f = opts.fov_rad;
    F = [0 0; -f f; f f; -f -f; f -f];
end

% Normalise configs to a plain list (name + setter cells).  We apply each
% with the public perturb / absolute setters and undo with the exact
% inverse; footprint measurement does not need the full config_axis
% snapshot/assert integrity guard (that is for Jacobian columns), so this
% helper stays in the design layer with no private-package dependency.
cfgs = opts.configs;
if isempty(cfgs), cfgs = struct('name', {'nom'}, 'set', {{}}); end

n_max = zeros(1, nElt);
% running union footprint per element: keep all in-slot positions, reduce later
Pcell = repmat({zeros(3,0)}, 1, nElt);

for ic = 1:numel(cfgs)
    sl = cfgs(ic).set;  if isempty(sl), sl = {}; end
    apply_setters_(m, sl);      m.modify();
    for k = 1:size(F, 1)
        new_dir = field_to_chfraydir_(nom.src_dir, F(k,1), F(k,2));
        m.set_src_fov('src_pos', nom.src_pos, 'src_dir', new_dir, ...
                      'zSrc', nom.zSrc);
        m.modify();
        macos.ray_hist('on');
        s = m.trace();
        h = macos.ray_hist(s.nRays);
        macos.ray_hist('off');
        for e = 1:nElt
            okc = logical(h.ok(:, e+1)).';        % slot 1 = source
            nk  = nnz(okc);
            n_max(e) = max(n_max(e), nk);
            if nk > 0
                Pcell{e} = [Pcell{e}, squeeze(h.P(:, okc, e+1))];
            end
        end
    end
    undo_setters_(m, sl);       m.modify();
end
% restore nominal field
m.set_src_fov('src_pos', nom.src_pos, 'src_dir', nom.src_dir, 'zSrc', nom.zSrc);
m.modify();

centroid = nan(3, nElt);  radius = zeros(1, nElt);  type = cell(1, nElt);
cmap = containers.Map('KeyType','double','ValueType','any');
rmap = containers.Map('KeyType','double','ValueType','double');
for e = 1:nElt
    type{e} = m_type_(e);
    P = Pcell{e};
    if isempty(P), continue; end
    c = mean(P, 2);
    centroid(:, e) = c;
    radius(e) = max(sqrt(sum((P - c).^2, 1)));
    cmap(e) = c;  rmap(e) = radius(e);
end

fp = struct('nElt', nElt, 'type', {type}, 'n_max', n_max, ...
    'centroid', centroid, 'radius', radius, ...
    'centroid_map', cmap, 'radius_map', rmap);
end


% ---------------------------------------------------------------------
function t = m_type_(e)
try, t = macos.get_elt_info(e).type; catch, t = ''; end
end


function new_dir = field_to_chfraydir_(dir_nom, dx_rad, dy_rad)
% Direction-cosine offset on the nominal ChfRayDir, then renormalise --
% the same convention the dw_d*_multi supervisors use for their field set.
v = dir_nom(:) + [dx_rad; dy_rad; 0];
n = norm(v);
assert(n > 0, 'optic_footprints: zero-magnitude direction after field offset');
new_dir = v / n;
end


% ---------------------------------------------------------------------
function apply_setters_(m, sl)
%APPLY_SETTERS_  Dispatch a configs_from_table setter list {fname,elt,args}.
for k = 1:numel(sl)
    e = sl{k};  fn = e{1};  if isstring(fn), fn = char(fn); end
    m.(fn)(e{2:end});
end
end


function undo_setters_(m, sl)
%UNDO_SETTERS_  Reverse the list.  perturb is undone by the engine's exact
%   inverse (negated rotation+translation); absolute setters are not used by
%   configs_from_table (it emits perturb only), so this covers the footprint
%   use.  Not the Jacobian-grade restore -- see private/config_axis for that.
for k = numel(sl):-1:1
    e = sl{k};  fn = e{1};  if isstring(fn), fn = char(fn); end
    if strcmp(fn, 'perturb')
        args = e(3:end);                 % name/value pairs
        for q = 1:2:numel(args)
            key = args{q};  if isstring(key), key = char(key); end
            if any(strcmpi(key, {'rotation','translation'}))
                args{q+1} = -args{q+1};  % negate the deltas
            end
        end
        m.perturb(e{2}, args{:});
    else
        error('optic_footprints:undo', ...
            'cannot invert setter ''%s'' (footprint configs must be perturb)', fn);
    end
end
end
