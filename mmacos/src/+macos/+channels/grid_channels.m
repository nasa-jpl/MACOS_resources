function chans = grid_channels(session, influence, opts)
%MACOS.CHANNELS.GRID_CHANNELS  Build grid-data channels per grid element × map.
%   chans = macos.channels.grid_channels(SESSION, INFLUENCE) returns a cell
%   array of GridChannel handles, one per (grid-bearing element, influence
%   map).  Channels are built in canonical order: element-major, map-minor.
%   Eligibility (any GridData-enabled SrfType) comes from macos.find_grid_elts;
%   restrict the element set with 'elts'.
%
%   INFLUENCE may be:
%     * [N×N×K] numeric -- K influence-function maps applied to EVERY grid
%       element (the original behaviour).  N must match each element's grid.
%     * a segment_grid_basis OUTPUT struct (has field .seg) -- a PER-SEGMENT
%       basis: element ie uses OUT.seg(s).B where OUT.seg(s).iElt == ie.  This
%       is what lets a generator give each (edge) segment its own bespoke
%       mask + modes.  Every segment must supply the same number of maps K.
%     * a cell array, one [N×N×K] per grid element in find_grid_elts order.
%
%   See also: macos.channels.GridChannel, macos.dw_dgrid,
%             macos.find_grid_elts, macos.segment_grid_basis.
arguments
    session
    influence                          % [N×N×K] | segment_grid_basis struct | cell
    opts.elts (:,1) double = []
end
g = macos.find_grid_elts();
if ~isempty(opts.elts)
    g = intersect(g, opts.elts(:));
end
% A per-segment basis struct is authoritative for WHICH elements get channels:
% restrict to the elements it covers.  find_grid_elts keys on nGridMat alone,
% so it also lists grid-bearing non-candidates -- a conforming Reference (a
% passive trace target holding a Zernike basis definition) or a downstream
% full-aperture refractor -- for which the per-segment basis has no entry. -CC
if isstruct(influence) && isfield(influence, 'seg')
    g = intersect(g, [influence.seg.iElt].', 'stable');
end
get_basis = resolve_influence_(influence, g);    % @(ie) -> [N×N×K] for element ie

chans = {};
for ie = g(:).'
    nsz = double(mmacos('elt_srf_grid_size', ie, 1));
    B   = get_basis(ie);
    if size(B,1) ~= nsz || size(B,2) ~= nsz
        error('macos:channels:grid_channels:size', ...
            'influence is %dx%d but elt %d grid is %dx%d.', ...
            size(B,1), size(B,2), ie, nsz, nsz);
    end
    for kk = 1:size(B,3)
        chans{end+1,1} = macos.channels.GridChannel( ...
            session, ie, B(:,:,kk), kk); %#ok<AGROW>
    end
end
if isempty(chans)
    error('macos:channels:grid_channels:none', ...
        'no grid-bearing elements found in the loaded prescription');
end
end

% ---------------------------------------------------------------------------
function get_basis = resolve_influence_(influence, g)
% Return a function get_basis(ie) -> [N×N×K] for grid element ie.
if isnumeric(influence)
    mustBeReal(influence);   mustBeFinite(influence);
    get_basis = @(ie) influence;                       % one basis for every element
elseif isstruct(influence) && isfield(influence, 'seg')
    segs = influence.seg;   ielt = [segs.iElt];        % per-segment, keyed by iElt
    get_basis = @(ie) seg_basis_(segs, ielt, ie);
elseif iscell(influence)
    if numel(influence) ~= numel(g)
        error('macos:channels:grid_channels:cell', ...
            'cell influence has %d entries but there are %d grid elements.', ...
            numel(influence), numel(g));
    end
    gv = g(:).';
    get_basis = @(ie) influence{find(gv == ie, 1)};
else
    error('macos:channels:grid_channels:influence', ...
        ['influence must be [N x N x K], a segment_grid_basis struct (with ', ...
         '.seg), or a cell array per grid element.']);
end
end

function B = seg_basis_(segs, ielt, ie)
s = find(ielt == ie, 1);
if isempty(s)
    error('macos:channels:grid_channels:noseg', ...
        'per-segment influence has no basis for grid element %d.', ie);
end
B = segs(s).B;
end
