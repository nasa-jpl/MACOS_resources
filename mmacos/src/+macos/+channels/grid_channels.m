function chans = grid_channels(session, influence, opts)
%MACOS.CHANNELS.GRID_CHANNELS  Build grid-data channels per grid element × map.
%   chans = macos.channels.grid_channels(SESSION, INFLUENCE) returns a cell
%   array of GridChannel handles, one per (grid-bearing element, influence
%   map).  INFLUENCE is [N×N×K] -- K influence-function maps; N must match each
%   element's grid sampling.  Restrict the element set with 'elts'.
%
%   Eligibility (any GridData-enabled SrfType) comes from
%   macos.find_grid_elts.  Channels are built in canonical order: element-
%   major, map-minor.
%
%   See also: macos.channels.GridChannel, macos.dw_dgrid,
%             macos.find_grid_elts, macos.channels.zernike_channels.
arguments
    session
    influence (:,:,:) double {mustBeReal, mustBeFinite}
    opts.elts (:,1) double = []
end
g = macos.find_grid_elts();
if ~isempty(opts.elts)
    g = intersect(g, opts.elts(:));
end
K = size(influence, 3);
chans = {};
for ie = g(:).'
    nsz = double(mmacos('elt_srf_grid_size', ie, 1));
    if size(influence,1) ~= nsz || size(influence,2) ~= nsz
        error('macos:channels:grid_channels:size', ...
            'influence is %dx%d but elt %d grid is %dx%d.', ...
            size(influence,1), size(influence,2), ie, nsz, nsz);
    end
    for kk = 1:K
        chans{end+1,1} = macos.channels.GridChannel( ...
            session, ie, influence(:,:,kk), kk); %#ok<AGROW>
    end
end
if isempty(chans)
    error('macos:channels:grid_channels:none', ...
        'no grid-bearing elements found in the loaded prescription');
end
end
