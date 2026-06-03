function out = zrn_freeform(srf, opts)
%MACOS.ZRN_FREEFORM  Get or set the composite description of a FreeForm element.
%   out = macos.zrn_freeform(SRF) returns a struct with fields:
%       .ff     [] or struct with fields zern_type, modes, coefs,
%               norm_rad, csys (csys: struct with pos, x, y, z 3-vectors)
%       .mon    [] or same shape as .ff (the MonZern overlay)
%       .grid   [] or struct with fields dx, mat, csys
%   [] means the component is not active on this element.
%
%   macos.zrn_freeform(SRF, 'ff', S) writes the FF Zernike component.
%   macos.zrn_freeform(SRF, 'mon', S) writes the Mon Zernike component.
%   macos.zrn_freeform(SRF, 'grid', G) writes the Grid component.
%
%   By default, ANY un-specified component is PRESERVED (the function
%   reads its current state and writes it back so the setter call
%   carries the existing values forward).  To EXPLICITLY clear a
%   component, pass the literal string 'clear' for it -- e.g.
%   macos.zrn_freeform(SRF, 'ff', new_ff, 'grid', 'clear') updates the
%   FF Zernike, wipes the Grid component, and preserves the Mon
%   overlay.
%
%   Mirrors pymacos's zrn_freeform composite accessor.  SRF must be a
%   single FreeForm (SrfType=14) element id.
%
%   Component struct schema:
%     ff  / mon: struct('zern_type', int, 'modes', N×1 int, ...
%                       'coefs', N×1 double, 'norm_rad', double, ...
%                       'csys', struct('pos', 3×1, 'x', 3×1, 'y', 3×1, ...
%                                       'z', 3×1))
%     grid:     struct('dx', double, 'mat', N×N double, ...
%                       'csys', struct(...))
%
%   See also: macos.set_elt_mon_zrn_coef, macos.set_elt_ff_zrn_coef.
arguments
    srf       (1,1) double {mustBeInteger, mustBePositive}
    opts.ff   = []   % [] preserve, 'clear' wipe, struct set
    opts.mon  = []
    opts.grid = []
end

N_ZRN_COEF_MAX = 66;

is_get = isempty(opts.ff) && isempty(opts.mon) && isempty(opts.grid);

% Find the live grid size on this surface so the getter allocates a
% buffer large enough for the Fortran routine to write back into.
% Returns 0 for elements without grid data; mmacos clamps the lower
% bound to 1 so we always have a non-degenerate matrix to pass.
grid_n = mmacos('elt_srf_grid_size', srf, 1);
grid_n = max(double(grid_n), 1);

% --- Inner get-or-set call dispatcher ----------------------------------
    function r = do_call(setter, ff_in, mon_in, grid_in, grid_sz)
        % For the GETTER, allocate full N_ZRN_COEF_MAX buffers so the
        % Fortran routine can write all 66 coefficient slots.  For the
        % SETTER, pass the actual user-supplied modes/coefs and let
        % nFF / nMon reflect their real lengths -- padding with zeros
        % would cause the Fortran scatter `FFZernCoef(FFZernModes_,
        % srf) = FFZernCoef_` to write to index 0 (out-of-bounds).
        %
        % FF
        if isempty(ff_in)
            ifFF = 0; FFType = 0;
            if setter
                FFModes = double(1); FFCoef = 0;  % single dummy slot
            else
                FFModes = zeros(N_ZRN_COEF_MAX, 1);
                FFCoef  = zeros(N_ZRN_COEF_MAX, 1);
            end
            lFF = 0;  pFF = zeros(3,1);
            xFF = zeros(3,1); yFF = zeros(3,1); zFF = zeros(3,1);
        else
            ifFF = 1;
            FFType  = double(ff_in.zern_type);
            FFModes = double(ff_in.modes(:));
            FFCoef  = double(ff_in.coefs(:));
            lFF = double(ff_in.norm_rad);
            pFF = ff_in.csys.pos(:);  xFF = ff_in.csys.x(:);
            yFF = ff_in.csys.y(:);    zFF = ff_in.csys.z(:);
        end
        % Mon
        if isempty(mon_in)
            ifMon = 0; MonType = 0;
            if setter
                MonModes = double(1); MonCoef = 0;
            else
                MonModes = zeros(N_ZRN_COEF_MAX, 1);
                MonCoef  = zeros(N_ZRN_COEF_MAX, 1);
            end
            lMon = 0; pMon = zeros(3,1);
            xMon = zeros(3,1); yMon = zeros(3,1); zMon = zeros(3,1);
        else
            ifMon = 1;
            MonType  = double(mon_in.zern_type);
            MonModes = double(mon_in.modes(:));
            MonCoef  = double(mon_in.coefs(:));
            lMon = double(mon_in.norm_rad);
            pMon = mon_in.csys.pos(:);  xMon = mon_in.csys.x(:);
            yMon = mon_in.csys.y(:);    zMon = mon_in.csys.z(:);
        end
        % Grid
        if isempty(grid_in)
            ifGrid = 0; GridDx = 0;
            GridMat = zeros(grid_sz, grid_sz);
            pData = zeros(3,1); xData = zeros(3,1);
            yData = zeros(3,1); zData = zeros(3,1);
        else
            ifGrid = 1;
            GridDx  = double(grid_in.dx);
            GridMat = double(grid_in.mat);
            pData = grid_in.csys.pos(:);  xData = grid_in.csys.x(:);
            yData = grid_in.csys.y(:);    zData = grid_in.csys.z(:);
        end
        nFF  = numel(FFModes);
        nMon = numel(MonModes);
        Nx = size(GridMat, 2);
        Ny = size(GridMat, 1);

        if setter
            % SET: dispatch and ignore plhs outputs.
            mmacos('elt_srf_zrn_FreeForm', srf, ...
                   ifFF, FFType, FFModes, FFCoef, lFF, pFF, xFF, yFF, zFF, ...
                   ifMon, MonType, MonModes, MonCoef, ...
                   lMon, pMon, xMon, yMon, zMon, ...
                   ifGrid, GridDx, GridMat, ...
                   pData, xData, yData, zData, 1, Nx, Ny, nFF, nMon);
            r = [];
        else
            [ifFFo, FFTypeo, ~, FFCoefo, lFFo, pFFo, xFFo, yFFo, zFFo, ...
             ifMono, MonTypeo, ~, MonCoefo, ...
             lMono, pMono, xMono, yMono, zMono, ...
             ifGrido, GridDxo, GridMato, ...
             pDatao, xDatao, yDatao, zDatao] = ...
                mmacos('elt_srf_zrn_FreeForm', srf, ...
                       ifFF, FFType, FFModes, FFCoef, lFF, pFF, xFF, yFF, zFF, ...
                       ifMon, MonType, MonModes, MonCoef, ...
                       lMon, pMon, xMon, yMon, zMon, ...
                       ifGrid, GridDx, GridMat, ...
                       pData, xData, yData, zData, 0, Nx, Ny, nFF, nMon);
            r = struct();
            r.ff   = zern_from_api(logical(ifFFo), int32(FFTypeo), ...
                                     FFCoefo, lFFo, pFFo, xFFo, yFFo, zFFo);
            r.mon  = zern_from_api(logical(ifMono), int32(MonTypeo), ...
                                     MonCoefo, lMono, pMono, xMono, yMono, zMono);
            r.grid = grid_from_api(logical(ifGrido), GridDxo, GridMato, ...
                                     pDatao, xDatao, yDatao, zDatao);
        end
    end

if is_get
    out = do_call(false, [], [], [], grid_n);
    return;
end

% Setter path: substitute current state for any un-specified arg
% ([]) and resolve 'clear' sentinels to [].
need_current = isempty(opts.ff) || isempty(opts.mon) || isempty(opts.grid);
if need_current
    cur = do_call(false, [], [], [], grid_n);
    if isempty(opts.ff),   opts.ff   = cur.ff;   end
    if isempty(opts.mon),  opts.mon  = cur.mon;  end
    if isempty(opts.grid), opts.grid = cur.grid; end
end
opts.ff   = resolve_clear(opts.ff);
opts.mon  = resolve_clear(opts.mon);
opts.grid = resolve_clear(opts.grid);

% Set: pick grid_sz from the data we'll send.
if isstruct(opts.grid)
    grid_sz = size(opts.grid.mat, 1);
else
    grid_sz = 1;
end
do_call(true, opts.ff, opts.mon, opts.grid, grid_sz);
out = [];
end


% =====================================================================
function out = resolve_clear(arg)
if (ischar(arg) || isstring(arg)) && strcmpi(arg, 'clear')
    out = [];
elseif isstruct(arg)
    out = arg;
else
    out = [];
end
end


function s = zern_from_api(is_active, ztype, coefs, norm_rad, pos, xd, yd, zd)
if ~is_active
    s = [];
    return;
end
% Trim trailing zeros so we don't return all 66 entries when only a
% few modes are populated.  Intermediate zeros are preserved so
% set/get round-trips faithfully.
nz = find(coefs ~= 0, 1, 'last');
if isempty(nz), nz = 1; end
s = struct();
s.zern_type = double(ztype);
s.modes     = (1:nz).';
s.coefs     = double(coefs(1:nz));
s.norm_rad  = double(norm_rad);
s.csys      = struct('pos', double(pos(:)), 'x', double(xd(:)), ...
                     'y', double(yd(:)),   'z', double(zd(:)));
end


function s = grid_from_api(is_active, dx, mat, pos, xd, yd, zd)
if ~is_active
    s = [];
    return;
end
s = struct();
s.dx   = double(dx);
s.mat  = double(mat);
s.csys = struct('pos', double(pos(:)), 'x', double(xd(:)), ...
                'y', double(yd(:)),    'z', double(zd(:)));
end
