classdef GridChannel < handle
%MACOS.CHANNELS.GRIDCHANNEL  One grid-data influence-function sensitivity channel.
%
%   A single (element, influence-map) pair on a grid-bearing surface of ANY
%   GridData-enabled SrfType (GridData / AsGrData / MonGrData / ZrnGridData /
%   FreeForm).  The DOF coefficient scales an N×N influence map that is ADDED
%   to the element's grid data (GridMat) via macos.elt_grid_add -- which
%   preserves SrfType (unlike GMI, which forces SrfType->9 and so clobbers a
%   composite surface's conic / Zernike / monomial components).
%
%   apply(d) leaves the surface at nominal + d*map (it tracks the applied
%   amount so apply(+δ)/apply(-δ)/restore() give a clean ±δ/0 central-
%   difference sequence -- see macos.dwdz_for_current_source).  restore()
%   returns to nominal.
%
%   Construction:
%     ch = macos.channels.GridChannel(session, iElt, map, idx)
%
%   See also: macos.channels.grid_channels, macos.dw_dgrid,
%             macos.channels.ZernikeCoefChannel, macos.elt_grid_add.

    properties (SetAccess = private)
        iElt    (1,1) double
        idx     (1,1) double      % influence-map index (for name())
        map     double            % N×N influence function
        session                   % macos.Session handle
    end
    properties (Access = private)
        applied (1,1) double = 0  % currently-applied coefficient
    end

    methods
        function obj = GridChannel(session, iElt, map, idx)
            arguments
                session
                iElt (1,1) double {mustBeInteger, mustBePositive}
                map  (:,:) double {mustBeReal, mustBeFinite}
                idx  (1,1) double {mustBeInteger, mustBePositive} = 1
            end
            obj.session = session;
            obj.iElt    = iElt;
            obj.map     = map;
            obj.idx     = idx;
        end

        function apply(obj, increment)
            arguments
                obj
                increment (1,1) double
            end
            % Add only the delta from the currently-applied amount so the
            % +δ / -δ / 0 finite-difference sequence is exact via the
            % additive elt_grid_add API (no full-grid snapshot needed).
            delta = increment - obj.applied;
            if delta ~= 0
                macos.elt_grid_add(obj.iElt, delta * obj.map);  % ADD: SrfType preserved
            end
            obj.applied = increment;
            % elt_grid_add already invalidates the trace cache (modified_rx);
            % the explicit modify() keeps the apply()->dirty contract.
            obj.session.modify();
        end

        function restore(obj)
            if obj.applied ~= 0
                macos.elt_grid_add(obj.iElt, -obj.applied * obj.map);
                obj.applied = 0;
                obj.session.modify();
            end
        end

        function s = name(obj)
            s = sprintf('Elt %d Grid%d', obj.iElt, obj.idx);
        end
    end
end
