classdef StubGroupSession < handle
%STUBGROUPSESSION  Counting stub for GroupedRigidBodyChannel control-flow tests.
%   A minimal duck-typed stand-in for macos.Session that implements exactly
%   the methods GroupedRigidBodyChannel calls, and COUNTS the engine
%   operations (stop_obj / stop / prb_grp / trace / sxp / srs / modify) so a
%   test can pin the WS1 speedup control flow deterministically -- WITHOUT
%   depending on whether a given deck's stop machinery is inert.
%
%   get_stop_info() returns element `stop_e` when stop_e > 0 (an element-stop
%   deck); when stop_e <= 0 it ERRORS, mimicking an object-space-only stop
%   where the engine cannot report a stop element (the ambiguous case).

    properties
        stop_e (1,1) double = 0
        n_elt  (1,1) double = 20
        grp    = []
        counts
    end

    methods
        function obj = StubGroupSession(stop_e, n_elt)
            if nargin >= 1, obj.stop_e = stop_e; end
            if nargin >= 2, obj.n_elt = n_elt;  end
            obj.counts = containers.Map('KeyType', 'char', 'ValueType', 'double');
        end

        function bump(obj, k)
            if isKey(obj.counts, k)
                obj.counts(k) = obj.counts(k) + 1;
            else
                obj.counts(k) = 1;
            end
        end

        function c = count(obj, k)
            if isKey(obj.counts, k), c = obj.counts(k); else, c = 0; end
        end

        function c = cbm(~),        c = 1; end
        function n = num_elt(obj),  n = obj.n_elt; end

        function s = get_stop_info(obj)
            if obj.stop_e <= 0
                error('macos:stub:noStop', 'stub: no element stop set');
            end
            s = struct('elt', obj.stop_e, 'offset', [0 0]);
        end

        function m = get_elt_grp(obj, ~),      m = obj.grp; end
        function set_elt_grp(obj, ~, m),       obj.grp = m(:); end
        function del_elt_grp(obj, ~),          obj.grp = []; end

        function prb_grp(obj, varargin),       obj.bump('prb_grp'); end
        function stop_obj(obj, varargin),      obj.bump('stop_obj'); end
        function stop(obj, varargin),          obj.bump('stop'); end
        function modify(obj),                  obj.bump('modify'); end
        function trace(obj, varargin),         obj.bump('trace'); end
        function xp = sxp(obj, varargin),      obj.bump('sxp'); xp = []; end
        function srs(obj, varargin),           obj.bump('srs'); end
    end
end
