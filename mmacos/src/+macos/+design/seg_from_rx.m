function seg = seg_from_rx(in_path, opts)
%SEG_FROM_RX  Rehydrate a segment_rx-style struct from a segmented .in.
%
%   seg = macos.design.seg_from_rx(IN_PATH) rebuilds the struct that
%   macos.design.segment_rx returns -- frames, seg_elts, tiling
%   metadata -- from the prescription file ALONE, so downstream stage
%   runners (add_met, seg_boundary, met_view, met_layout_opt, run_met)
%   can take a segmented .in as their interface without carrying the
%   in-memory struct across sessions.  The .in file IS the stage
%   handoff (Dave 2026-07-19).
%
%   Engine-truth sources (the Rx is LOADED into the current session as
%   a side effect unless 'load' is false):
%     - segment elements: elt_info_get type scan (Element= Segment)
%     - frames:           met_bodies (RptElt pivot + TElt triad -- the
%                         same triads segment_rx emitted) + lMon
%     - tiling:           src_seg_get (GridType / nSeg / width / gap)
%
%   OPTS:
%     'hx'     path to the SegMirMaker Hx.m sidecar ('' = none); kept
%              verbatim in seg.hx for edge_sensors
%     'load'   true (default) = macos.load_rx(in_path) first; false =
%              the Rx is already loaded (indices must match!)
%
%   The rehydrated struct carries seg.rehydrated = true and has no
%   .run/.presc fields (no SegMirMaker scratch state).
%
%   See also: macos.design.segment_rx, macos.design.add_met,
%             macos.design.seg_boundary, macos.design.met_bodies.

arguments
    in_path (1,1) string
    opts.hx (1,1) string = ""
    opts.load (1,1) logical = true
end
if ~isfile(in_path)
    error('macos:design:seg_from_rx:file', 'not found: %s', in_path);
end
if opts.load
    macos.load_rx(char(in_path));
end
n_elt = macos.num_elt();

% segment elements + names (engine type scan; names from the text)
L = readlines(in_path);  tl = strtrim(L);
se = zeros(1, 0);  names = strings(1, 0);
inames = find(startsWith(tl, "EltName="));
for k = 1:n_elt
    if strcmp(macos.get_elt_info(k).type, 'Segment')
        se(end+1) = k; %#ok<AGROW>
        nm = sprintf('Seg%d', numel(se));
        if k <= numel(inames)
            t = regexp(tl(inames(k)), 'EltName=\s*(\S+)', 'tokens', 'once');
            if ~isempty(t), nm = char(t{1}); end
        end
        names(end+1) = nm; %#ok<AGROW>
    end
end
if isempty(se)
    error('macos:design:seg_from_rx:noseg', ...
        '%s declares no Element= Segment blocks', in_path);
end

% frames: engine perturbation triads (identical to segment_rx's face
% triads by the met_bodies contract) + lMon per segment
bodies = macos.design.met_bodies(se);
frames = repmat(struct('name', '', 'rpt', zeros(3,1), 'xhat', zeros(3,1), ...
    'yhat', zeros(3,1), 'zhat', zeros(3,1), 'lmon', 0), 1, numel(se));
for s = 1:numel(se)
    T = bodies(s).T;
    frames(s) = struct('name', char(names(s)), 'rpt', bodies(s).rpt, ...
        'xhat', T(:,1), 'yhat', T(:,2), 'zhat', T(:,3), ...
        'lmon', macos.get_elt_info(se(s)).lmon);
end

% tiling metadata (source header truth)
[gid, nsg, w, gp] = mmacos('src_seg_get');
gnames = containers.Map({3, 4}, {'Hex', 'Pie'});
grid = '';
if isKey(gnames, gid), grid = gnames(gid); end
if nsg ~= numel(se)
    warning('macos:design:seg_from_rx:nseg', ...
        ['source header nSeg=%d but %d Segment elements found -- ' ...
         'using the element count'], nsg, numel(se));
end

seg = struct('in', char(in_path), 'nseg', numel(se), 'seg_elts', se, ...
             'n_elt', n_elt, 'frames', frames, 'hx', char(opts.hx), ...
             'width', w, 'gap', gp, 'grid', grid, ...
             'dropped_apstop', "", 'rehydrated', true);
end
