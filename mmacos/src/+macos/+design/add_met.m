function out = add_met(in_path, seg, opts)
%ADD_MET  Emit Stewart-truss laser metrology into a segmented Rx.
%
%   out = macos.design.add_met(in_path, seg, 'hub', k, 'r_fid', r)
%   appends engine met blocks (nMetPos/tMetElt/metBeamFlg) to the
%   prescription: 6 launchers on EACH segment (at the segment edge by
%   default, see below) beamed to nf fiducials on the hub element
%   ("M2"), plus 6 launchers ringed around each extra_sources element
%   ("points around M3") beamed to the same hub fiducials.  One
%   measurement per launcher = the change in straight-line
%   launcher->fiducial distance (Dave 2026-07-12: Stewart-platform
%   geometry between each segment and M2, and between M3 and M2;
%   >= as many measurements as DOFs).
%
%   seg = the macos.design.segment_rx output (frames + seg_elts).
%   opts: hub (element index, required), r_fid (hub fiducial ring
%   radius, BaseUnits, required), nf (3..6, default 3), extra_sources
%   ([] element indices), r_extra (default r_fid), out_in (default
%   <in>_met.in beside in_path).
%
%   Default launcher placement (Dave 2026-07-16): launchers sit AT the
%   segment edge with a small outward clearance (edge_off, default 5
%   BaseUnits) so they never obscure the reflecting surface -- 6 points
%   equally spaced along the segment's TRUE boundary
%   (macos.design.seg_boundary: hex tiles OR pie wedges, offset
%   outward), phased by launch_clock.  Pass r_launch_frac to instead
%   use the legacy interior ring at r_launch_frac*lMon in the segment
%   face triad (also the fallback for tilings seg_boundary does not
%   model).
%
%   out: .in, .n_beams, .src_pts/.tgt_pts (3 x n_beams, global,
%   BaseUnits — engine buffer order) so tests can pin met().l exactly.

arguments
    in_path (1,1) string
    seg (1,1) struct
    opts.hub (1,1) double {mustBeInteger, mustBePositive}
    opts.r_fid (1,1) double {mustBePositive}
    opts.nf (1,1) double {mustBeMember(opts.nf, [3 4 5 6])} = 3
    opts.extra_sources double = []
    opts.r_extra double = []
    opts.r_launch_frac double = []          % legacy interior ring (frac of lMon)
    opts.edge_off (1,1) double {mustBeNonnegative} = 5  % edge clearance, BaseUnits
    opts.launch_clock (1,1) double = pi/6   % launcher hexagon clocking, rad
    opts.extra_clock double = []            % extra-ring clocking ([] = launch_clock)
    opts.extra_pair_map double = []         % extra-ring fiducial map ([] = pair)
    opts.fid_clock (1,1) double = 0         % hub fiducial ring clocking, rad
    opts.launch_pts cell = {}               % override: {nseg} of 3x6 GLOBAL launcher points
    opts.pair_map double = []               % override: 1x6 fiducial index per
                                            % launcher, or nseg x 6 for a
                                            % per-segment (rotational) assignment
    opts.out_in (1,1) string = ""
end
% r_extra: [] (default) = size each extra ring from that element's own
% physical radius + edge_off (resolved per element below)

lines = readlines(in_path);
starts = find(startsWith(strtrim(lines), "iElt="));
    function v = vec3_(k, key)
        bend = numel(lines);
        if k < numel(starts), bend = starts(k+1) - 1; end
        b = lines(starts(k):bend);
        t = regexp(b(find(startsWith(strtrim(b), key + "="), 1)), ...
                   key + '=\s*(\S+)\s+(\S+)\s+(\S+)', 'tokens', 'once');
        v = str2double(string(t))';
    end
    function r = eltrad_(k)
        % element physical semi-diameter: circular ApVec radius if
        % present, else lMon (normalization radius), else NaN
        bend = numel(lines);
        if k < numel(starts), bend = starts(k+1) - 1; end
        b = strtrim(lines(starts(k):bend));
        r = NaN;
        ia = find(startsWith(b, "ApVec="), 1);
        if ~isempty(ia)
            t = regexp(b(ia), 'ApVec=\s*(\S+)', 'tokens', 'once');
            r = str2double(string(t));
        end
        if ~isfinite(r)
            il = find(startsWith(b, "lMon="), 1);
            if ~isempty(il)
                t = regexp(b(il), 'lMon=\s*(\S+)', 'tokens', 'once');
                r = str2double(string(t));
            end
        end
    end

% Hub fiducials: ring of nf points about the hub vertex, in the plane
% perpendicular to its psi.  Fiducials must be MOUNTED ON the hub
% mirror (Dave 2026-07-16: near its edge, ~25 mm inside the rim; there
% is no structure beyond) -- warn when the requested ring leaves it.
pv = vec3_(opts.hub, "VptElt");
ps = vec3_(opts.hub, "psiElt"); ps = ps/norm(ps);
hub_rad = eltrad_(opts.hub);
if isfinite(hub_rad) && opts.r_fid > hub_rad
    warning('macos:design:add_met:fid_off_hub', ...
        ['fiducial ring r_fid=%g exceeds the hub element''s ' ...
         'semi-diameter %g -- no structure to mount on'], ...
        opts.r_fid, hub_rad);
end
[~, imin] = min(abs(ps)); e = zeros(3,1); e(imin) = 1;
xh = cross(ps, e); xh = xh/norm(xh); yh = cross(ps, xh);
th = opts.fid_clock + 2*pi*(0:opts.nf-1)'/opts.nf;
fid = pv + opts.r_fid*(xh*cos(th') + yh*sin(th'));   % 3 x nf

% Beam assignment: launcher k -> fiducial pair(k).  Default = Stewart
% crossing struts; pair_map overrides (the layout optimizer iterates
% assignment combinations, Dave 2026-07-16).
if ~isempty(opts.pair_map)
    pm = opts.pair_map;
    if isvector(pm), pm = repmat(pm(:).', seg.nseg, 1); end
    if size(pm, 2) ~= 6 || size(pm, 1) ~= seg.nseg ...
            || any(pm(:) < 1) || any(pm(:) > opts.nf)
        error('macos:design:add_met:pairmap', ...
              ['pair_map must be 1x6 (shared) or nseg x 6 (per-segment ' ...
               'rotational assignment) fiducial indices in 1..nf']);
    end
elseif opts.nf == 3, pm = repmat([1 2 2 3 3 1], seg.nseg, 1);
else,                pm = repmat(mod((0:5), opts.nf) + 1, seg.nseg, 1);
end
pair = pm(1, :);                 % extras fall back to segment 1's map

src_pts = zeros(3,0); tgt_pts = zeros(3,0);
ins = cell(numel(starts), 1);      % met text to insert per element

% Per-segment launchers: edge-offset boundary points (default, hex AND
% pie via seg_boundary) or the legacy interior ring / explicit override.
tl6 = opts.launch_clock + 2*pi*(0:5)'/6;
use_edge = isempty(opts.r_launch_frac);
Boff = [];
if use_edge && isempty(opts.launch_pts)
    try
        Boff = macos.design.seg_boundary(seg, opts.edge_off);
    catch                                   % unmodeled tiling -> legacy ring
        use_edge = false;
    end
end
if isempty(opts.r_launch_frac), opts.r_launch_frac = 0.7; end   % fallback ring
for s = 1:seg.nseg
    k = seg.seg_elts(s); f = seg.frames(s);
    if ~isempty(opts.launch_pts)
        L = opts.launch_pts{s};                           % explicit (3x6 global)
    elseif ~isempty(Boff)
        % 6 equal-arc-length points on the offset boundary, phased by
        % launch_clock (fraction of the perimeter = clock/2pi)
        L = Boff.sample(s, 6, opts.launch_clock/(2*pi));
    else
        r = opts.r_launch_frac * f.lmon;
        L = f.rpt + r*(f.xhat*cos(tl6') + f.yhat*sin(tl6'));  % 3 x 6
    end
    ins{k} = met_block_(L, opts.hub, opts.nf, pm(s, :));
    src_pts = [src_pts, L]; %#ok<AGROW>
    tgt_pts = [tgt_pts, fid(:, pm(s, :))]; %#ok<AGROW>
end
% Extra sources ("around M3"): ring about the element vertex, hugging
% the element's PHYSICAL extent (its aperture/lMon radius + edge_off)
% unless r_extra is given explicitly -- launchers must mount on the
% element's cell, not float in space (Dave 2026-07-16).
xclk = opts.extra_clock;
if isempty(xclk), xclk = opts.launch_clock; end
xpair = opts.extra_pair_map;
if isempty(xpair), xpair = pair; end
if numel(xpair) ~= 6 || any(xpair < 1) || any(xpair > opts.nf)
    error('macos:design:add_met:xpairmap', ...
          'extra_pair_map must be 1x6 fiducial indices in 1..nf');
end
tx6 = xclk + 2*pi*(0:5)'/6;
for k = opts.extra_sources(:)'
    pvk = vec3_(k, "VptElt"); psk = vec3_(k, "psiElt"); psk = psk/norm(psk);
    [~, imin] = min(abs(psk)); e = zeros(3,1); e(imin) = 1;
    xk = cross(psk, e); xk = xk/norm(xk); yk = cross(psk, xk);
    rk = opts.r_extra;
    if isempty(rk)
        rk = eltrad_(k) + opts.edge_off;
        if ~isfinite(rk)
            error('macos:design:add_met:r_extra', ...
                ['element %d has no ApVec/lMon to size its launcher ' ...
                 'ring -- pass r_extra explicitly'], k);
        end
    end
    L = pvk + rk*(xk*cos(tx6') + yk*sin(tx6'));
    ins{k} = met_block_(L, opts.hub, opts.nf, xpair);
    src_pts = [src_pts, L]; %#ok<AGROW>
    tgt_pts = [tgt_pts, fid(:, xpair)]; %#ok<AGROW>
end
% Hub target points (no tMetElt -> contributes no beams itself).
hubtxt = ["          nMetPos=  " + opts.nf; pts_rows_(fid)];
ins{opts.hub} = hubtxt;

% Splice all insertions (after each block's EltName line), bottom-up so
% line indices stay valid.
for k = numel(starts):-1:1
    if isempty(ins{k}), continue; end
    b0 = starts(k);
    b1 = numel(lines); if k < numel(starts), b1 = starts(k+1)-1; end
    ie = find(startsWith(strtrim(lines(b0:b1)), "EltName="), 1) + b0 - 1;
    lines = [lines(1:ie); ins{k}; lines(ie+1:end)];
end

out_in = opts.out_in;
if strlength(out_in) == 0
    [d, b, x] = fileparts(in_path);
    out_in = fullfile(d, b + "_met" + x);
end
writelines(lines, out_in);
out = struct('in', char(out_in), 'n_beams', size(src_pts, 2), ...
             'src_pts', src_pts, 'tgt_pts', tgt_pts, ...
             'hub_pv', pv, 'hub_ps', ps, 'hub_rad', hub_rad);

    function txt = met_block_(L, hub, nf, pr)
        flg = strings(6,1);
        for q = 1:6
            row = zeros(1, nf); row(pr(q)) = 1;
            flg(q) = "  " + join(string(row), "  ");
        end
        txt = ["          nMetPos=  6"; pts_rows_(L); ...
               sprintf("          tMetElt=  %d  %d", hub, nf); flg];
    end
    function rows = pts_rows_(P)
        rows = strings(size(P,2),1);
        for q = 1:size(P,2)
            rows(q) = sprintf('  %.15E  %.15E  %.15E', P(:,q));
        end
    end
end
