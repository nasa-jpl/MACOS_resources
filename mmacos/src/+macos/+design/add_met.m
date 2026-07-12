function out = add_met(in_path, seg, opts)
%ADD_MET  Emit Stewart-truss laser metrology into a segmented Rx.
%
%   out = macos.design.add_met(in_path, seg, 'hub', k, 'r_fid', r)
%   appends engine met blocks (nMetPos/tMetElt/metBeamFlg) to the
%   prescription: 6 launchers on EACH segment (hexagon at
%   r_launch_frac*lMon in the segment's own face triad) beamed to nf
%   fiducials on the hub element ("M2"), plus 6 launchers ringed
%   around each extra_sources element ("points around M3") beamed to
%   the same hub fiducials.  One measurement per launcher = the change
%   in straight-line launcher->fiducial distance (Dave 2026-07-12:
%   Stewart-platform geometry between each segment and M2, and between
%   M3 and M2; >= as many measurements as DOFs).
%
%   seg = the macos.design.segment_rx output (frames + seg_elts).
%   opts: hub (element index, required), r_fid (hub fiducial ring
%   radius, BaseUnits, required), nf (3..6, default 3), extra_sources
%   ([] element indices), r_extra (default r_fid), r_launch_frac
%   (default 0.7), out_in (default <in>_met.in beside in_path).
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
    opts.r_launch_frac (1,1) double {mustBePositive} = 0.7
    opts.launch_clock (1,1) double = pi/6   % launcher hexagon clocking, rad
    opts.fid_clock (1,1) double = 0         % hub fiducial ring clocking, rad
    opts.out_in (1,1) string = ""
end
if isempty(opts.r_extra), opts.r_extra = opts.r_fid; end

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

% Hub fiducials: ring of nf points about the hub vertex, in the plane
% perpendicular to its psi.
pv = vec3_(opts.hub, "VptElt");
ps = vec3_(opts.hub, "psiElt"); ps = ps/norm(ps);
[~, imin] = min(abs(ps)); e = zeros(3,1); e(imin) = 1;
xh = cross(ps, e); xh = xh/norm(xh); yh = cross(ps, xh);
th = opts.fid_clock + 2*pi*(0:opts.nf-1)'/opts.nf;
fid = pv + opts.r_fid*(xh*cos(th') + yh*sin(th'));   % 3 x nf

% Stewart pairing: launcher k -> fiducial pair(k); crossing struts.
if opts.nf == 3, pair = [1 2 2 3 3 1];
else,            pair = mod((0:5), opts.nf) + 1;
end

src_pts = zeros(3,0); tgt_pts = zeros(3,0);
ins = cell(numel(starts), 1);      % met text to insert per element

% Per-segment launchers: hexagon in the segment face triad.
tl6 = opts.launch_clock + 2*pi*(0:5)'/6;
for s = 1:seg.nseg
    k = seg.seg_elts(s); f = seg.frames(s);
    r = opts.r_launch_frac * f.lmon;
    L = f.rpt + r*(f.xhat*cos(tl6') + f.yhat*sin(tl6'));  % 3 x 6
    ins{k} = met_block_(L, opts.hub, opts.nf, pair);
    src_pts = [src_pts, L]; %#ok<AGROW>
    tgt_pts = [tgt_pts, fid(:, pair)]; %#ok<AGROW>
end
% Extra sources ("around M3"): ring about the element vertex.
for k = opts.extra_sources(:)'
    pvk = vec3_(k, "VptElt"); psk = vec3_(k, "psiElt"); psk = psk/norm(psk);
    [~, imin] = min(abs(psk)); e = zeros(3,1); e(imin) = 1;
    xk = cross(psk, e); xk = xk/norm(xk); yk = cross(psk, xk);
    L = pvk + opts.r_extra*(xk*cos(tl6') + yk*sin(tl6'));
    ins{k} = met_block_(L, opts.hub, opts.nf, pair);
    src_pts = [src_pts, L]; %#ok<AGROW>
    tgt_pts = [tgt_pts, fid(:, pair)]; %#ok<AGROW>
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
             'src_pts', src_pts, 'tgt_pts', tgt_pts);

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
