function rep = fold_station_report(t, opts)
% FOLD_STATION_REPORT  Where can a fold mirror live?
%
%   rep = fold_station_report(t)
%   rep = fold_station_report(t, 'mirror','M3', 'z',linspace(0.05,0.75,8))
%
%   For the leg INTO and the leg OUT OF a named powered mirror (default:
%   the last powered Reflector), scan z-stations and report each bundle's
%   lateral y-interval (from the YZ meridian DRAW fan) and the DAYLIGHT
%   GAP between them.  A fold picking off the OUT leg needs
%
%       gap  >  fold body extent beyond the OUT bundle  (its mount margin)
%
%   at its station, or it also clips the IN leg.  This is the quantitative
%   form of the centered-TMA focal-plane extraction question: the coaxial
%   Korsch's M2->M3 feed and M3->FP return bundles ride the same axis; a
%   FIELD BIAS (t.set_field_bias) separates them laterally near the exit
%   pupil, and this report says where (and whether) a fold of a given size
%   fits.  Run it on the UNFOLDED design (stations parameterize global z);
%   after choosing a station, insert the fold with t.add_fold and verify
%   with t.check_clipping.
%
%   Options:
%     'mirror'  name of the mirror whose in/out legs are scanned
%               (default: the last powered Reflector in the chain)
%     'z'       station list (global z, m); default 12 stations across
%               the z-overlap of the two legs
%     'quiet'   suppress the table (default false)
%     'noload'  skip build/reload when the design is already loaded
%
%   Returns struct array, one row per station:
%     .z                 station (global z, m)
%     .c_in  .hw_in      feed-leg bundle center + halfwidth (y, m)
%     .c_out .hw_out     return-leg bundle center + halfwidth (y, m)
%     .gap               daylight between the intervals (m; >0 = a fold
%                        fits between them)
%
%   See also macos.design.Telescope/add_fold, set_hole, check_clipping.

arguments
    t (1,1) macos.design.Telescope
    opts.mirror (1,:) char = ''
    opts.z      (1,:) double = []
    opts.quiet  (1,1) logical = false
    opts.noload (1,1) logical = false
end

if ~opts.noload
    if ~macos.has_rx(), t.build(); else, t.build('','init',false); end
end
e  = t.spec.elt;  nE = numel(e);

% the mirror whose legs we scan
if isempty(opts.mirror)
    m = find(arrayfun(@(x) strcmp(x.kind,'Reflector') && abs(x.Kr) < 1e21, e), ...
             1, 'last');
    if isempty(m)
        error('macos:design:fold_station_report:noMirror', ...
              'no powered Reflector in the chain.');
    end
else
    m = find(strcmp({e.name}, opts.mirror), 1);
    if isempty(m)
        error('macos:design:fold_station_report:mirror', ...
              'mirror ''%s'' not found.', opts.mirror);
    end
end
if m < 2 || m >= nE
    error('macos:design:fold_station_report:legs', ...
          '''%s'' needs both an IN leg and an OUT leg.', e(m).name);
end

% YZ meridian fan -> polyline segments of the two legs
byz = macos.draw_rays('YZ', 0, nE);
segin = zeros(0,4);  segout = zeros(0,4);          % [z1 y1 z2 y2]
for r = 1:byz.nray
    for i = 1:byz.nper(r)-1
        ea = byz.elt(i,r);  eb = byz.elt(i+1,r);
        s  = [byz.U(i,r) byz.V(i,r) byz.U(i+1,r) byz.V(i+1,r)];
        if ea == m-1 && eb == m,   segin(end+1,:)  = s; end            %#ok<AGROW>
        if ea == m   && eb == m+1, segout(end+1,:) = s; end            %#ok<AGROW>
    end
end
if isempty(segin) || isempty(segout)
    error('macos:design:fold_station_report:trace', ...
          'DRAW fan has no rays on the %s legs -- is the design loaded?', ...
          e(m).name);
end

% default stations: across the z-overlap of the two legs
zz = opts.z;
if isempty(zz)
    zin  = [min(segin(:,[1 3]),[],'all'),  max(segin(:,[1 3]),[],'all')];
    zout = [min(segout(:,[1 3]),[],'all'), max(segout(:,[1 3]),[],'all')];
    lo = max(zin(1), zout(1));  hi = min(zin(2), zout(2));
    if ~(hi > lo)
        error('macos:design:fold_station_report:overlap', ...
              'the two legs share no z-range (already folded?).');
    end
    pad = 0.02*(hi-lo);
    zz  = linspace(lo+pad, hi-pad, 12);
end

rep = struct('z',{},'c_in',{},'hw_in',{},'c_out',{},'hw_out',{},'gap',{});
for j = 1:numel(zz)
    yi = ycross_(segin,  zz(j));
    yo = ycross_(segout, zz(j));
    if isempty(yi) || isempty(yo), continue; end
    ci = 0.5*(min(yi)+max(yi));  hi_ = 0.5*(max(yi)-min(yi));
    co = 0.5*(min(yo)+max(yo));  ho_ = 0.5*(max(yo)-min(yo));
    g  = max(min(yi)-max(yo), min(yo)-max(yi));
    rep(end+1) = struct('z',zz(j), 'c_in',ci, 'hw_in',hi_, ...
                        'c_out',co, 'hw_out',ho_, 'gap',g); %#ok<AGROW>
end

if ~opts.quiet
    fprintf('fold_station_report  (legs %s->%s and %s->%s)\n', ...
            e(m-1).name, e(m).name, e(m).name, e(m+1).name);
    fprintf('  %8s | %9s %8s | %9s %8s | %9s\n', ...
            'z (m)','feed ctr','halfw','ret ctr','halfw','gap');
    for j = 1:numel(rep)
        mark = '';  if rep(j).gap > 0, mark = '  <- fold fits'; end
        fprintf('  %8.3f | %9.4f %8.4f | %9.4f %8.4f | %9.4f%s\n', ...
                rep(j).z, rep(j).c_in, rep(j).hw_in, ...
                rep(j).c_out, rep(j).hw_out, rep(j).gap, mark);
    end
    if isempty(rep)
        fprintf('  (no station where both legs cross -- widen ''z'')\n');
    elseif all([rep.gap] <= 0)
        fprintf(['  => NO daylight at any station: increase the field bias ' ...
                 'and/or move %s back (longer spacing pulls the exit pupil ' ...
                 'behind the primary).\n'], e(m).name);
    end
end
end

function yy = ycross_(seg, z0)
    yy = [];
    for s = 1:size(seg,1)
        z1 = seg(s,1); y1 = seg(s,2); z2 = seg(s,3); y2 = seg(s,4);
        if (z0-z1)*(z0-z2) < 0
            yy(end+1) = y1 + (z0-z1)/(z2-z1)*(y2-y1); %#ok<AGROW>
        end
    end
end
