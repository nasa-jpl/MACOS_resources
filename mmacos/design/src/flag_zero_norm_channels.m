function dead = flag_zero_norm_channels(out, say)
%FLAG_ZERO_NORM_CHANNELS  Warn on per-element channel groups that are all zero.
%   dead = flag_zero_norm_channels(OUT) inspects a dw_d*_multi harvest and
%   NOISILY flags every element whose entire channel block is ~zero -- an
%   optic that contributes NO sensitivity (obscured, vignetted, or a virtual
%   reference element that only passes the chief ray).  This is the
%   number-free way to catch a dead optic: it keys on the RESPONSE, not on an
%   element id, so it adapts to any deck and any promotion (Dave 2026-08-21).
%
%   A channel's element is read from OUT.channel_names ('Elt N ...'); columns
%   are grouped by element and a group counts as dead when its largest column
%   RMS (over finite rows) is a negligible fraction of the harvest's median
%   live-column RMS.  Returns the dead element ids (empty when all optics
%   respond).  Pass a SAY function handle (fprintf-like) to route the warning
%   into a report; default prints to the command window.
%
%   Typical use in a driver, after the harvest:
%       dead = flag_zero_norm_channels(art.oz);   % or art.og / art.ox / art.os
%   A live optic that comes back dead means it is not being illuminated or
%   its figure channel is not armed -- investigate, do not silently ship it.
%
%   See also: optic_footprints, run_sensitivities, macos.dw_dx_multi.

if nargin < 2 || isempty(say), say = @(varargin) fprintf(1, varargin{:}); end
dead = [];
if isempty(out) || ~isstruct(out), return; end

% the channel-specific Jacobian, else the generic alias
if     isfield(out,'dwdgall'), A = out.dwdgall;
elseif isfield(out,'dwdsall'), A = out.dwdsall;
elseif isfield(out,'dwdzall'), A = out.dwdzall;
elseif isfield(out,'dwdxall'), A = out.dwdxall;
else, return; end
cn = out.channel_names;
if isempty(A) || isempty(cn), return; end

fin = all(isfinite(A), 2);           % rows every channel reached
if ~any(fin), return; end
colrms = sqrt(mean(A(fin, :).^2, 1));            % per-channel RMS
elt = zeros(1, numel(cn));
for k = 1:numel(cn)
    t = regexp(cn{k}, '^Elt\s+(\d+)', 'tokens', 'once');
    if ~isempty(t), elt(k) = str2double(t{1}); end
end
uelt = unique(elt(elt > 0));
if isempty(uelt), return; end

% per-element block RMS (max column in the group), and the scale to compare
% against: the median of the live blocks
blk = zeros(1, numel(uelt));
for i = 1:numel(uelt)
    blk(i) = max(colrms(elt == uelt(i)));
end
scale = median(blk(blk > 0));
if isempty(scale) || ~isfinite(scale) || scale == 0, return; end
thr = 1e-6 * scale;                  % six decades below the typical live optic

for i = 1:numel(uelt)
    if blk(i) <= thr
        dead(end+1) = uelt(i); %#ok<AGROW>
        say(['WARNING: element %d contributes all-zero sensitivity ' ...
             '(block RMS %.3e vs median live %.3e) -- obscured / vignetted / ' ...
             'virtual?  Exclude it or check the model.\n'], ...
             uelt(i), blk(i), scale);
    end
end
end
