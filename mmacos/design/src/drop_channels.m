function out = drop_channels(out, drop_elts, say)
%DROP_CHANNELS  Remove an element's channel columns from a dw_d*_multi harvest.
%   out = drop_channels(OUT, DROP_ELTS) returns OUT with every channel that
%   belongs to an element in DROP_ELTS removed -- the channel-specific and
%   generic Jacobians (dwd*all), channel_names, and the per-channel iElt /
%   map_idx / zmodes arrays are all pruned CONSISTENTLY, so the result is a
%   smaller-but-well-formed harvest.  Row-indexed data (w0_stacked, indxall,
%   OPDall, per_field_*) is untouched: dropping an element removes COLUMNS
%   (state-vector entries), not observations.
%
%   Use it to drop an optic the zero-norm flag identified as dead -- e.g.
%   the obscured virtual centre segment -- number-free, keyed on the
%   element id the flag reports, not on a literal written into the driver.
%
%   DROP_ELTS  vector of element ids to remove ([] = no-op).
%   SAY        optional fprintf-like handle for a one-line note (default
%              command window); pass @()[] to silence.
%
%   See also: flag_zero_norm_channels, save_dw_flat, macos.dw_dx_multi.

if nargin < 3 || isempty(say), say = @(varargin) fprintf(1, varargin{:}); end
if isempty(drop_elts), return; end
if ~isfield(out, 'channel_names') || isempty(out.channel_names), return; end

cn = out.channel_names;
elt = zeros(numel(cn), 1);
for k = 1:numel(cn)
    t = regexp(cn{k}, '^Elt\s+(\d+)', 'tokens', 'once');
    if ~isempty(t), elt(k) = str2double(t{1}); end
end
keep = ~ismember(elt, drop_elts(:).');
if all(keep), return; end

% column-wise fields (one entry / row per CHANNEL)
for f = {'dwdxall','dwdzall','dwdsall','dwdgall'}
    if isfield(out, f{1}) && size(out.(f{1}), 2) == numel(cn)
        out.(f{1}) = out.(f{1})(:, keep);
    end
end
for f = {'channel_names','iElt','map_idx','zmodes'}
    if isfield(out, f{1}) && numel(out.(f{1})) == numel(cn)
        v = out.(f{1});  out.(f{1}) = v(keep);
    end
end
say('drop_channels: removed %d channels for element(s) %s\n', ...
    nnz(~keep), mat2str(intersect(unique(elt).', drop_elts(:).')));
end
