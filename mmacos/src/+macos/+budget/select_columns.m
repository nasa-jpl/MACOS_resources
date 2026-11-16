function mask = select_columns(meta, selector)
%SELECT_COLUMNS  Logical column mask for a budget group's selector.
%   MASK = macos.budget.select_columns(META, SELECTOR) returns an
%   Nz-by-1 logical over the unified Jacobian column metadata META
%   (one entry per column; see macos.budget.assemble) selecting the
%   columns a budget group owns.  The result is the logical AND of
%   every predicate present in SELECTOR -- an absent field is not
%   constrained.
%
%   META fields (array per column, length Nz):
%       channel  cellstr   'dwdx'|'dwdz'|'dwdsurf'|'dwdgrid'
%       iElt     double    element id (0 for Source/Group)
%       dof      double    rigid-body DOF 0..5 [Rx Ry Rz Tx Ty Tz], NaN else
%       kind     cellstr   'Source'|'RigidBody'|'Group'|'FocalPlane'|
%                          'MonZern'|'Zern'|'Surf'|'Grid'
%       mode     double    Noll mode (figure channels), NaN else
%       param    cellstr   'Kr'|'Kc' (surf), '' else
%       map      double    grid influence-map index, NaN else
%       name     cellstr   channel_names, e.g. 'Elt 3 Ry'
%
%   SELECTOR fields (any subset):
%       channel  cellstr   match META.channel
%       kind     cellstr   match META.kind
%       iElt     vector    element ids
%       dof      vector    DOF indices 0..5 (OR names via resolve upstream)
%       mode     vector    Noll modes
%       param    cellstr   'Kr'/'Kc'
%       map      vector    grid map indices
%       name_rx  char      regexp over META.name (fallback)
%
%   See also macos.budget.assemble, macos.budget.spec_from_table.

nz = numel(meta.channel);
mask = true(nz, 1);

if isfield(selector, 'channel') && ~isempty(selector.channel)
    mask = mask & ismember(meta.channel(:), cellstr(selector.channel));
end
if isfield(selector, 'kind') && ~isempty(selector.kind)
    mask = mask & ismember(meta.kind(:), cellstr(selector.kind));
end
if isfield(selector, 'iElt') && ~isempty(selector.iElt)
    mask = mask & ismember(meta.iElt(:), selector.iElt(:));
end
if isfield(selector, 'dof') && ~isempty(selector.dof)
    mask = mask & ismember(meta.dof(:), selector.dof(:));
end
if isfield(selector, 'mode') && ~isempty(selector.mode)
    mask = mask & ismember(meta.mode(:), selector.mode(:));
end
if isfield(selector, 'param') && ~isempty(selector.param)
    mask = mask & ismember(meta.param(:), cellstr(selector.param));
end
if isfield(selector, 'map') && ~isempty(selector.map)
    mask = mask & ismember(meta.map(:), selector.map(:));
end
if isfield(selector, 'name_rx') && ~isempty(selector.name_rx)
    hit = ~cellfun(@isempty, regexp(meta.name(:), selector.name_rx, 'once'));
    mask = mask & hit;
end
end
