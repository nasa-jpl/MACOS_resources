function T = suggest_groups(M, varargin)
%SUGGEST_GROUPS  Auto-propose the default budget group table.
%   T = macos.budget.suggest_groups(M) partitions the assembled Jacobian
%   M (macos.budget.assemble) along the standard budget axes -- element
%   role x rigid-body DOF type (tilt/clock/decenter/despace) x figure mode
%   band (Z4-10 / Z11-21 / Z22-45) x surface param (Kr/Kc) x grid -- into
%   a canonical group table the user then edits (sigmas / ranges / dist).
%   Sigmas are placeholders (1 in the default unit); EDIT THEM.
%
%   Options:
%     'roles'   containers.Map role -> iElt vector.  Without it, each
%               distinct element becomes its own role 'Elt<k>'.
%     'bands'   figure mode bands, default {[4 10],[11 21],[22 45]}
%               labelled Low/Mid/High.
%
%   See also macos.budget.read_spec, macos.budget.spec_from_table.

p = inputParser;
p.addParameter('roles', containers.Map('KeyType','char','ValueType','any'));
p.addParameter('bands', {[4 10], [11 21], [22 45]});
p.parse(varargin{:});
roles = p.Results.roles;
bands = p.Results.bands;
bandnm = {'Low', 'Mid', 'High'};

meta = M.meta;
if roles.Count == 0
    el = unique(meta.iElt(~isnan(meta.iElt) & meta.iElt > 0));
    for e = el(:)'
        roles(sprintf('Elt%d', e)) = e;
    end
end
rk = roles.keys;

rows = {};  % each: {Name Elements DOFs Modes Param Channel Dist Sigma Units Min Max Cov Control}
dofgrp = {'Tilt', 'tilt', 0; 'Clock', 'clock', 1; 'Decenter', 'decenter', 1; ...
          'Despace', 'despace', 1};
for ri = 1:numel(rk)
    role = rk{ri};
    ev = roles(role);
    inrole = ismember(meta.iElt, ev);
    % rigid-body groups
    for d = 1:size(dofgrp, 1)
        dd = resolve_dof_(dofgrp{d, 2});
        has = inrole & strcmp(meta.channel, 'dwdx') & ismember(meta.dof, dd);
        if any(has)
            u = 'nrad'; if dofgrp{d, 3} == 0, u = 'nrad'; else, u = 'nm'; end
            if dofgrp{d, 3} == 0, u = 'nrad'; end
            rows(end+1, :) = {sprintf('%s %s', role, dofgrp{d,1}), role, ...
                dofgrp{d,2}, '', '', 'dwdx', 'normal', 1, u, NaN, NaN, 'iid', ''}; %#ok<AGROW>
        end
    end
    % figure bands (dwdz)
    for b = 1:numel(bands)
        bb = bands{b}(1):bands{b}(2);
        has = inrole & strcmp(meta.channel, 'dwdz') & ismember(meta.mode, bb);
        if any(has)
            rows(end+1, :) = {sprintf('%s %s Figure', role, bandnm{min(b,end)}), ...
                role, '', sprintf('%d-%d', bands{b}(1), bands{b}(2)), '', ...
                'dwdz', 'normal', 1, 'nm', NaN, NaN, 'iid', ''}; %#ok<AGROW>
        end
    end
    % surface Kr/Kc (dwdsurf)
    for prm = {'Kr', 'Kc'}
        has = inrole & strcmp(meta.channel, 'dwdsurf') & strcmpi(meta.param, prm{1});
        if any(has)
            u = 'nm'; if strcmpi(prm{1}, 'Kc'), u = ''; end
            rows(end+1, :) = {sprintf('%s %s', role, prm{1}), role, '', '', ...
                prm{1}, 'dwdsurf', 'normal', 1, u, NaN, NaN, 'iid', ''}; %#ok<AGROW>
        end
    end
    % grid (dwdgrid)
    has = inrole & strcmp(meta.channel, 'dwdgrid');
    if any(has)
        rows(end+1, :) = {sprintf('%s Grid Figure', role), role, '', '', '', ...
            'dwdgrid', 'normal', 1, 'nm', NaN, NaN, 'iid', ''}; %#ok<AGROW>
    end
end

vn = {'Name','Elements','DOFs','Modes','Param','Channel','Dist', ...
    'Sigma','Units','Min','Max','Cov','Control'};
if isempty(rows)
    T = cell2table(cell(0, numel(vn)), 'VariableNames', vn);
else
    T = cell2table(rows, 'VariableNames', vn);
end
end

function dd = resolve_dof_(name)
map = struct('tilt', [0 1], 'clock', 2, 'decenter', [3 4], 'despace', 5);
dd = map.(name);
end
