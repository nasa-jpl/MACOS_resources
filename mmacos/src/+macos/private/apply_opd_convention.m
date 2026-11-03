function out = apply_opd_convention(out, orient, sgn)
%APPLY_OPD_CONVENTION  Apply 'orient'/'sign' options to a dw_d* result.
%   Shared by the dw_d* drivers; conventions documented in
%   mmacos/doc/opd_conventions.md.
%
%   'orient'='xy': remap the linear pixel indices (indx / indxall) to
%   the transposed grid and transpose every 2-D map.  Jacobian ROW
%   ORDER is unchanged -- each row keeps its physical pixel; only the
%   index values change so unflattening lands on the transposed grid.
%   'sign'='wavefront': negate every wavefront-valued output
%   (Jacobians and nominal wavefronts; centroid/spot outputs are not
%   wavefronts and are untouched).
%
%   Records what was applied in out.opd_orient / out.opd_sign.

% grid size from whichever nominal map exists
N = [];
if isfield(out,'w_nom_2d') && ~isempty(out.w_nom_2d)
    N = size(out.w_nom_2d);
elseif isfield(out,'per_field_w_nom_2d') && ~isempty(out.per_field_w_nom_2d)
    m = out.per_field_w_nom_2d;
    if iscell(m), m = m{1}; end
    N = size(m);
end

if strcmp(orient,'xy')
    for fn = {'indx','indxall'}
        f = fn{1};
        if isfield(out,f)
            if iscell(out.(f))
                out.(f) = cellfun(@tr_indx, out.(f), 'UniformOutput', false);
            else
                out.(f) = tr_indx(out.(f));
            end
        end
    end
    for fn = {'w_nom_2d','per_field_w_nom_2d'}
        f = fn{1};
        if isfield(out,f)
            if iscell(out.(f))
                out.(f) = cellfun(@(M) M.', out.(f), 'UniformOutput', false);
            elseif ndims(out.(f)) == 3
                out.(f) = permute(out.(f), [2 1 3]);
            else
                out.(f) = out.(f).';
            end
        end
    end
end

if strcmp(sgn,'wavefront')
    wf_fields = {'dwdx','dwdg','dwds','dwdz', ...
                 'dwdxall','dwdgall','dwdsall','dwdzall', ...
                 'per_field_dwdx','per_field_dwdg','per_field_dwds', ...
                 'per_field_dwdz','w_nom_2d','w_nom_vec','w0_stacked', ...
                 'per_field_w_nom_2d'};
    for fn = wf_fields
        f = fn{1};
        if isfield(out,f)
            if iscell(out.(f))
                out.(f) = cellfun(@(M) -M, out.(f), 'UniformOutput', false);
            else
                out.(f) = -out.(f);
            end
        end
    end
end

out.opd_orient = orient;
out.opd_sign   = sgn;
end

function ix2 = tr_indx(ix)
    % m2v bookkeeping struct (.i/.j/.size): swap the subscripts.
    % (Numeric linear indices are not used by the dw_d* drivers.)
    ix2 = ix;
    if isstruct(ix)
        ix2.i    = ix.j;
        ix2.j    = ix.i;
        ix2.size = fliplr(ix.size);
    end
end
