function varargout = jac_fingerprint(action, varargin)
%JAC_FINGERPRINT  Small committed sidecar for a large derived .mat product.
%   Derived binary products >= ~20 MB (the sensitivity Jacobian
%   s4_jacobians.mat, the MET products e2e_*_met.mat) are NOT committed
%   (see mmacos/.gitignore); a small <file>.fp.json fingerprint is
%   committed in their place so the one-attributable-cause regen
%   discipline stays auditable without the blob.  The regen scripts WRITE
%   the fingerprint; tests and stale-generation checks COMPARE a
%   locally-rebuilt product against the committed fingerprint.
%
%   fp = jac_fingerprint('build', S, meta)
%       Build a fingerprint struct from a loaded product struct S (the
%       variables saved in the .mat -- e.g. struct('ox',ox,'oz',oz,...))
%       and a meta struct of provenance stamps (reset_xp, upstream commit,
%       rx path, delta/method, ...).  Captures, per numeric array field of
%       S: size, a per-column 2-norm vector (downsampled to <=64 samples),
%       and a scalar frobenius norm; copies meta verbatim.
%
%   jac_fingerprint('write', fp_path, S, meta)
%       Build + write the fingerprint to fp_path as JSON (jsonencode).
%
%   fp = jac_fingerprint('read', fp_path)
%       Read a committed fingerprint back (jsondecode).
%
%   [ok, report] = jac_fingerprint('check', S, fp_path, tol)
%       Compare a freshly-built product S against the committed
%       fingerprint at fp_path.  ok=false with a human-readable report of
%       the first mismatching field/metric when dims differ or any
%       captured norm is off by more than tol (relative; default 1e-6).
%       Use in asset-gated tests when the blob is absent but the
%       fingerprint is present.
%
%   Design: the fingerprint is O(fields x 64) doubles + the meta struct --
%   a few KB of JSON, diffable and reviewable, carrying enough to detect a
%   generation change (a reset_xp flip shifts the column norms) without
%   the 60 MB matrix.

switch action
    case 'build'
        [S, meta] = varargin{1:2};
        varargout{1} = build_(S, meta);

    case 'write'
        [fp_path, S, meta] = varargin{1:3};
        fp = build_(S, meta);
        fid = fopen(fp_path, 'w');
        if fid < 0, error('jac_fingerprint:write', 'cannot open %s', fp_path); end
        c = onCleanup(@() fclose(fid));
        fwrite(fid, jsonencode(fp, 'PrettyPrint', true));
        varargout{1} = fp;

    case 'read'
        fp_path = varargin{1};
        assert(isfile(fp_path), 'jac_fingerprint: %s not found', fp_path);
        varargout{1} = jsondecode(fileread(fp_path));

    case 'check'
        S = varargin{1};  fp_path = varargin{2};
        tol = 1e-6;  if numel(varargin) >= 3 && ~isempty(varargin{3}), tol = varargin{3}; end
        [ok, report] = check_(S, fp_path, tol);
        varargout{1} = ok;  if nargout >= 2, varargout{2} = report; end

    otherwise
        error('jac_fingerprint:action', 'unknown action ''%s''', action);
end
end


% ---------------------------------------------------------------------
function fp = build_(S, meta)
fp = struct();
fp.meta   = meta;
fp.fields = struct([]);
fn = fieldnames(S);
for i = 1:numel(fn)
    v = S.(fn{i});
    if ~isnumeric(v) || isempty(v), continue; end
    v = double(v);
    % complex arrays (e.g. the EFC Jacobian G): fingerprint |v| -- the
    % column 2-norms are then the true complex column norms, and
    % jsonencode (which cannot encode complex) stays happy
    if ~isreal(v), v = abs(v); end
    e = struct();
    e.name    = fn{i};
    e.size    = size(v);
    e.fro     = norm(v(:));
    % per-column 2-norms, downsampled to <=64 anchor columns
    if ismatrix(v)
        cn = sqrt(sum(v.^2, 1));
    else
        cn = sqrt(sum(reshape(v, size(v,1), []).^2, 1));
    end
    e.ncol    = numel(cn);
    idx       = unique(round(linspace(1, numel(cn), min(64, numel(cn)))));
    e.col_idx = idx;
    e.col_nrm = cn(idx);
    fp.fields = [fp.fields, e]; %#ok<AGROW>
end
end


% ---------------------------------------------------------------------
function [ok, report] = check_(S, fp_path, tol)
ok = true;  report = 'match';
fp = jsondecode(fileread(fp_path));
flds = fp.fields;   % struct array (same-shape entries) or cell (ragged)
n = numel(flds);
for i = 1:n
    if iscell(flds), e = flds{i}; else, e = flds(i); end
    nm = e.name;
    if ~isfield(S, nm) || ~isnumeric(S.(nm))
        ok = false;  report = sprintf('field %s missing in rebuilt product', nm);  return;
    end
    v = double(S.(nm));
    if ~isequal(size(v), e.size(:).')
        ok = false;
        report = sprintf('field %s size [%s] != fingerprint [%s]', nm, ...
            num2str(size(v)), num2str(e.size(:).'));  return;
    end
    fro = norm(v(:));
    if abs(fro - e.fro) > tol * max(1, abs(e.fro))
        ok = false;
        report = sprintf('field %s frobenius %.6g != fingerprint %.6g (tol %g)', ...
            nm, fro, e.fro, tol);  return;
    end
    cn = sqrt(sum(reshape(v, size(v,1), []).^2, 1));
    idx = e.col_idx(:).';
    got = cn(idx);
    exp = e.col_nrm(:).';
    d = max(abs(got - exp) ./ max(1, abs(exp)));
    if d > tol
        ok = false;
        report = sprintf('field %s per-column norms off by %.3e (> tol %g)', nm, d, tol);
        return;
    end
end
end
