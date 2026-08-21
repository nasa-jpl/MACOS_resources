function varargout = config_axis(action, varargin)
%CONFIG_AXIS  Shared CONFIGURATION-axis machinery for the dw_d*_multi family.
%   A CONFIGURATION is a named set of element setting overrides -- a zoom
%   position in the classical sense, more often a COMPENSATION state (the
%   j18-family steering mirror at a pupil fold re-pointed to cancel
%   pointing drift).  The four supervisors evaluate their Jacobian per
%   (configuration, field) block from ONE call; this helper owns the
%   validate / snapshot / apply / undo / assert cycle so the rules live in
%   ONE place instead of in every supervisor and every user driver.
%
%   Design + rationale: design/PLAN_CONFIGURATIONS.md.
%
%   cfgs = config_axis('validate', configs, nElt, caller, ep_elt)
%       Normalise and validate a configuration struct array BEFORE
%       anything is applied.  Returns a 1xNc struct with fields
%         .name  char
%         .set   1xK cell of normalised setter records (struct: fn, elt,
%                and the per-setter payload)
%         .elts  sorted unique element ids the list touches
%       [] / empty in -> 1x0 struct out (the supervisors then take the
%       single-block path, which is byte-identical to the pre-configs
%       call).  ep_elt (pass [] to disable the check) is the exit-pupil
%       element nElt-1: a configuration may not touch it, because the
%       per-field reset_xp legitimately rewrites it and it is carried in
%       the RUN-level snapshot instead (PLAN §2.1).
%
%   snap = config_axis('snapshot', session, elts)
%       Record every pose quantity the veneer can READ for each element:
%       vpt / psi / rpt, the output frame TElt (csys + its two flags),
%       the aperture frame axis xObs, and -- when the surface carries one
%       -- the figure frame pMon/xMon/yMon/zMon.
%
%   config_axis('apply', session, cfg)
%       Dispatch the setter list against the Session, then modify() ONCE.
%       The modify()-after-setters rule (a cached trace is stale after a
%       geometry write) is enforced here so drivers never carry it.
%
%   config_axis('undo', session, cfg, snap)
%       Reverse the list, then modify() once.  See RESTORE MECHANISM.
%
%   drift = config_axis('assert', session, snap, cfg_name, caller)
%       Re-read the snapshot quantities and HARD ERROR on any mismatch,
%       naming the element and the quantity.  This is the load-bearing
%       part of the design: a per-configuration Jacobian silently computed
%       from the PREVIOUS configuration's geometry is the failure mode
%       that would be hardest to notice.  Never weaken it to a warning.
%       Returns the worst observed drift so the caller can PRINT it every
%       run -- a drift that is growing run over run is visible long before
%       it trips the tolerance.
%
%   WHAT THE TOLERANCE IS MEASURING -- and why it is not ULP-tight
%   -------------------------------------------------------------
%   The floor is NOT set by this restore.  The configuration element is
%   typically ALSO a Jacobian channel (that is what a zoom-dependent
%   sensitivity IS), so between the apply and the assert the channel loop
%   has run its own poke/restore cycle on that element once per DOF per
%   field -- 60 round trips for a 6-DOF element over 5 fields -- each
%   leaving a few ULP behind.  Measured on e5hex1 (segment element, 12
%   channels x 5 fields): 1.7e-12 in the vertex, with the CONFIGURATION's
%   own inverse exact.  A ULP-tight tolerance therefore reports the
%   channel loop's round-off as a configuration failure.
%   The tolerance is set instead by what it has to CATCH: a setter whose
%   effect the restore missed leaves a residual the size of the
%   CONFIGURATION (an FSM tilt is 1e-4 rad; a zoom translation is
%   millimetres), which is many decades above round-off.  RTOL 1e-9 keeps
%   three-plus decades of margin over the observed floor and still fires
%   on anything that would bias the next block.
%
%   V1 SETTER WHITELIST
%   -------------------
%   perturb, set_elt_vpt, set_elt_psi, set_elt_rpt, set_elt_csys -- and
%   nothing else.  The Session surface also carries set_elt_kr/kc,
%   set_elt_zrn_coef, set_elt_grid, the grating setters and the set_src_*
%   family; a configuration invoking one of those would apply cleanly and
%   then RESTORE SILENTLY WRONG, which is exactly the contamination the
%   assertion exists to prevent.  Extending the axis to them means
%   extending the snapshot per state category first.  Anything else in a
%   configuration list is a loud error at validation time, before apply.
%
%   RESTORE MECHANISM -- and why it is not snapshot-write-back alone
%   ---------------------------------------------------------------
%   PLAN §2 specifies restore-by-snapshot.  That is exact for the three
%   ABSOLUTE setters (elt_vpt/elt_psi/elt_rpt in macos_api_mod.F90 are
%   plain array writes with no auxiliary bookkeeping) and for set_elt_csys
%   (write back, or rm_elt_csys when no local frame was defined), so those
%   ARE restored from the snapshot.
%
%   It is NOT sufficient for `perturb`.  CPERTURB_PROG (funcsub.F) moves
%   far more than vpt/psi/rpt: the aperture axis xObs, the figure frames
%   pMon/pData/pFF, the chief-ray nominal incidence CRIncidPosNom, the
%   metrology surface points, the HOE points, and any linked children.
%   The veneer can READ only some of those and WRITE fewer still, so a
%   write-back of vpt/psi/rpt would leave a rotated xObs and a shifted
%   figure frame behind -- and the assertion as originally scoped would
%   not have caught it, because the quantities it re-reads restore fine.
%
%   So `perturb` is undone by the engine's own inverse: a second perturb
%   with the rotation and translation negated, same frame.  That is EXACT,
%   not approximate, and the reasons are worth recording because "an
%   inverse perturb is easy to get subtly wrong" is the right prior:
%     * Qform (mathsub.F) is the exact Rodrigues rotation about th/|th|,
%       so Q(-th) == Q(th)' == Q(th)^-1 -- not a small-angle or Euler
%       -sequence approximation, which would NOT invert for a rotation
%       about two axes at once (the fixture's [+-0.5', +-0.5', 0] case).
%     * The pivot algebra composes to the identity: forward is
%       V' = Rpt + Q(V-Rpt) + del with Rpt' = Rpt+del, so the negated call
%       gives V'' = Rpt' + Q^-1(V'-Rpt') - del = V exactly.
%     * In LOCAL frame the rotation axis in global coordinates is an
%       eigenvector of its own Q, so the post-perturb TElt maps -th_local
%       to exactly -th_global.
%     * It is the mechanism macos.channels.RigidBodyChannel.restore
%       already relies on for every column of every Jacobian in this tree.
%   The one residual: a perturb carrying BOTH a rotation and a translation
%   in LOCAL frame on an element whose TElt tracks perturbations
%   (LUpdateTElt_FLG) inverts its translation through the ROTATED TElt,
%   leaving O(|theta|*|del|) behind.  The assertion catches it; split such
%   a configuration into a rotation entry and a translation entry.
%
%   The snapshot is retained -- and widened to xObs and the figure frame
%   -- as the VERIFIER.  Restore writes, snapshot checks.
%
%   See also: macos.dw_dx_multi, macos.design.configs_from_table.

switch lower(action)
    case 'validate', [varargout{1:max(nargout,1)}] = do_validate(varargin{:});
    case 'snapshot', [varargout{1:max(nargout,1)}] = do_snapshot(varargin{:});
    case 'apply',    do_apply(varargin{:});    varargout = {};
    case 'undo',     do_undo(varargin{:});     varargout = {};
    case 'assert',   [varargout{1:max(nargout,1)}] = do_assert(varargin{:});
    otherwise
        error('macos:config_axis:action', ...
            'unknown config_axis action ''%s''', action);
end
end


% =====================================================================
function names = whitelist()
names = {'perturb', 'set_elt_vpt', 'set_elt_psi', 'set_elt_rpt', ...
         'set_elt_csys'};
end


% =====================================================================
function cfgs = do_validate(configs, nElt, caller, ep_elt)
if nargin < 4, ep_elt = []; end
proto = struct('name', {}, 'set', {}, 'elts', {}, 'raw', {});
if isempty(configs)
    cfgs = proto;
    return
end
if ~isstruct(configs)
    error(sprintf('macos:%s:configs', caller), ...
        ['''configs'' must be a struct array with fields ''name'' and ' ...
         '''set'' (see macos.design.configs_from_table); got %s'], ...
        class(configs));
end
if ~all(isfield(configs, {'name', 'set'}))
    error(sprintf('macos:%s:configs', caller), ...
        '''configs'' struct needs both a ''name'' and a ''set'' field');
end

cfgs = proto;
seen = {};
for c = 1:numel(configs)
    nm = configs(c).name;
    if isstring(nm), nm = char(nm); end
    if ~ischar(nm) || isempty(strtrim(nm))
        error(sprintf('macos:%s:configName', caller), ...
            'configs(%d).name must be a non-empty char/string', c);
    end
    nm = strtrim(nm);
    if any(strcmp(seen, nm))
        error(sprintf('macos:%s:configName', caller), ...
            'duplicate configuration name ''%s'' (configs(%d))', nm, c);
    end
    seen{end+1} = nm; %#ok<AGROW>

    sl = configs(c).set;
    if isempty(sl), sl = {}; end
    if ~iscell(sl)
        error(sprintf('macos:%s:configSet', caller), ...
            ['configs(%d) (''%s''): ''set'' must be a cell array of ' ...
             'setter invocations, each itself a cell {fname, args...}; ' ...
             'got %s'], c, nm, class(sl));
    end
    rec = cell(1, numel(sl));
    elts = zeros(1, numel(sl));
    for k = 1:numel(sl)
        rec{k} = validate_one(sl{k}, c, nm, k, nElt, caller, ep_elt);
        elts(k) = rec{k}.elt;
    end
    c_ = struct();
    c_.name = nm;
    c_.set  = rec;                  % normalised setter records
    c_.elts = unique(elts);
    c_.raw  = sl;                   % the caller's list, verbatim
    cfgs(end+1) = c_; %#ok<AGROW>
end
end


% ---------------------------------------------------------------------
function r = validate_one(entry, c, nm, k, nElt, caller, ep_elt)
where = sprintf('configs(%d) (''%s'') entry %d', c, nm, k);
if ~iscell(entry) || numel(entry) < 2
    error(sprintf('macos:%s:configSet', caller), ...
        '%s: must be a cell {fname, elt, args...}', where);
end
fn = entry{1};
if isstring(fn), fn = char(fn); end
if ~ischar(fn)
    error(sprintf('macos:%s:configSet', caller), ...
        '%s: first element must be the setter NAME (char)', where);
end
if ~any(strcmp(whitelist(), fn))
    error(sprintf('macos:%s:configSetter', caller), ...
        ['%s: ''%s'' is not an accepted v1 configuration setter.\n' ...
         'The v1 whitelist is: %s.\n' ...
         'The snapshot records POSE state only, so any other setter ' ...
         'would apply cleanly and then RESTORE SILENTLY WRONG -- ' ...
         'extending the axis means extending the snapshot per state ' ...
         'category first (design/PLAN_CONFIGURATIONS.md §2).'], ...
        where, fn, strjoin(whitelist(), ', '));
end

e = entry{2};
if ~(isnumeric(e) && isscalar(e) && e == fix(e) && e >= 1)
    error(sprintf('macos:%s:configElt', caller), ...
        '%s: second element must be a positive integer element id', where);
end
e = double(e);
if ~isempty(nElt) && e > nElt
    error(sprintf('macos:%s:configElt', caller), ...
        '%s: element %d is out of range (the Rx has %d elements)', ...
        where, e, nElt);
end
if ~isempty(ep_elt) && e == ep_elt
    error(sprintf('macos:%s:configElt', caller), ...
        ['%s: element %d is the exit-pupil element (nElt-1).  A ' ...
         'configuration may not touch it: the per-field reset_xp ' ...
         'rewrites it legitimately, so it is carried in the RUN-level ' ...
         'snapshot and excluded from the per-configuration assertion ' ...
         '(design/PLAN_CONFIGURATIONS.md §2.1).  Pass ''reset_xp'', ' ...
         'false if you really mean to configure that element.'], where, e);
end

r = struct('fn', fn, 'elt', e);
args = entry(3:end);
switch fn
    case 'perturb'
        p = parse_perturb(args, where, caller);
        r.rotation = p.rotation;
        r.translation = p.translation;
        r.frame = p.frame;
    case {'set_elt_vpt', 'set_elt_psi', 'set_elt_rpt'}
        if numel(args) ~= 1
            error(sprintf('macos:%s:configSet', caller), ...
                '%s: %s takes exactly one 3-vector argument', where, fn);
        end
        v = args{1};
        if ~(isnumeric(v) && numel(v) == 3)
            error(sprintf('macos:%s:configSet', caller), ...
                '%s: %s argument must be a 3-vector', where, fn);
        end
        r.value = double(v(:));
    case 'set_elt_csys'
        if numel(args) < 3
            error(sprintf('macos:%s:configSet', caller), ...
                ['%s: set_elt_csys takes xDir, yDir, zDir ' ...
                 '(+ optional ''update'', TF)'], where);
        end
        for q = 1:3
            if ~(isnumeric(args{q}) && numel(args{q}) == 3)
                error(sprintf('macos:%s:configSet', caller), ...
                    '%s: set_elt_csys axis %d must be a 3-vector', where, q);
            end
        end
        r.axes = [double(args{1}(:)), double(args{2}(:)), double(args{3}(:))];
        r.opts = args(4:end);
end
end


% ---------------------------------------------------------------------
function p = parse_perturb(args, where, caller)
p = struct('rotation', zeros(3,1), 'translation', zeros(3,1), ...
           'frame', 'local');
if mod(numel(args), 2) ~= 0
    error(sprintf('macos:%s:configSet', caller), ...
        '%s: perturb takes name-value pairs after the element id', where);
end
for q = 1:2:numel(args)
    key = args{q};
    if isstring(key), key = char(key); end
    val = args{q+1};
    switch lower(key)
        case 'rotation'
            p.rotation = check3(val, 'rotation', where, caller);
        case 'translation'
            p.translation = check3(val, 'translation', where, caller);
        case 'frame'
            if isstring(val), val = char(val); end
            if ~any(strcmp(val, {'local', 'global'}))
                error(sprintf('macos:%s:configSet', caller), ...
                    '%s: perturb ''frame'' must be ''local'' or ''global''', ...
                    where);
            end
            p.frame = val;
        otherwise
            error(sprintf('macos:%s:configSet', caller), ...
                ['%s: unknown perturb option ''%s'' (accepted: ' ...
                 '''rotation'', ''translation'', ''frame'')'], ...
                where, num2str(key));
    end
end
end

function v = check3(v, nm, where, caller)
if ~(isnumeric(v) && numel(v) == 3)
    error(sprintf('macos:%s:configSet', caller), ...
        '%s: perturb ''%s'' must be a 3-vector', where, nm);
end
v = double(v(:));
end


% =====================================================================
function snap = do_snapshot(session, elts)
elts = unique(double(elts(:))).';
snap = struct('elt', {}, 'vpt', {}, 'psi', {}, 'rpt', {}, ...
              'csys', {}, 'csys_lcs', {}, 'csys_upd', {}, ...
              'x_obs', {}, 'has_srf', {}, 'srf', {});
for e = elts
    s = struct();
    s.elt = e;
    s.vpt = session.get_elt_vpt(e);  s.vpt = s.vpt(:);
    s.psi = session.get_elt_psi(e);  s.psi = s.psi(:);
    s.rpt = session.get_elt_rpt(e);  s.rpt = s.rpt(:);
    cs = session.get_elt_csys(e);
    s.csys = cs.csys(:, :, 1);
    s.csys_lcs = cs.csys_lcs(1);
    s.csys_upd = cs.csys_upd(1);
    % xObs -- rotated by CPERTURB_PROG, readable but NOT writable through
    % the veneer, so it is a check-only quantity.
    s.x_obs = [];
    try
        s.x_obs = reshape(macos.get_elt_info(e).x_obs, 3, 1);
    catch
    end
    % the figure frame exists only on figured surfaces (Monomial(4),
    % Zernike(8), GridData(9) and composites); the getter errors otherwise
    s.has_srf = false;
    s.srf = [];
    try
        s.srf = macos.get_elt_srf_csys(e);
        s.has_srf = true;
    catch
    end
    snap(end+1) = s; %#ok<AGROW>
end
end


% =====================================================================
function do_apply(session, cfg)
for k = 1:numel(cfg.set)
    r = cfg.set{k};
    switch r.fn
        case 'perturb'
            session.perturb(r.elt, 'rotation', r.rotation, ...
                'translation', r.translation, 'frame', r.frame);
        case 'set_elt_vpt',  session.set_elt_vpt(r.elt, r.value);
        case 'set_elt_psi',  session.set_elt_psi(r.elt, r.value);
        case 'set_elt_rpt',  session.set_elt_rpt(r.elt, r.value);
        case 'set_elt_csys'
            session.set_elt_csys(r.elt, r.axes(:,1), r.axes(:,2), ...
                r.axes(:,3), r.opts{:});
    end
end
% ONE modify() for the whole configuration -- the config runner owns the
% modify()-after-setters rule so user drivers never have to.
session.modify();
end


% =====================================================================
function do_undo(session, cfg, snap)
for k = numel(cfg.set):-1:1
    r = cfg.set{k};
    switch r.fn
        case 'perturb'
            % the engine's own exact inverse -- see RESTORE MECHANISM
            session.perturb(r.elt, 'rotation', -r.rotation, ...
                'translation', -r.translation, 'frame', r.frame);
        case 'set_elt_vpt',  session.set_elt_vpt(r.elt, get_snap(snap, r.elt).vpt);
        case 'set_elt_psi',  session.set_elt_psi(r.elt, get_snap(snap, r.elt).psi);
        case 'set_elt_rpt',  session.set_elt_rpt(r.elt, get_snap(snap, r.elt).rpt);
        case 'set_elt_csys'
            s = get_snap(snap, r.elt);
            if s.csys_lcs
                session.set_elt_csys(r.elt, s.csys(1:3,1), s.csys(1:3,2), ...
                    s.csys(1:3,3), 'update', s.csys_upd);
            else
                macos.rm_elt_csys(r.elt);
            end
    end
end
session.modify();
end

function s = get_snap(snap, e)
i = find([snap.elt] == e, 1);
if isempty(i)
    error('macos:config_axis:snapshot', ...
        'element %d is not in the configuration snapshot', e);
end
s = snap(i);
end


% =====================================================================
function drift = do_assert(session, snap, cfg_name, caller)
% Tolerances: the restore paths are exact (a plain array write-back, or
% the engine's own inverse rigid-body operation), so the only admissible
% residual is round-off.  Unit vectors get an absolute floor; positions
% are scaled by their own magnitude (BaseUnits may be mm, so a vertex is
% O(1e4) and an absolute tolerance would be meaningless).
RTOL = 1e-9;        % relative to the quantity's own scale
ATOL = 1e-12;       % absolute floor for near-zero quantities
bad = {};
worst = 0;
for q = 1:numel(snap)
    s = snap(q);
    e = s.elt;
    chk_pos('VptElt', s.vpt, session.get_elt_vpt(e));
    chk_pos('RptElt', s.rpt, session.get_elt_rpt(e));
    chk_unit('PsiElt', s.psi, session.get_elt_psi(e));
    cs = session.get_elt_csys(e);
    chk_unit('TElt', s.csys(:), reshape(cs.csys(:,:,1), [], 1));
    if s.csys_lcs ~= cs.csys_lcs(1)
        bad{end+1} = sprintf('elt %d TElt local-frame flag', e); %#ok<AGROW>
    end
    if s.csys_upd ~= cs.csys_upd(1)
        bad{end+1} = sprintf('elt %d TElt update flag', e); %#ok<AGROW>
    end
    if ~isempty(s.x_obs)
        chk_unit('xObs', s.x_obs, ...
            reshape(macos.get_elt_info(e).x_obs, 3, 1));
    end
    if s.has_srf
        now_ = macos.get_elt_srf_csys(e);
        chk_pos('pMon', s.srf.pMon(:), now_.pMon(:));
        chk_unit('xMon', s.srf.xMon(:), now_.xMon(:));
        chk_unit('yMon', s.srf.yMon(:), now_.yMon(:));
        chk_unit('zMon', s.srf.zMon(:), now_.zMon(:));
    end
end
if ~isempty(bad)
    error(sprintf('macos:%s:configRestore', caller), ...
        ['configuration ''%s'' did not restore: %s.\n' ...
         'The NEXT configuration''s Jacobian would have been computed ' ...
         'from this one''s geometry.  If the entry perturbs a rotation ' ...
         'AND a translation together in the LOCAL frame, split it into ' ...
         'two entries (design/PLAN_CONFIGURATIONS.md §2).'], ...
        cfg_name, strjoin(bad, '; '));
end
% worst drift as a FRACTION of its own tolerance -- one comparable
% number across quantities of different scales
drift = worst;

    function chk_pos(nm, a, b)
        a = a(:); b = b(:);
        d = max(abs(a - b));
        t = max(ATOL, RTOL * max(1, max(abs(a))));
        worst = max(worst, d / t);
        if ~(d <= t)
            bad{end+1} = sprintf('elt %d %s by %.3e (tol %.3e)', ...
                e, nm, d, t);
        end
    end
    function chk_unit(nm, a, b)
        a = a(:); b = b(:);
        d = max(abs(a - b));
        worst = max(worst, d / RTOL);
        if ~(d <= RTOL)
            bad{end+1} = sprintf('elt %d %s by %.3e (tol %.3e)', ...
                e, nm, d, RTOL);
        end
    end
end
