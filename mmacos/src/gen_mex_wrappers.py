#!/usr/bin/env python3
"""
gen_mex_wrappers.py -- one generator, three binding targets, from macos_api_mod.F90.

Parses every public subroutine in macos_api_mod ONCE and emits:
  mex (mmacos):
  - mmacos_gen.F: one do_<name>(nlhs, plhs, nrhs, prhs) helper per routine
                  + a gen_dispatch(cmd, nlhs, plhs, nrhs, prhs, handled)
                  subroutine that the main mexFunction calls after its
                  own hand-written cases.
  - mmacos_gen_cmds.txt: machine-readable command list (one per line),
                  consumed by MacosSession.m and the smoke test.
  f2py (pymacos):
  - ../../pymacos/src/cmake/source/pymacos_f2py.f90: `module api`, one
                  forwarding wrapper per routine (NAME_impl => NAME); the
                  !f2py hints and interface rewrites come from
                  binding_overrides.py (generator INPUT).  New routines get
                  the mex's rule: intent(inout) is read AND returned.
  veneer (pymacos):
  - ../../pymacos/src/pymacos/api_gen.py: one Python stub per routine --
                  f2py's arguments, ok checked (ApiError), the other outputs
                  returned; docstring = the Fortran comment block above the
                  routine.  Behaviour stays hand-written in macos.py.
Generated files are regenerated, never hand-edited.  Targets: --targets
mex,f2py,veneer (default all); --out-dir DIR writes everything into DIR (a
side tree, or a byte-compare against the committed outputs).

Conventions:
  - prhs slots = intent(in) and intent(inout) args, in declaration order
                 (the command-name string is prhs(1), so the first sub
                 arg lands in prhs(2)).
  - plhs slots = intent(out) and intent(inout) args, in declaration order,
                 with `ok` SKIPPED (replaced by mexErrMsgTxt on failure).
  - Setter logical args pass straight through as a real(8) prhs slot,
                 converted to .true. for any nonzero value.
  - Character args: only `load_rx`/`save_rx` use them today, and both
                    are already hand-written in mmacos_mex.F, so codegen
                    SKIPS them.
  - Routines whose name appears in HAND_WRITTEN are skipped (the
                    hand-written helper in mmacos_mex.F still wins).
  - Helper routines (SystemCheck, EltRangeChk, etc.) are skipped via
                    PRIVATE_HELPERS.

Run from MACOS_resources/mmacos/src/ (plain python3; no numpy needed):
  python3 gen_mex_wrappers.py [path/to/macos_api_mod.F90] [--targets ...] [--out-dir DIR]
"""
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# --- Skip lists --------------------------------------------------------

HAND_WRITTEN = {
    'init', 'load_rx', 'save_rx', 'modified_rx', 'n_elt',
    'opd_val', 'int_cmd', 'int_get',
    'cfield_cmd', 'cfield_get', 'cfield_plane_get',  # Path B: folded into
                                          # the 'complex_field' cmd as an
                                          # optional 4th arg (plane)
    'cfield_apodize',  # cfield_apodize_complex removed from skip list
                       # (codegen handles it cleanly; Phase 5 DM tests need it)
    'elt_dx_get', 'base_unit_to_metres',
    'trace_rays', 'prb_elt',
    'rayfield_get',   # Path B: the mex helper queries mdttl internally
                      # (currrent_macos_model_size) so the veneer signature
                      # is macos.ray_field(iElt) -- no N from the caller.
}
# Command-string aliases for the hand-written ones (mexFunction
# dispatch already handles these — listed here so the +macos/ package
# layer sees the full surface).
HAND_WRITTEN_CMDS = {
    'init', 'load_rx', 'save_rx', 'modified_rx', 'n_elt',
    'opd', 'intensity', 'complex_field', 'apodize',
    'dx_at', 'base_unit_to_metres', 'trace_rays', 'prb_elt',
    'elt_csys_get', 'ray_field',
}

PRIVATE_HELPERS = {
    'SystemCheck', 'StatusChk1', 'EltRangeChk',
    'checkSurfaceID', 'checkEltID',
    'translateSurfaceID', 'translateEltID',
    'currrent_macos_model_size',
}

# --- Data model --------------------------------------------------------

@dataclass
class Arg:
    name: str
    type: str        # 'logical' | 'integer' | 'real' | 'character' | '?'
    intent: str      # 'in' | 'out' | 'inout' | '?'
    dims: list = field(default_factory=list)  # list of symbolic dim exprs
    base: str = ''   # declared type spec verbatim, e.g. 'real(8)', 'character(len=*)'
    comment: str = ''  # trailing `!` comment of the declaration line

    @property
    def is_scalar(self):
        return not self.dims

    @property
    def is_ok(self):
        return self.name.lower() == 'ok'

# --- Parser ------------------------------------------------------------

SUB_RE = re.compile(
    # Match across line continuations: `subroutine NAME(arg1, arg2, &\n      &  arg3, ...)`
    r'^\s+subroutine\s+(\w+)\s*\(((?:[^)&]|&\s*\n)*)\)\s*$',
    re.MULTILINE | re.IGNORECASE,
)
END_SUB_RE = re.compile(r'^\s+end\s+subroutine\s+\w+', re.MULTILINE | re.IGNORECASE)

# Match a declaration line like:
#   real(8), dimension(N,M), intent(in) :: name1, name2
#   logical, intent(out) :: ok
#   integer, intent(in) :: n
DECL_RE = re.compile(
    r'''
    ^\s*
    (?P<base>(?:logical|integer|real(?:\s*\(\s*8\s*\))?|character\s*\([^)]+\)))
    \s*,?\s*
    (?P<attrs>(?:(?:dimension\s*\([^)]+\)|intent\s*\([^)]+\))\s*,?\s*)*)
    ::\s*
    (?P<names>[^!\n]+?)
    \s*(?:!(?P<comment>.*))?$
    ''',
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

DIM_RE = re.compile(r'dimension\s*\(([^)]+)\)', re.IGNORECASE)
INTENT_RE = re.compile(r'intent\s*\(([^)]+)\)', re.IGNORECASE)


def _split_namelist(text):
    """Split a Fortran comma list at top-level commas only.

    Used for both the names list in a declaration and per-name dim
    suffixes like ``arr(max(n,1))`` or ``mat(N,M)`` -- a naive
    str.split(',') would shred those.  Returns a list of stripped
    non-empty pieces.
    """
    parts = []
    depth = 0
    buf = []
    for ch in text:
        if ch == '(':
            depth += 1
            buf.append(ch)
        elif ch == ')':
            depth -= 1
            buf.append(ch)
        elif ch == ',' and depth == 0:
            tok = ''.join(buf).strip()
            if tok:
                parts.append(tok)
            buf = []
        else:
            buf.append(ch)
    tail = ''.join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def parse_type(base):
    b = base.strip().lower()
    if b.startswith('logical'):
        return 'logical'
    if b.startswith('integer'):
        return 'integer'
    if b.startswith('real'):
        return 'real'
    if b.startswith('character'):
        return 'character'
    return '?'


LOCAL_PARAM_RE = re.compile(
    r'^\s*integer\s*,\s*parameter\s*::\s*(\w+)\s*=\s*([^\n!]+?)\s*(?:!.*)?$',
    re.MULTILINE | re.IGNORECASE,
)
USE_ONLY_RE = re.compile(
    r'^\s*use\s+(\w+)\s*,\s*only\s*:\s*([^\n!]+)',
    re.MULTILINE | re.IGNORECASE,
)


def parse_subroutines(src):
    subs = []
    for m in SUB_RE.finditer(src):
        name = m.group(1)
        # Strip Fortran line-continuation `&` and newlines.
        raw = re.sub(r'&\s*\n\s*&?', ' ', m.group(2))
        arg_names = [a.strip() for a in raw.split(',') if a.strip()]
        end = END_SUB_RE.search(src, m.end())
        body = src[m.end(): end.start() if end else len(src)]
        # Only look at declarations BEFORE the first executable statement.
        # Heuristic: stop at the first non-blank, non-decl line.
        decl_section = body.split('\n! ----')[0]  # crude but works for this file
        # Join Fortran line continuations (`&` at EOL, optional `&` at
        # next-line start) so multi-line decls and `use ..., only:` lists
        # are parsed as one logical line.
        decl_section = re.sub(r'&\s*\n\s*&?', ' ', decl_section)
        args_by_name = {n.lower(): Arg(name=n, type='?', intent='?') for n in arg_names}
        for dm in DECL_RE.finditer(decl_section):
            base = dm.group('base')
            attrs = dm.group('attrs') or ''
            names_raw = dm.group('names')
            ttype = parse_type(base)
            intent_m = INTENT_RE.search(attrs)
            dim_m = DIM_RE.search(attrs)
            intent_v = intent_m.group(1).strip().lower() if intent_m else '?'
            dims = []
            if dim_m:
                dims = [d.strip() for d in dim_m.group(1).split(',')]
            # Each decl can also have per-name dim suffix like `arr(N,M)`
            # or `arr(max(n,1))`.  Split the name list on commas that are
            # NOT inside parens, then peel each per-name (...) suffix.
            for nm in _split_namelist(names_raw):
                bare_name = nm
                per_name_dim = None
                paren = re.match(r'(\w+)\s*\((.+)\)\s*$', nm)
                if paren:
                    bare_name = paren.group(1)
                    per_name_dim = _split_namelist(paren.group(2))
                key = bare_name.lower()
                if key in args_by_name:
                    a = args_by_name[key]
                    a.type = ttype
                    a.base = re.sub(r'\s+', '', base)
                    a.comment = (dm.group('comment') or '').strip()
                    if intent_v != '?':
                        a.intent = intent_v
                    if per_name_dim:
                        a.dims = per_name_dim
                    elif dims:
                        a.dims = dims
        # Capture local INTEGER PARAMETER declarations so dim symbols
        # like `mZernCoef = mZernModes` propagate into the generated
        # helper. Also capture the source module of the rhs symbol when
        # the routine `use`s it.
        local_params = {}  # name -> (rhs_expr, source_module_or_None)
        use_only_map = {}  # symbol_lower -> module
        for um in USE_ONLY_RE.finditer(decl_section):
            mod = um.group(1)
            for sym in [s.strip() for s in um.group(2).split(',')]:
                if sym:
                    use_only_map[sym.lower()] = mod
        for pm in LOCAL_PARAM_RE.finditer(decl_section):
            pname = pm.group(1)
            rhs = pm.group(2).strip()
            # If rhs is a bare identifier that was use-imported, record
            # its module for `use <mod>, only: <sym>` in the helper.
            src_mod = None
            rhs_id = re.match(r'^(\w+)$', rhs)
            if rhs_id and rhs_id.group(1).lower() in use_only_map:
                src_mod = use_only_map[rhs_id.group(1).lower()]
            local_params[pname] = (rhs, src_mod)
        subs.append((name, [args_by_name[n.lower()] for n in arg_names],
                     body, local_params, use_only_map))
    return subs


# --- Generator ---------------------------------------------------------

def array_size_expr(dim_exprs):
    """Return a Fortran integer*8 expression for total element count."""
    parts = [f'int({d}, kind=8)' for d in dim_exprs]
    return ' * '.join(parts)


def emit_helper(name, args, local_params=None, use_only_map=None):
    """Return (helper_text, prhs_descr, plhs_descr) or (None, reason, None)
    if codegen can't handle this signature."""
    local_params = local_params or {}
    use_only_map = use_only_map or {}

    # Skip routines with character args -- only load_rx/save_rx use them
    # and those are hand-written.
    if any(a.type == 'character' for a in args):
        return None, 'character arg (hand-written)', None

    # Build prhs / plhs ordering.
    prhs_args = [a for a in args if a.intent in ('in', 'inout') and not a.is_ok]
    plhs_args = [a for a in args if a.intent in ('out', 'inout') and not a.is_ok]
    has_ok = any(a.is_ok for a in args)

    # If we couldn't classify any arg, abandon.
    if any(a.intent == '?' for a in args):
        return None, f'unclassified intent on arg(s) {[a.name for a in args if a.intent=="?"]}', None
    if any(a.type == '?' for a in args):
        return None, f'unclassified type on arg(s) {[a.name for a in args if a.type=="?"]}', None

    # Collect non-arg dim names. Two categories:
    #   - Local PARAMETERs in the original routine — replicated as
    #     `integer, parameter :: name = rhs` in the helper.
    #   - Module symbols (typically from elt_mod) — added to
    #     `use elt_mod, only: ...`.
    arg_name_set = {a.name.lower() for a in args}
    used_dim_names = set()
    for a in args:
        for d in a.dims:
            tok = d.strip()
            if re.match(r'^[A-Za-z_]\w*$', tok) and tok.lower() not in arg_name_set:
                used_dim_names.add(tok)
    helper_param_lines = []
    extern_use_by_module = {}
    for nm in used_dim_names:
        if nm in local_params:
            rhs, src_mod = local_params[nm]
            if src_mod:
                # `use <mod>, only: <rhs>` so the parameter's rhs symbol
                # is available.
                extern_use_by_module.setdefault(src_mod, set()).add(rhs)
            helper_param_lines.append(
                f'        integer, parameter :: {nm} = {rhs}')
        else:
            # Consult use_only_map first -- the subroutine may already
            # have imported this symbol from a specific module (e.g.
            # max_fov / max_wl from dopt_mod).  Fall back to elt_mod
            # only when we have no other hint.
            src_mod = use_only_map.get(nm.lower(), 'elt_mod')
            extern_use_by_module.setdefault(src_mod, set()).add(nm)

    # Resolve actual declared types for `ok` and `setter` (they vary —
    # some routines declare integer instead of logical).
    ok_arg = next((a for a in args if a.is_ok), None)
    ok_type = ok_arg.type if ok_arg else 'logical'

    lines = []
    L = lines.append
    L(f'C=======================================================================')
    L(f'C     {name}  (auto-generated)')
    def _emit_arg_comment(label, alist):
        if not alist:
            L(f'C     {label}: (none)')
            return
        head = f'C     {label}: '
        line = head + ', '.join(a.name for a in alist)
        if len(line) <= 120:
            L(line)
            return
        cont = 'C            '   # align under after-label column
        cur = head + alist[0].name
        for a in alist[1:]:
            tentative = cur + ', ' + a.name
            if len(tentative) <= 120:
                cur = tentative
            else:
                L(cur + ',')
                cur = cont + a.name
        L(cur)
    _emit_arg_comment('prhs', prhs_args)
    _emit_arg_comment('plhs', plhs_args)
    L(f'C=======================================================================')
    L(f'        subroutine do_{name}(nlhs, plhs, nrhs, prhs)')
    L(f'        use macos_api_mod, only: {name}')
    for mod, syms in sorted(extern_use_by_module.items()):
        L(f'        use {mod}, only: {", ".join(sorted(syms))}')
    L(f'        implicit none')
    L(f'        mwPointer plhs(*), prhs(*)')
    L(f'        integer nlhs, nrhs')
    L(f'')
    L(f'        mwPointer mxGetPr, mxGetM, mxGetN')
    L(f'        mwPointer mxCreateDoubleMatrix, mxCreateDoubleScalar')
    L(f'        mwPointer pr')
    L(f'        integer*8 :: nbytes_tmp')
    L(f'        real(8)   :: dbuf(1)')

    # Local declarations: for each arg, declare the Fortran-side variable
    # at the type the api routine wants.
    local_decls = []
    for a in args:
        if a.is_ok:
            ok_ftype = {'logical': 'logical', 'integer': 'integer'}[ok_type]
            local_decls.append(f'        {ok_ftype} :: {a.name}')
            continue
        ftype = {'logical': 'logical', 'integer': 'integer', 'real': 'real(8)'}[a.type]
        if a.is_scalar:
            local_decls.append(f'        {ftype} :: {a.name}')
        else:
            shape = ', '.join(a.dims)
            local_decls.append(f'        {ftype}, allocatable :: {a.name}(:{",:"*(len(a.dims)-1)})')
    for line in local_decls:
        L(line)
    for line in helper_param_lines:
        L(line)
    # Workspace for logical/integer conversions
    L(f'        integer :: i_tmp')
    L(f'        real(8), allocatable :: rbuf(:)')

    # nrhs check: expects len(prhs_args) inputs (plus prhs(1) = cmd)
    n_in = len(prhs_args)
    L(f'')
    L(f'        IF (nrhs .ne. {n_in + 1}) CALL mexErrMsgTxt(')
    L(f'     &    \'{name}: expects {n_in} input arg(s)\')')

    # Collect the set of dim names used by any array arg. Scalar prhs
    # args whose name matches a dim name must be READ from prhs before
    # any array allocation. Re-ordering only affects READ ORDER, not
    # slot positions.
    dim_names = set()
    for a in args:
        for d in a.dims:
            # Pull every bare identifier out of the dim expression so
            # patterns like `max(n_zern, 1)` add n_zern to dim_names.
            # Filter out Fortran intrinsics (max, min, etc.) since
            # they're never prhs args.
            for tok in re.findall(r'[A-Za-z_]\w*', d):
                if tok.lower() in ('max', 'min', 'size', 'kind', 'len'):
                    continue
                dim_names.add(tok.lower())

    def is_dim_scalar(a):
        return a.is_scalar and a.name.lower() in dim_names

    # Step 1: read all prhs scalars into Fortran locals.
    # Read dim-scalars FIRST so subsequent allocate() calls have the size.
    L(f'')
    read_order = (
        [(idx, a) for idx, a in enumerate(prhs_args) if is_dim_scalar(a)]
        + [(idx, a) for idx, a in enumerate(prhs_args) if not is_dim_scalar(a)]
    )
    for idx, a in read_order:
        slot = idx + 2  # prhs(1) is cmd
        if a.is_scalar:
            L(f'        pr = mxGetPr(prhs({slot}))')
            L(f'        CALL mxCopyPtrToReal8(pr, dbuf, int(1, kind=8))')
            if a.type == 'logical':
                L(f'        {a.name} = (dbuf(1) .ne. 0d0)')
            elif a.type == 'integer':
                L(f'        {a.name} = int(dbuf(1))')
            else:
                L(f'        {a.name} = dbuf(1)')
        else:
            # Array. Allocate, copy in.
            shape = ', '.join(a.dims)
            nbytes = array_size_expr(a.dims)
            L(f'        allocate({a.name}({shape}))')
            if a.type == 'integer':
                # Allocate a real buffer, copy in, reshape+cast to int.
                L(f'        nbytes_tmp = {nbytes}')
                L(f'        allocate(rbuf(nbytes_tmp))')
                L(f'        pr = mxGetPr(prhs({slot}))')
                L(f'        CALL mxCopyPtrToReal8(pr, rbuf, nbytes_tmp)')
                shape_lit = '[' + ', '.join(a.dims) + ']'
                L(f'        {a.name} = int(reshape(rbuf, {shape_lit}))')
                L(f'        deallocate(rbuf)')
            elif a.type == 'logical':
                L(f'        nbytes_tmp = {nbytes}')
                L(f'        allocate(rbuf(nbytes_tmp))')
                L(f'        pr = mxGetPr(prhs({slot}))')
                L(f'        CALL mxCopyPtrToReal8(pr, rbuf, nbytes_tmp)')
                shape_lit = '[' + ', '.join(a.dims) + ']'
                L(f'        {a.name} = reshape(rbuf, {shape_lit}) .ne. 0d0')
                L(f'        deallocate(rbuf)')
            else:
                # Real array — direct copy.
                L(f'        nbytes_tmp = {nbytes}')
                L(f'        pr = mxGetPr(prhs({slot}))')
                L(f'        CALL mxCopyPtrToReal8(pr, {a.name}, nbytes_tmp)')

    # Step 2: allocate pure-output arrays.
    for a in plhs_args:
        if a.intent == 'inout':
            continue  # already allocated above
        if a.is_scalar:
            continue
        shape = ', '.join(a.dims)
        L(f'        allocate({a.name}({shape}))')

    # Step 3: Call the api routine. Build the arg list in subroutine
    # declaration order.
    sig_args = ', '.join(a.name for a in args)
    L(f'')
    # Wrap the CALL across fixed-form continuation lines if the
    # one-liner would exceed the project's -132 column limit.
    one_liner = f'        CALL {name}({sig_args})'
    if len(one_liner) <= 120:
        L(one_liner)
    else:
        # Fixed-form continuation: `&` at column 6 of the NEXT line
        # marks it as a continuation of the previous statement.  No
        # trailing `&` on the previous line (that's free-form style
        # and fixed-form rejects it).  We split args across lines
        # with the comma at the END of the previous line (so the
        # continuation line never starts with a comma).
        head = f'        CALL {name}('
        cont = '     &      '   # cols 1-5 spaces, col 6 '&', then indent
        names = [a.name for a in args]
        call_lines = []
        cur = head + names[0]
        for nm in names[1:]:
            tentative = cur + ', ' + nm
            if len(tentative) <= 120:
                cur = tentative
            else:
                call_lines.append(cur + ',')
                cur = cont + nm
        call_lines.append(cur + ')')
        for line in call_lines:
            L(line)

    # Step 4: ok check. (FAIL = 0/.false., PASS = 1/.true. — see
    # macos_api_mod's parameters.)
    if has_ok:
        if ok_type == 'integer':
            fail_test = 'ok == 0'
        else:
            fail_test = '.not. ok'
        L(f'        IF ({fail_test}) THEN')
        # Free anything we allocated
        for a in args:
            if not a.is_ok and not a.is_scalar:
                L(f'          IF (allocated({a.name})) deallocate({a.name})')
        L(f'          CALL mexErrMsgTxt(\'mmacos: {name} failed\')')
        L(f'        END IF')

    # Step 5: copy out plhs.
    for idx, a in enumerate(plhs_args):
        slot = idx + 1
        L(f'        IF (nlhs .ge. {slot}) THEN')
        if a.is_scalar:
            if a.type == 'logical':
                L(f'          plhs({slot}) = mxCreateDoubleScalar(merge(1d0,0d0,{a.name}))')
            elif a.type == 'integer':
                L(f'          plhs({slot}) = mxCreateDoubleScalar(dble({a.name}))')
            else:
                L(f'          plhs({slot}) = mxCreateDoubleScalar({a.name})')
        else:
            # Array out.
            if len(a.dims) == 1:
                M_expr = f'int({a.dims[0]}, kind=8)'
                N_expr = 'int(1, kind=8)'
            elif len(a.dims) == 2:
                M_expr = f'int({a.dims[0]}, kind=8)'
                N_expr = f'int({a.dims[1]}, kind=8)'
            else:
                return None, f'array {a.name} has rank {len(a.dims)} >2 (codegen handles ≤2D)', None
            L(f'          plhs({slot}) = mxCreateDoubleMatrix({M_expr}, {N_expr}, 0)')
            L(f'          pr = mxGetPr(plhs({slot}))')
            L(f'          nbytes_tmp = {array_size_expr(a.dims)}')
            if a.type == 'integer':
                L(f'          allocate(rbuf(nbytes_tmp))')
                L(f'          rbuf = dble(reshape({a.name}, [nbytes_tmp]))')
                L(f'          CALL mxCopyReal8ToPtr(rbuf, pr, nbytes_tmp)')
                L(f'          deallocate(rbuf)')
            elif a.type == 'logical':
                L(f'          allocate(rbuf(nbytes_tmp))')
                L(f'          rbuf = merge(1d0, 0d0, reshape({a.name}, [nbytes_tmp]))')
                L(f'          CALL mxCopyReal8ToPtr(rbuf, pr, nbytes_tmp)')
                L(f'          deallocate(rbuf)')
            else:
                L(f'          CALL mxCopyReal8ToPtr({a.name}, pr, nbytes_tmp)')
        L(f'        END IF')

    # Step 6: free remaining allocations.
    L(f'')
    for a in args:
        if not a.is_ok and not a.is_scalar:
            L(f'        IF (allocated({a.name})) deallocate({a.name})')

    L(f'        end subroutine do_{name}')
    L(f'')
    return '\n'.join(lines), prhs_args, plhs_args


# --- pymacos targets: the f2py layer and the Python veneer stubs ---------
#
# Both are emitted from the SAME parse as the mex helpers above.  The f2py
# layer (pymacos_f2py.f90, `module api`) is one forwarding wrapper per api
# routine; the per-routine `!f2py` hints and the few interface rewrites
# live in binding_overrides.py (generator INPUT).  The veneer stubs
# (pymacos/api_gen.py) are one Python function per bound routine: the
# arguments f2py exposes, `ok` checked and raised on (ApiError), the other
# outputs returned in declaration order -- the mex's plhs rule.

F2PY_USES = ['Kinds', 'math_mod', 'param_mod', 'src_mod', 'elt_mod',
             'macos_mod', 'smacos_mod']
# functions + module state re-exported to Python through lib.api
F2PY_REEXPORT = ['currrent_macos_model_size', 'checkSurfaceID', 'checkEltID',
                 'SystemCheck', 'EltRangeChk', 'StatusChk1']
# internal helper subroutines the hand-written f2py layer also wrapped
# (lib.api.translatesurfaceid / translateeltid); kept so the Python
# surface only grows.  No veneer stub.
F2PY_HELPERS_WRAPPED = ['translateSurfaceID', 'translateEltID']


def subroutine_doc(src, name):
    """The `!` comment block immediately above `subroutine NAME(`, rules
    (`!----`) dropped, the leading `! ` stripped.  [] when there is none."""
    m = re.search(r'^\s+subroutine\s+' + re.escape(name) + r'\s*\(',
                  src, re.MULTILINE | re.IGNORECASE)
    if not m:
        return []
    lines = src[:m.start()].split('\n')
    block = []
    for ln in reversed(lines[:-1] if lines and not lines[-1].strip() else lines):
        st = ln.strip()
        if st.startswith('!'):
            block.append(st)
        elif not st and not block:
            continue
        else:
            break
    out = []
    for st in reversed(block):
        body = st[1:]
        if re.match(r'^\s*[-=]{4,}\s*$', body):
            continue
        out.append(body[1:] if body.startswith(' ') else body)
    while out and not out[0].strip():
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return out


def _wrap_free(head, items, cont_indent, width=120):
    """Join `items` with ', ' after `head`, breaking with a trailing `&`
    (free form) whenever a line would pass `width`."""
    lines = []
    cur = head + items[0]
    for it in items[1:]:
        if len(cur) + 2 + len(it) <= width:
            cur += ', ' + it
        else:
            lines.append(cur + ', &')
            cur = cont_indent + it
    lines.append(cur)
    return lines


def _hint_attrs(hints):
    """name(lower) -> set of intent words the !f2py hints give it."""
    out = {}
    for h in hints:
        body = re.sub(r'^!f2py\s*', '', h, flags=re.IGNORECASE)
        if '::' not in body:
            continue
        attrs, names = body.split('::', 1)
        names = names.split('!')[0]
        im = re.search(r'intent\s*\(([^)]*)\)', attrs, re.IGNORECASE)
        words = {w.strip().lower() for w in im.group(1).split(',')} if im else set()
        for nm in _split_namelist(names):
            key = nm.split('=')[0].strip().lower()
            out.setdefault(key, set()).update(words)
    return out


def f2py_view(name, args, ov):
    """The Python-level view f2py gives this wrapper: (inputs, outputs),
    each a list of (lower-name, Arg); inputs carry an `optional` flag.

    f2py's rules as pymacos uses them: intent(out) -> returned, not an
    argument; intent(hide) -> not an argument; intent(in,out) -> argument
    AND returned; intent(inout) -> argument, modified in place; a scalar
    that sizes an input array (and that no hint names) -> optional."""
    hints = ov.F2PY_HINTS.get(name, [])
    hattr = _hint_attrs(hints)
    legacy = name in ov.F2PY_LEGACY
    dims_of_inputs = set()
    for a in args:
        if a.intent in ('in', 'inout') and a.dims:
            for d in ov.F2PY_DIMS.get(name, {}).get(a.name, a.dims):
                for tok in re.findall(r'[A-Za-z_]\w*', d):
                    dims_of_inputs.add(tok.lower())
    ins, outs = [], []
    for a in args:
        k = a.name.lower()
        words = hattr.get(k, set())
        intent = a.intent
        if not legacy and intent == 'inout':
            words = words | {'in', 'out'}
        hidden = 'hide' in words
        if intent == 'out' or 'out' in words:
            outs.append((k, a))
            if intent == 'out' or hidden:
                continue
        if hidden:
            continue
        optional = (a.is_scalar and k in dims_of_inputs and not words)
        ins.append((k, a, optional))
    return ins, outs


def emit_f2py_wrapper(name, args, local_params, ov):
    if name in ov.F2PY_VERBATIM:
        return ov.F2PY_VERBATIM[name].rstrip('\n')
    dims_ov = ov.F2PY_DIMS.get(name, {})
    head = _wrap_free(f'      subroutine {name}(', [a.name for a in args],
                      ' ' * len(f'      subroutine {name}('))
    L = head[:-1] + [head[-1] + ')']
    used = set()
    for a in args:
        for d in dims_ov.get(a.name, a.dims):
            used.update(re.findall(r'[A-Za-z_]\w*', d))
    mods = {}
    for p, (rhs, src_mod) in local_params.items():
        if p in used and src_mod:
            mods.setdefault(src_mod, []).append(rhs)
    for mod, syms in sorted(mods.items()):
        L.append(f'        use {mod}, only: {", ".join(sorted(syms))}')
    L.append('        implicit none')
    for p, (rhs, _) in local_params.items():
        if p in used:
            L.append(f'        integer, parameter :: {p} = {rhs}')
    for a in args:
        dims = dims_ov.get(a.name, a.dims)
        spec = a.base or {'logical': 'logical', 'integer': 'integer',
                          'real': 'real(8)'}[a.type]
        if dims:
            spec += f', dimension({",".join(dims)})'
        line = f'        {spec}, intent({a.intent}) :: {a.name}'
        if a.comment and len(line) + 4 + len(a.comment) <= 130:
            line = f'{line:<64s}  ! {a.comment}' if len(line) < 64 else \
                f'{line}  ! {a.comment}'
            if len(line) > 130:
                line = f'        {spec}, intent({a.intent}) :: {a.name}'
        L.append(line)
    hints = list(ov.F2PY_HINTS.get(name, []))
    if name not in ov.F2PY_LEGACY:
        for a in args:
            if a.intent == 'inout':
                hints.append(f'!f2py intent(in,out) :: {a.name}')
    for h in hints:
        L.append(f'        {h}')
    L.append('')
    call = _wrap_free(f'        CALL {name}_impl(', [a.name for a in args],
                      ' ' * len(f'        CALL {name}_impl('))
    L.extend(call[:-1] + [call[-1] + ')'])
    L.append(f'      end subroutine {name}')
    return '\n'.join(L)


def check_param_asserts(ov, api_path):
    """Each (module, PARAM, literal) of F2PY_PARAM_ASSERTS must hold in the
    macos source next to the api file, or generation stops."""
    srcdir = Path(api_path).resolve().parent
    for mod, par, lit in ov.F2PY_PARAM_ASSERTS:
        hits = []
        for f in list(srcdir.glob('*.F')) + list(srcdir.glob('*.F90')) + \
                list(srcdir.glob('*.f90')):
            t = f.read_text(errors='replace')
            if not re.search(r'^\s*module\s+' + mod + r'\b', t,
                             re.MULTILINE | re.IGNORECASE):
                continue
            hits += re.findall(r'\b' + par + r'\s*=\s*(\w+)', t, re.IGNORECASE)
        if lit not in hits:
            sys.exit(f'binding_overrides: {mod}::{par} expected {lit}, '
                     f'source has {hits or "nothing"} -- update F2PY_DIMS')


def engine_constants(ov, api_path):
    """(PYNAME, value) for each (module, PARAM) of ENGINE_CONSTANTS, read from
    the macos source next to the api file: the limits a veneer checks against
    (never a literal in macos.py).  A parameter found zero or several times
    stops generation."""
    srcdir = Path(api_path).resolve().parent
    out = []
    for mod, par in ov.ENGINE_CONSTANTS:
        hits = []
        for f in list(srcdir.glob('*.F')) + list(srcdir.glob('*.F90')) + \
                list(srcdir.glob('*.f90')):
            t = f.read_text(errors='replace')
            if not re.search(r'^\s*module\s+' + mod + r'\b', t,
                             re.MULTILINE | re.IGNORECASE):
                continue
            hits += re.findall(r'\b' + par + r'\s*=\s*(\d+)\b', t, re.IGNORECASE)
        if len(set(hits)) != 1:
            sys.exit(f'binding_overrides: ENGINE_CONSTANTS {mod}::{par} '
                     f'reads {hits or "nothing"} -- expected one integer')
        out.append((par.upper(), int(hits[0]), mod))
    return out


def emit_f2py(subs, ov):
    names = [s[0] for s in subs if s[0] not in PRIVATE_HELPERS]
    bound = names + F2PY_HELPERS_WRAPPED
    out = []
    A = out.append
    A('! pymacos_f2py.f90 -- f2py-facing wrapper layer (`module api`).')
    A('!')
    A('! AUTO-GENERATED by MACOS_resources/mmacos/src/gen_mex_wrappers.py from')
    A('! macos_api_mod.F90 (+ binding_overrides.py).  DO NOT EDIT BY HAND:')
    A('! re-run the generator.  Hand-written Python behaviour belongs in')
    A('! pymacos/macos.py, over the generated pymacos/api_gen.py.')
    A('!')
    A('! Every routine forwards to its language-neutral sibling in')
    A('! macos_api_mod (NAME_impl => NAME); the !f2py annotations live here so')
    A('! the pure-Fortran backbone stays free of Python-specific markers.')
    A(f'! Routines: {len(names)} api + {len(ov.F2PY_EXTRA)} f2py-only.')
    A('')
    A('')
    A('  module api')
    A('')
    for u in F2PY_USES:
        A(f'    use {u}')
    A('')
    A('    use macos_api_mod, only: &')
    for i, n in enumerate(bound):
        A(f'      {n}_impl => {n}' + (', &' if i < len(bound) - 1 else ''))
    A('')
    A('    ! Re-export functions and module state that Python may reach via lib.api.*')
    A('    use macos_api_mod, only: &')
    for i, n in enumerate(F2PY_REEXPORT):
        A(f'      {n}' + (', &' if i < len(F2PY_REEXPORT) - 1 else ''))
    A('')
    A('    implicit none')
    A('')
    A('  contains')
    A('')
    for name, args, body, local_params, use_only_map in subs:
        if name in PRIVATE_HELPERS and name not in F2PY_HELPERS_WRAPPED:
            continue
        A(emit_f2py_wrapper(name, args, local_params, ov))
        A('')
    for x in ov.F2PY_EXTRA:
        A(x.rstrip('\n'))
        A('')
    A('  end module api')
    A('')
    return '\n'.join(out), names


def _pyname(k):
    import keyword
    return k + '_' if keyword.iskeyword(k) else k


def _py_doc(doc_lines, name, ins, outs, ok_name):
    sig_in = ', '.join([k for k, a, o in ins if not o] +
                       [f'[{k}]' for k, a, o in ins if o])
    rets = [k for k, a in outs if k != ok_name]
    D = []
    D.extend(doc_lines or [f'macos_api_mod::{name}'])
    D.append('')
    D.append(f'Generated stub over lib.api.{name}({sig_in}).')
    if ok_name:
        D.append(f'Raises ApiError when the api returns {ok_name} = FAIL.')
    if rets:
        D.append('Returns: ' + ', '.join(rets))
    return D


def emit_veneer(subs, src, ov, consts=()):
    out = []
    A = out.append
    A('"""api_gen.py -- one Python function per macos_api_mod routine.')
    A('')
    A('AUTO-GENERATED by MACOS_resources/mmacos/src/gen_mex_wrappers.py from')
    A('macos_api_mod.F90 -- DO NOT EDIT BY HAND; re-run the generator.')
    A('')
    A('The marshalling rule is the mex\'s: inputs are the routine\'s intent(in)')
    A('and intent(inout) arguments in declaration order (minus the sizes f2py')
    A('derives from the arrays, which are optional); outputs are its intent(out)')
    A('and intent(inout) arguments in declaration order with `ok` dropped --')
    A('a FAIL raises ApiError.  One output is returned bare, several as a tuple.')
    A('(Routines bound before the generator keep their in-place intent(inout)')
    A('arrays: those are passed in and filled, not returned.)')
    A('')
    A('Behaviour (unit conversion, defaults, shapes) belongs in macos.py, which')
    A('calls these.')
    A('"""')
    A('from . import pymacosf90 as lib')
    A('')
    A('_api = lib.api')
    A('')
    A('')
    A('class ApiError(RuntimeError):')
    A('    """A macos_api_mod routine returned ok = FAIL."""')
    A('')
    if consts:
        A('')
        A('# engine limits, read from the macos source at generation time')
        A('# (binding_overrides.ENGINE_CONSTANTS); veneers check against these')
        for k, v, mod in consts:
            A(f'{k} = {v}   # {mod}')
        A('')
    names = []
    sigs = []
    for name, args, body, local_params, use_only_map in subs:
        if name in PRIVATE_HELPERS:
            continue
        if name in ov.F2PY_VERBATIM:
            # the Python view is the verbatim wrapper's, not the api's
            args = parse_subroutines(ov.F2PY_VERBATIM[name])[0][1]
        ins, outs = f2py_view(name, args, ov)
        sigs.append((name.lower(), [k for k, a, o in ins if not o],
                     [k for k, a, o in ins if o], [k for k, a in outs]))
        ok = next((k for k, a in outs if k == 'ok'), None)
        # f2py puts the optional (array-derived size) arguments last; a
        # Python keyword (perturb_elt's `del`) gets a trailing underscore
        params = ', '.join([_pyname(k) for k, a, o in ins if not o] +
                           [f'{_pyname(k)}=None' for k, a, o in ins if o])
        A('')
        A(f'def {name.lower()}({params}):')
        doc = _py_doc(subroutine_doc(src, name), name, ins, outs, ok)
        A('    """' + doc[0].replace('\\', '\\\\').replace('"""', "'''"))
        for d in doc[1:]:
            d = d.replace('\\', '\\\\').replace('"""', "'''")
            A(('    ' + d).rstrip())
        A('    """')
        req = [_pyname(k) for k, a, opt in ins if not opt]
        opt = [k for k, a, o in ins if o]
        A(f'    _kw = {{}}')
        for k in opt:
            A(f'    if {_pyname(k)} is not None:')
            A(f'        _kw[{k!r}] = {_pyname(k)}')
        A(f'    _r = _api.{name.lower()}({", ".join(req + ["**_kw"])})')
        nout = len(outs)
        rets = [k for k, a in outs if k != ok]
        if nout == 0:
            A('    return None')
        elif nout == 1:
            if ok:
                A('    if not _r:')
                A(f'        raise ApiError({name.lower()!r})')
                A('    return None')
            else:
                A('    return _r')
        else:
            if ok:
                idx = [k for k, a in outs].index(ok)
                A(f'    if not _r[{idx}]:')
                A(f'        raise ApiError({name.lower()!r})')
                keep = [i for i, (k, a) in enumerate(outs) if k != ok]
                if len(keep) == 1:
                    A(f'    return _r[{keep[0]}]')
                else:
                    A(f'    return ({", ".join(f"_r[{i}]" for i in keep)})')
            else:
                A('    return _r')
        names.append(name)
    A('')
    A('')
    A('# every macos_api_mod routine this module binds (the coverage gate reads it)')
    A('API_ROUTINES = (')
    for n in names:
        A(f'    {n.lower()!r},')
    A(')')
    A('')
    A('# the f2py view each stub was generated against: name -> (required inputs,')
    A('# optional inputs, outputs); test_api_coverage checks it against lib.api.')
    A('F2PY_SIGNATURES = {')
    for n, req, opt, outs in sigs:
        A(f'    {n!r}: ({tuple(req)!r}, {tuple(opt)!r}, {tuple(outs)!r}),')
    A('}')
    A('')
    return '\n'.join(out), names



# --- defaults target: cmdref/25_defaults.md (BRIEF_to_bindings A.6) ----------

def _read_default(root, spec, what):
    """(file, regex) -> the default as written in that source; stops the
    generator when the file or the pattern is gone (fail closed)."""
    rel, rx = spec
    p = Path(root) / rel
    if not p.is_file():
        sys.exit(f'defaults: {what}: {p} not found')
    m = re.search(rx, p.read_text(errors='replace'), re.MULTILINE)
    if not m:
        sys.exit(f'defaults: {what}: /{rx}/ no longer matches {p} -- '
                 f'update binding_overrides.OPTIONS')
    return m.group(1).strip(), rel


def emit_defaults(ov, api_names, engine_root, mmacos_root, pymacos_root):
    api = {n.lower() for n in api_names}
    L = []
    A = L.append
    A('## Session and user options: one default each')
    A('')
    A('<!-- GENERATED by MACOS_resources/mmacos/src/gen_mex_wrappers.py --targets defaults')
    A('     from binding_overrides.OPTIONS; every quoted default is READ from the named')
    A('     source at generation time.  Do not edit: re-run the generator. -->')
    A('')
    A('Every session or user option the bindings expose has ONE default, held by the '
      'engine (set at allocation when it is engine state); the CLI, mmacos and pymacos '
      'inherit it.  **A veneer never carries a default an api getter can supply** '
      '(Dave 2026-10-10).  *Status*: **inherits** -- the veneer reads the engine; '
      '**agrees** -- the veneer restates the engine\'s value and the api has no getter '
      'to inherit from; **fixed** -- a restatement removed 2026-10-16 (A.6); '
      '**Dave** -- a judgment call, routed with the measurement.  Gates: mmacos '
      '`tDefaults`, pymacos `tests/test_defaults.py` (same decks, same numbers).')
    A('')
    A('| Option | Engine default | api get / set | CLI | mmacos | pymacos | Override | Status |')
    A('|---|---|---|---|---|---|---|---|')

    def cell(spec, note, root, what):
        if spec is None:
            return note or '--'
        if isinstance(spec, str):
            return spec
        val, rel = _read_default(root, spec, what)
        c = f'`{val or "\'\'"}` (`{rel}`)'
        return f'{c}: {note}' if note else c

    for o in ov.OPTIONS:
        for r in o['get'] + o['set']:
            if r.lower() not in api:
                sys.exit(f"defaults: {o['option']}: {r} is not a macos_api_mod routine")
        gs = ', '.join(f'`{r}`' for r in o['get']) or '(no getter)'
        ss = ', '.join(f'`{r}`' for r in o['set']) or '--'
        row = [o['option'],
               cell(o['engine'], o.get('engine_note'), engine_root, o['option'] + ' engine'),
               f'{gs} / {ss}',
               o['cli'],
               cell(o['mmacos'], o.get('mmacos_note'), mmacos_root, o['option'] + ' mmacos'),
               cell(o['pymacos'], o.get('pymacos_note'), pymacos_root, o['option'] + ' pymacos'),
               o['override'],
               f"**{o['status']}**"]
        A('| ' + ' | '.join(x.replace('|', '\\|') for x in row) + ' |')
    A('')
    return '\n'.join(L)



# --- the cmdref BINDING column (BRIEF_to_bindings B.1) -------------------------

SMACOS_CMD_RE = re.compile(r"^\s*command\s*=\s*'([A-Za-z!]+)'", re.MULTILINE)


def smacos_commands(src):
    """{COMMAND: [api routines that issue it through SMACOS]} from the api source."""
    out = {}
    for name, args, body, lp, um in parse_subroutines(src):
        if name in PRIVATE_HELPERS:
            continue
        for c in SMACOS_CMD_RE.findall(body):
            out.setdefault(c.upper(), [])
            if name not in out[c.upper()]:
                out[c.upper()].append(name)
    return out


def _cli_forms(names):
    """(minimum-match prefix, full name) for each comma/slash-separated HELP name."""
    forms = []
    for nm in re.split(r'[,/]', names):
        w = nm.strip().split()[0] if nm.strip() else ''
        w = re.sub(r'[<>\[\]]', '', w)
        if not w:
            continue
        mm = re.match(r'^[A-Z!_]+', w)
        forms.append(((mm.group(0) if mm else w).upper(), w.upper()))
    return forms


def _issues(cmd, mm, full):
    """Does an api `command = 'CMD'` reach the CLI entry (mm, full)?  The
    dispatcher compares the first len(mm) characters; the HELP spelling and the
    api's can differ in the tail (POLarized / 'POLARIZATION'), so: one is a prefix
    of the other, or they share at least max(len(mm), 5) leading characters."""
    if not cmd.startswith(mm):
        return False
    if full.startswith(cmd) or cmd.startswith(full):
        return True
    k = len(os.path.commonprefix([cmd, full]))
    return k >= max(len(mm), 5)


def binding_classes(help_entries, api_src=None):
    """{entry names (as HELP writes them): (class, detail)} for every cmdref engine
    entry.  `help_entries` = [names, ...].  SMACOS commands are detected from the
    api source; every other entry must be in binding_overrides.CLI_BINDINGS, or
    this raises (fail closed)."""
    here = Path(__file__).resolve().parent
    if api_src is None:
        api_src = (here / '..' / '..' / '..' / 'macos' / 'macos_f90' /
                   'macos_api_mod.F90').resolve().read_text()
    sys.path.insert(0, str(here))
    import binding_overrides as ov
    cmds = smacos_commands(api_src)
    out, unclassified, used = {}, [], set()
    for names in help_entries:
        first = names.split(',')[0].strip().split()[0]
        hits = []
        for mm, full in _cli_forms(names):
            for c, routines in cmds.items():
                if _issues(c, mm, full):
                    hits += [r for r in routines if r not in hits]
        if hits:
            out[names] = ('SMACOS command', ', '.join(f'`{r}`' for r in hits))
        elif first in ov.CLI_BINDINGS:
            out[names] = ov.CLI_BINDINGS[first]
            used.add(first)
        else:
            unclassified.append(names)
    if unclassified:
        raise SystemExit('binding_classes: no class for ' + ', '.join(unclassified)
                         + ' -- add them to binding_overrides.CLI_BINDINGS')
    stale = sorted(set(ov.CLI_BINDINGS) - used)
    if stale:
        raise SystemExit('binding_classes: CLI_BINDINGS entries match no HELP entry '
                         'or shadow an SMACOS command: ' + ', '.join(stale))
    return out


def main():
    here = Path(__file__).parent          # mmacos/src
    # ../../../macos/macos_f90 : src -> mmacos -> MACOS_resources -> ~/dev
    default_src = here / '..' / '..' / '..' / 'macos' / 'macos_f90' / 'macos_api_mod.F90'
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('api', nargs='?', default=str(default_src.resolve()),
                    help='macos_api_mod.F90 (default: the sibling macos tree)')
    ap.add_argument('--targets', default='mex,f2py,veneer',
                    help='comma list of mex, f2py, veneer, defaults (default: the '
                         'three binding targets; defaults = cmdref/25_defaults.md)')
    ap.add_argument('--out-dir', default=None,
                    help='write every output into this directory instead of '
                         'its place in the mmacos / pymacos trees')
    opts = ap.parse_args()
    targets = {t.strip() for t in opts.targets.split(',') if t.strip()}
    bad = targets - {'mex', 'f2py', 'veneer', 'defaults'}
    if bad:
        sys.exit(f'unknown target(s): {sorted(bad)}')
    src_path = Path(opts.api)
    src = src_path.read_text()
    subs = parse_subroutines(src)
    pyroot = here / '..' / '..' / 'pymacos' / 'src'
    if opts.out_dir:
        od = Path(opts.out_dir)
        od.mkdir(parents=True, exist_ok=True)
        mex_dir = f2py_dir = ven_dir = od
    else:
        mex_dir = here
        f2py_dir = pyroot / 'cmake' / 'source'
        ven_dir = pyroot / 'pymacos'

    if {'f2py', 'veneer'} & targets:
        sys.path.insert(0, str(here))
        import binding_overrides as ov
        check_param_asserts(ov, src_path)
    if 'f2py' in targets:
        text, names = emit_f2py(subs, ov)
        p = f2py_dir / 'pymacos_f2py.f90'
        p.write_text(text)
        print(f'Wrote {p}: {len(names)} api routines + '
              f'{len(ov.F2PY_EXTRA)} f2py-only')
    if 'veneer' in targets:
        text, names = emit_veneer(subs, src, ov,
                                  engine_constants(ov, src_path))
        p = ven_dir / 'api_gen.py'
        p.write_text(text)
        print(f'Wrote {p}: {len(names)} stubs')
    if 'defaults' in targets:
        if 'ov' not in dir():
            sys.path.insert(0, str(here))
            import binding_overrides as ov  # noqa: F811
        api_names = [s_[0] for s_ in subs if s_[0] not in PRIVATE_HELPERS]
        text = emit_defaults(ov, api_names, src_path.resolve().parent,
                             here / '+macos', pyroot / 'pymacos')
        cmdref = (src_path.resolve().parent.parent / 'docs' / 'macos-manual'
                  / 'cmdref')
        p = (Path(opts.out_dir) if opts.out_dir else cmdref) / '25_defaults.md'
        p.write_text(text)
        print(f'Wrote {p}: {len(ov.OPTIONS)} options')
    if 'mex' not in targets:
        return

    helpers = []
    cmds = []
    skipped = []
    for name, args, body, local_params, use_only_map in subs:
        if name in PRIVATE_HELPERS or name in HAND_WRITTEN:
            continue
        text, prhs_args, plhs_args = emit_helper(name, args, local_params,
                                                  use_only_map)
        if text is None:
            skipped.append((name, prhs_args))  # prhs_args is reason str when text is None
            continue
        helpers.append((name, text, prhs_args, plhs_args))
        cmds.append(name)

    # Header for the generated file.
    out_lines = []
    out_lines.append('#include "fintrf.h"')
    out_lines.append('C=======================================================================')
    out_lines.append('C  mmacos_gen.F -- auto-generated by gen_mex_wrappers.py')
    out_lines.append('C  DO NOT EDIT BY HAND.  Re-run the codegen script if macos_api_mod')
    out_lines.append('C  signatures change.')
    out_lines.append(f'C  Routines covered: {len(helpers)}')
    out_lines.append(f'C  Hand-written: {len(HAND_WRITTEN_CMDS)} (dispatched in mmacos_mex.F)')
    out_lines.append(f'C  Skipped (signature too exotic for codegen): {len(skipped)}')
    for nm, reason in skipped:
        out_lines.append(f'C    - {nm}: {reason}')
    out_lines.append('C=======================================================================')
    out_lines.append('')

    # Emit the dispatch routine first.
    out_lines.append('        subroutine gen_dispatch(cmd, nlhs, plhs, nrhs, prhs, handled)')
    out_lines.append('        implicit none')
    out_lines.append('        character(len=*) :: cmd')
    out_lines.append('        mwPointer plhs(*), prhs(*)')
    out_lines.append('        integer nlhs, nrhs')
    out_lines.append('        logical handled')
    out_lines.append('        handled = .true.')
    out_lines.append('        SELECT CASE (trim(cmd))')
    for name, _, _, _ in helpers:
        out_lines.append(f'          CASE (\'{name}\')')
        out_lines.append(f'            CALL do_{name}(nlhs, plhs, nrhs, prhs)')
    out_lines.append('          CASE DEFAULT')
    out_lines.append('            handled = .false.')
    out_lines.append('        END SELECT')
    out_lines.append('        end subroutine gen_dispatch')
    out_lines.append('')

    # Emit each helper.
    for name, text, _, _ in helpers:
        out_lines.append(text)

    out_path = mex_dir / 'mmacos_gen.F'
    out_path.write_text('\n'.join(out_lines))
    print(f'Wrote {out_path}: {len(helpers)} routines, {len(skipped)} skipped')

    cmds_path = mex_dir / 'mmacos_gen_cmds.txt'
    all_cmds = sorted(set(cmds) | HAND_WRITTEN_CMDS)
    cmds_path.write_text('\n'.join(all_cmds) + '\n')
    print(f'Wrote {cmds_path}: {len(all_cmds)} total commands')

    if skipped:
        print('\nSkipped (need hand-written helper if exercised):')
        for nm, reason in skipped:
            print(f'  {nm}: {reason}')


if __name__ == '__main__':
    main()
