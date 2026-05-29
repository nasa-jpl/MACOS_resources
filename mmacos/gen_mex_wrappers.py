#!/usr/bin/env python3
"""
gen_mex_wrappers.py -- Codegen mmacos mex helpers from macos_api_mod.F90.

Parses every public subroutine in macos_api_mod and emits:
  - mmacos_gen.F: one do_<name>(nlhs, plhs, nrhs, prhs) helper per routine
                  + a gen_dispatch(cmd, nlhs, plhs, nrhs, prhs, handled)
                  subroutine that the main mexFunction calls after its
                  own hand-written cases.
  - mmacos_gen_cmds.txt: machine-readable command list (one per line),
                  consumed by MacosSession.m and the smoke test.

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

Run from MACOS_resources/mmacos/:
  python3 gen_mex_wrappers.py [path/to/macos_api_mod.F90]
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
    'cfield_cmd', 'cfield_get',
    'cfield_apodize', 'cfield_apodize_complex',
    'elt_dx_get', 'base_unit_to_metres',
    'trace_rays', 'prb_elt',
    # The hand-written 'perturb_elt' MATLAB cmd in mmacos_mex.F calls
    # api `prb_elt` (array, 6xN form). The api ALSO has a single-element
    # `perturb_elt(ok, iElt, th, del, useLocalCoord)` routine — skip it
    # here to avoid colliding with the hand-written do_perturb_elt. If a
    # caller needs the single-element form, expose it as a distinct mmacos
    # command (e.g. 'perturb_elt_single') with a hand-written helper.
    'perturb_elt',
}
# Command-string aliases for the hand-written ones (mexFunction
# dispatch already handles these — listed here so MacosSession.m sees
# the full surface).
HAND_WRITTEN_CMDS = {
    'init', 'load_rx', 'save_rx', 'modified_rx', 'n_elt',
    'opd', 'intensity', 'complex_field', 'apodize',
    'dx_at', 'base_unit_to_metres', 'trace_rays', 'perturb_elt',
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
    \s*(?:!.*)?$
    ''',
    re.MULTILINE | re.IGNORECASE | re.VERBOSE,
)

DIM_RE = re.compile(r'dimension\s*\(([^)]+)\)', re.IGNORECASE)
INTENT_RE = re.compile(r'intent\s*\(([^)]+)\)', re.IGNORECASE)


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
            # Each decl can also have per-name dim suffix like `arr(N)`
            for nm in [s.strip() for s in names_raw.split(',')]:
                bare_name = nm
                per_name_dim = None
                paren = re.match(r'(\w+)\s*\(([^)]+)\)', nm)
                if paren:
                    bare_name = paren.group(1)
                    per_name_dim = [d.strip() for d in paren.group(2).split(',')]
                key = bare_name.lower()
                if key in args_by_name:
                    a = args_by_name[key]
                    a.type = ttype
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
                     body, local_params))
    return subs


# --- Generator ---------------------------------------------------------

def array_size_expr(dim_exprs):
    """Return a Fortran integer*8 expression for total element count."""
    parts = [f'int({d}, kind=8)' for d in dim_exprs]
    return ' * '.join(parts)


def emit_helper(name, args, local_params=None):
    """Return (helper_text, prhs_descr, plhs_descr) or (None, reason, None)
    if codegen can't handle this signature."""
    local_params = local_params or {}

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
            extern_use_by_module.setdefault('elt_mod', set()).add(nm)

    # Resolve actual declared types for `ok` and `setter` (they vary —
    # some routines declare integer instead of logical).
    ok_arg = next((a for a in args if a.is_ok), None)
    ok_type = ok_arg.type if ok_arg else 'logical'

    lines = []
    L = lines.append
    L(f'C=======================================================================')
    L(f'C     {name}  (auto-generated)')
    L(f'C     prhs: ' + (', '.join(f'{a.name}' for a in prhs_args) or '(none)'))
    L(f'C     plhs: ' + (', '.join(f'{a.name}' for a in plhs_args) or '(none)'))
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
            tok = d.strip()
            if re.match(r'^[A-Za-z_]\w*$', tok):
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
    L(f'        CALL {name}({sig_args})')

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


def main():
    here = Path(__file__).parent
    default_src = here / '..' / '..' / 'macos' / 'macos_f90' / 'macos_api_mod.F90'
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_src.resolve()
    src = src_path.read_text()
    subs = parse_subroutines(src)

    helpers = []
    cmds = []
    skipped = []
    for name, args, body, local_params in subs:
        if name in PRIVATE_HELPERS or name in HAND_WRITTEN:
            continue
        text, prhs_args, plhs_args = emit_helper(name, args, local_params)
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

    out_path = here / 'mmacos_gen.F'
    out_path.write_text('\n'.join(out_lines))
    print(f'Wrote {out_path}: {len(helpers)} routines, {len(skipped)} skipped')

    cmds_path = here / 'mmacos_gen_cmds.txt'
    all_cmds = sorted(set(cmds) | HAND_WRITTEN_CMDS)
    cmds_path.write_text('\n'.join(all_cmds) + '\n')
    print(f'Wrote {cmds_path}: {len(all_cmds)} total commands')

    if skipped:
        print('\nSkipped (need hand-written helper if exercised):')
        for nm, reason in skipped:
            print(f'  {nm}: {reason}')


if __name__ == '__main__':
    main()
