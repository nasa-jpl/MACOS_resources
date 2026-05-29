# mmacos — MATLAB mex bridge to MACOS / SMACOS

The MATLAB sibling of [pymacos](../pymacos).  Single `mmacos.mexa64`
with command-string dispatch, sharing the same `MODULE macos_api_mod`
(in `libsmacos.a`) that pymacos's f2py wrappers bind into.  Same
SMACOS-call layer, two languages.

## Usage

```matlab
addpath('/home/dcr/dev/MACOS_resources/mmacos')

mmacos('init', 128)                        % allocate macos at model size
n = mmacos('load_rx', '/path/Rx_Cass')     % loads Rx_Cass.in (no .in suffix!)
nE = mmacos('n_elt')

mmacos('modified_rx')                      % MODIFY -- reset trace state
[nRays, rmsWFE] = mmacos('trace_rays', n)  % full ray trace through Elt 1..n

opd = mmacos('opd')                        % OPDMat from last trace (N x N)
int = mmacos('intensity', n)               % |WFElt|^2 at element n
cf  = mmacos('complex_field', n)           % complex WFElt at element n
dx  = mmacos('dx_at', n)                   % per-element grid pitch (metres)
cbm = mmacos('base_unit_to_metres')        % BaseUnits -> metres factor

mmacos('apodize', n, mask)                 % in-place: WFElt(:,:,n) .*= mask

mmacos('perturb_elt', [2], [Rx Ry Rz Tx Ty Tz]', [1])  % global=1
mmacos('save_rx', '/path/Rx_perturbed')
```

The mex prints SMACOS diagnostics (load messages, "MODIFY reset
performed", etc.) to stdout the same way the interactive `macos` CLI
does.  Errors come back as MATLAB exceptions via `mexErrMsgTxt`.

## Build

Requires:
- MATLAB R2024a or newer (auto-detected under `/usr/local/MATLAB`)
- A built macos tree at `~/dev/macos/build_release_gfortran/` (default)
  — `cd ~/dev/macos && source ./makegfortran.sh release` gets you there.
- gfortran (default) OR Intel oneAPI ifx; both produce numerically
  identical mex artefacts.

```bash
cd /home/dcr/dev/MACOS_resources/mmacos
make                                                     # gfortran default
make FC=ifx                                              # ifx alternative
make MACOS_BUILD_DIR=~/dev/macos/build_release           # ifx tree
make MATLAB_DIR=/usr/local/MATLAB/R2025b                 # specific MATLAB
make test                                                # build + smoke
```

`mmacos.mexa64` lands next to the Makefile.  `make clean` removes
build artefacts.

### gfortran vs ifx

gfortran is the default: it produces a mex that **exits MATLAB cleanly**
(same reason GMI defaults to gfortran).  ifx works but historically
SIGSEGVs at MATLAB process-exit because of how libifcoremt's parked
threads outlive the mex DSO.  Mitigated for mmacos by `-reentrancy=none`
on the ifx link (single-threaded `libifcore.so.5` — same workaround
GMI uses).  Both compilers produce bit-identical numeric results.

## MVP command surface

Commands wired today (smoke test exercises all of them, 11/11 pass):

| Command | Inputs | Outputs | Notes |
|---|---|---|---|
| `init` | model_size (int) | — | Allocates macos at model size; one-shot per process |
| `load_rx` | filename (string, no `.in`) | nElt (int) | Wraps SMACOS `OLD` |
| `save_rx` | filename (string) | — | Wraps SMACOS `SAVE` |
| `modified_rx` | — | — | Wraps SMACOS `MODIFY` (reset trace state) |
| `n_elt` | — | nElt (int) | Current Rx element count |
| `trace_rays` | iElt (int) | [nRays, rmsWFE] | Runs the full trace; needed before opd/intensity/cfield |
| `opd` | — | N×N real matrix | OPDMat from last trace |
| `intensity` | iElt, [reset_trace] | N×N real matrix | `|WFElt|²` at iElt |
| `complex_field` | iElt, [reset_trace] | N×N complex matrix | Raw WFElt at iElt |
| `dx_at` | iElt | dx (m) | Per-element diffraction-grid pitch in SI metres |
| `base_unit_to_metres` | — | CBM | BaseUnits → metres conversion factor |
| `apodize` | iElt, mask (N×N real) | — | In-place multiply WFElt by mask |
| `perturb_elt` | iElt-vec, prb (6×N), ifGlobal-vec | — | Rigid-body PERTURB of one or more elements |

Additional commands land as the use cases surface (`sxp`, `xp_*`,
`spot_cmd/get`, `ray_info_get/set`, `stop_info_*`, `ors_run`,
`srs_run`, source-side `set_src_*/get_src_*`, the `elt_*` family).

## Files

| File | Role |
|---|---|
| `mmacos_mex.F` | The mex itself — fixed-form Fortran, single `mexFunction` that dispatches via `SELECT CASE (trim(cmd))` to per-command `do_<name>` helpers |
| `Makefile` | GMI-style build; auto-detects MATLAB; ifx + gfortran arms |
| `test_mmacos.m` | `matlab -batch`-friendly smoke test |
| `mmacos.mexa64` | Built artefact (gitignored or committed depending on policy) |

The Fortran source links against:
- `libsmacos.a` (carries `macos_api_mod`, the SMACOS engine, etc.)
- `libfitslib.a` (FITS I/O — pulled in by smacos's plot stack)
- Optionally `libnpsol.a` + `liblapacklib.a` + `libblaslib.a` when the
  macos tree was built with `-DUSE_NPSOL=ON`
- MATLAB `libmx.so` + `libmex.so` + `libmat`
