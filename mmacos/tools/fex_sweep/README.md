# fex_sweep — FEX exit-pupil-radius compatibility sweep

Runs the engine `FEX` (via `macos.fex`) over a legacy Rx corpus and
logs **both** exit-pupil radius legs per prescription:

```
***** FEX: zp_iEm1 = <legacy iEm1->EP>   zp = <EP->next element (default)>
```

Built for the 2026-07-03 FEX EP-radius rework (Dave's spec):

- **`zp` = chief-ray distance from the EP to the NEXT element's plane
  (`iElt+1`, whatever it is — FocalPlane, coronagraph mask, ZWFS…).**
  That is the far-field propagation distance for physical optics, so it
  is the radius written to the EP Return.  No element-type scan.
- The legacy `iEm1→EP` distance is the **fallback** (no next element /
  degenerate plane) and the autoswitch alternative.
- Guards exercised by the sweep: telecentric detection (parallel probe
  chief rays), beam-footprint sanity autoswitch (reference sphere
  smaller than the beam ⇒ mass "surface miss"), and the noisy Rx-order
  flag (a Return immediately preceding the EP return usually marks an
  intermediate focus that should be a passive **Reference**; correct
  pattern: `Reference (FP), Return (EP), Return (FP)`).

**Key compatibility fact:** on a conforming double-pass Rx the element
before the EP return sits AT the focus, so the two legs are equal by
construction and the rework is a bit-exact no-op (verified: e5hex1,
6MST, SegDemo3conic_bespoke).  The legs differ only where the element
before the EP is *not* at the focus (e.g. a last optic close to the
EP) — exactly the geometries where the legacy radius was degenerate.

## Usage

```bash
./run_fex_sweep.sh [outdir]     # default ./sweep_logs (gitignored)
grep -h "zp_iEm1\|WARNING\|AUTOSWITCH\|TELECENTRIC" sweep_logs/*.log
```

Corpus dirs are listed in `CORPUS_DIRS` at the top of the script
(currently `~/dev/MACOS_sandbox/old_Rx` + the macos-manual examples).
An Rx is swept when it has ≥4 `Element=` lines and its next-to-last
element is Return/Reference; Rx without `ApStop=` get a heuristic stop
at the first reflective element (most of the legacy corpus predates
`ApStop=` — see PLAN §0: SAVE must learn to round-trip it).

One MATLAB process per Rx (`fex_one.m`, env-driven): loader crashes
on ancient Rx can't kill the sweep, and Fortran-vs-MATLAB stdout
interleave can't mispair results.
