#!/usr/bin/env python3
"""Merge the per-model-size driver parts into generated/numbers.json.

The MATLAB driver can only run ONE model size per process (macos_init_all()
corrupts the heap across model_size transitions -- see mmacos/CLAUDE.md), so
run_pol_validation.m is invoked once per size and each run writes
generated/parts/numbers_<model>.json.  This merges them.

Refuses to merge when:
  * no parts exist, or an expected part is missing;
  * two parts define the SAME token -- that would mean one measurement
    silently overwriting another, and which one won would depend on file
    ordering.

Provenance is carried from the parts: the git/host/MATLAB fields must agree
across parts (they were captured in the same session of make_polval.sh), and
`model_size` becomes the list of sizes actually run, so the report stamps
every size its numbers came from rather than just the first.

Usage:  merge_numbers.py <polvalDir> [expected_model ...]
"""
from __future__ import annotations

import json
import pathlib
import sys

# fields that must be identical across parts -- they describe the tree and
# the box, not the run, so a disagreement means the parts are from different
# sessions and must not be stitched into one report
SHARED = (
    "engine_sha", "engine_branch", "engine_dirty",
    "resources_sha", "resources_branch", "resources_dirty",
    "matlab", "host",
)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    polval = pathlib.Path(sys.argv[1])
    expected = [int(a) for a in sys.argv[2:]]

    partdir = polval / "generated" / "parts"
    parts = sorted(partdir.glob("numbers_*.json"))
    if not parts:
        sys.exit(f"merge_numbers: no parts under {partdir} -- run the MATLAB driver")

    got: dict[int, dict] = {}
    for p in parts:
        data = json.loads(p.read_text())
        got[int(data["provenance"]["model_size"])] = data

    missing = [m for m in expected if m not in got]
    if missing:
        sys.exit(
            "merge_numbers: missing part(s) for model size "
            + ", ".join(str(m) for m in missing)
        )

    values: dict[str, dict] = {}
    owner: dict[str, int] = {}
    for model in sorted(got):
        for k, v in got[model]["values"].items():
            if k in values:
                sys.exit(
                    f"merge_numbers: token {k} is measured by BOTH model "
                    f"{owner[k]} and model {model} -- rename one; a silent "
                    f"overwrite would make the report depend on file order."
                )
            values[k] = v
            owner[k] = model

    base = got[sorted(got)[0]]["provenance"]
    for model in sorted(got):
        for f in SHARED:
            if got[model]["provenance"].get(f) != base.get(f):
                sys.exit(
                    f"merge_numbers: provenance field '{f}' differs between "
                    f"model {sorted(got)[0]} and model {model} -- the parts are "
                    f"from different sessions and must not be merged."
                )
    prov = dict(base)
    prov["model_size"] = " / ".join(str(m) for m in sorted(got))
    prov["generated"] = max(got[m]["provenance"]["generated"] for m in got)

    out = polval / "generated" / "numbers.json"
    out.write_text(json.dumps({"provenance": prov, "values": values}, indent=2) + "\n")
    print(
        f"merge_numbers: {len(values)} tokens from model size(s) "
        f"{prov['model_size']} -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
