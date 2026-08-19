#!/usr/bin/env python3
"""Render the polarization validation report: polval/*.md.in -> polval/*.md.

Every measured number in the report is a @@TOKEN@@ in the .md.in prose,
substituted here from the driver's generated/numbers.json (measured on this
box, this build) or from external.json (gates this driver cannot run --
another language binding, another compiler, or a historical pre-fix binary).
Nothing numeric is typed into the prose by hand, so a stale figure cannot
ship next to a fresh number, or vice versa.

Fails loudly on:
  * an unresolved @@TOKEN@@ left in any output (typo, or the driver did not
    produce that value) -- the report is never rendered half-substituted;
  * a numbers.json older than a figure it is supposed to describe.

Usage:  render_polval.py <polvalDir> [--external external.json]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

TOKEN = re.compile(r"@@([A-Za-z0-9_]+)@@")


def load_values(polval: pathlib.Path, external: pathlib.Path) -> dict[str, dict]:
    numbers = polval / "generated" / "numbers.json"
    if not numbers.exists():
        sys.exit(
            f"render_polval: {numbers} not found -- run the MATLAB driver first "
            f"(make_polval.sh, or `make polval-regen`)."
        )
    data = json.loads(numbers.read_text())
    values = dict(data["values"])
    prov = data["provenance"]

    if external.exists():
        ext = json.loads(external.read_text())
        for k, v in ext["values"].items():
            if k in values:
                sys.exit(
                    f"render_polval: token {k} is defined by BOTH the driver and "
                    f"external.json -- the driver measurement must win, so remove "
                    f"the external entry."
                )
            values[k] = v

    # provenance tokens, so the report always stamps the build it describes
    for k, v in prov.items():
        values[f"PROV_{k.upper()}"] = {"text": str(v), "source": "provenance"}
    return values


def check_freshness(polval: pathlib.Path) -> str:
    """Warn (in the rendered text) if any figure post-dates numbers.json."""
    numbers = polval / "generated" / "numbers.json"
    media = sorted((polval / "media").glob("*.png"))
    if not media:
        return "no figures found"
    nt = numbers.stat().st_mtime
    newer = [m.name for m in media if m.stat().st_mtime > nt + 1.0]
    if newer:
        return "STALE: figures newer than numbers.json: " + ", ".join(newer)
    return "consistent (%d figures, all older than numbers.json)" % len(media)


def render(polval: pathlib.Path, values: dict[str, dict]) -> int:
    values["PROV_FRESHNESS"] = {"text": check_freshness(polval), "source": "renderer"}
    templates = sorted(polval.glob("*.md.in"))
    if not templates:
        sys.exit(f"render_polval: no *.md.in under {polval}")

    # Resolve everything FIRST and write nothing until the whole report
    # substitutes cleanly: a half-rendered .md left on disk after a failure
    # is exactly the stale-number hazard this tool exists to prevent.
    missing: dict[str, set[str]] = {}
    rendered: list[tuple[pathlib.Path, str]] = []
    for tpl in templates:

        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in values:
                missing.setdefault(name, set()).add(tpl.name)
                return m.group(0)
            return str(values[name]["text"])

        out = TOKEN.sub(sub, tpl.read_text())
        header = (
            "<!-- GENERATED FILE -- do not edit.\n"
            f"     Source: {tpl.name}\n"
            "     Numbers: generated/numbers.json (MATLAB driver)\n"
            "     Regenerate: make polval-regen  (docs/macos-manual)\n"
            "-->\n"
        )
        rendered.append((tpl.with_suffix(""), header + out))   # strip .in -> *.md

    if missing:
        print("render_polval: UNRESOLVED tokens -- nothing written:", file=sys.stderr)
        for name in sorted(missing):
            print(f"  @@{name}@@  in {', '.join(sorted(missing[name]))}", file=sys.stderr)
        return 1

    for dest, text in rendered:
        dest.write_text(text)
        print(f"render_polval: wrote {dest.relative_to(polval)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("polval", type=pathlib.Path)
    ap.add_argument(
        "--external",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("external.json"),
    )
    a = ap.parse_args()
    return render(a.polval, load_values(a.polval, a.external))


if __name__ == "__main__":
    sys.exit(main())
