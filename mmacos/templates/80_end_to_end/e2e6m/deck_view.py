#!/usr/bin/env python3
"""deck_view -- render deck_e2e6m.pptx to a PDF and a one-page HTML viewer.

The .pptx is the deliverable, but it is awkward to review in a terminal
or to hand round for comment.  This renders it to

    deck_e2e6m.pdf        every page, 0.7 MB, opens anywhere
    deck_e2e6m_view.html  all pages in one scrollable page, self-contained
                          (images inlined as data URIs, no external assets)

Both are DERIVED -- deck_e2e6m.py is the source of truth and the .pptx is
never hand-edited, so these are never hand-edited either.  Slide titles
and kickers are read from deck_e2e6m.md so the viewer's contents list
cannot drift from the deck.

Usage:  python3 deck_view.py
Needs:  libreoffice (soffice) and pdftoppm on PATH.

See also: deck_e2e6m.py (the generator), e2e6m_records.py (the parsers).
"""
import base64
import glob
import html
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PPTX = os.path.join(HERE, "deck_e2e6m.pptx")
MD   = os.path.join(HERE, "deck_e2e6m.md")
PDF  = os.path.join(HERE, "deck_e2e6m.pdf")
HTML = os.path.join(HERE, "deck_e2e6m_view.html")
DPI  = 110

for tool in ("soffice", "pdftoppm"):
    if shutil.which(tool) is None:
        sys.exit(f"deck_view: {tool} not on PATH")
if not os.path.isfile(PPTX):
    sys.exit(f"deck_view: {PPTX} missing -- run deck_e2e6m.py first")


def headings(md_path):
    """(kind, title, kicker) per slide, in deck order, from the markdown."""
    out = []
    for line in open(md_path, encoding="utf-8").read().split("\n"):
        if line.startswith("# "):
            out.append(("title", line[2:].strip(), ""))
        elif line.startswith("## "):
            t = line[3:].strip()
            if "|" in t:
                a, b = t.split("|", 1)
                out.append(("slide", a.strip(), b.strip()))
            elif t == "Backup Slides":
                out.append(("divider", t, ""))
            else:
                out.append(("slide", t, ""))
    return out


def render(tmp):
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf",
                    "--outdir", tmp, PPTX], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    made = os.path.join(tmp, "deck_e2e6m.pdf")
    shutil.copyfile(made, PDF)
    subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-aa", "yes",
                    "-aaVector", "yes", made, os.path.join(tmp, "sl")],
                   check=True)
    return sorted(glob.glob(os.path.join(tmp, "sl-*.png")))


CSS = """
:root{
  --ground:#F6F8FA; --surface:#FFFFFF; --ink:#121821; --muted:#59647A;
  --rule:#DBE2EB; --accent:#1F4E79; --accent-soft:#E8EFF6; --shadow:rgba(18,24,33,.10);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0C1116; --surface:#151C24; --ink:#E4EAF1; --muted:#94A2B8;
    --rule:#232D3A; --accent:#79ADDC; --accent-soft:#182634; --shadow:rgba(0,0,0,.5);
  }
}
:root[data-theme="dark"]{
  --ground:#0C1116; --surface:#151C24; --ink:#E4EAF1; --muted:#94A2B8;
  --rule:#232D3A; --accent:#79ADDC; --accent-soft:#182634; --shadow:rgba(0,0,0,.5);
}
*{box-sizing:border-box}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16px; line-height:1.55; -webkit-font-smoothing:antialiased}
.mono,.idx,.bar .meta,.bar .tag,nav.toc .n,.break-label,footer code{
  font-family:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace}
header.bar{position:sticky; top:0; z-index:10; background:var(--surface);
  border-bottom:1px solid var(--rule); display:flex; align-items:baseline;
  gap:.9rem 1.4rem; flex-wrap:wrap; padding:.75rem clamp(1rem,4vw,2.5rem)}
.bar h1{margin:0; font-size:1rem; font-weight:640; letter-spacing:-.01em}
.bar .meta{font-size:.75rem; color:var(--muted); letter-spacing:.02em;
  font-variant-numeric:tabular-nums}
.bar .tag{font-size:.68rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--accent); border:1px solid var(--accent); padding:.12rem .45rem; border-radius:2px}
.wrap{max-width:1180px; margin:0 auto;
  padding:clamp(1.25rem,3vw,2.25rem) clamp(1rem,4vw,2.5rem) 5rem;
  display:flex; flex-direction:column; gap:clamp(1.5rem,3vw,2.5rem)}
.lede{max-width:62ch; display:flex; flex-direction:column; gap:.6rem}
.lede p{margin:0; color:var(--muted); font-size:.95rem}
.lede strong{color:var(--ink); font-weight:600}
nav.toc{border:1px solid var(--rule); background:var(--surface); border-radius:3px;
  padding:1rem 1.1rem; display:grid; gap:1rem 2rem;
  grid-template-columns:repeat(auto-fit,minmax(min(100%,17rem),1fr))}
nav.toc h2{margin:0 0 .5rem; font-size:.68rem; letter-spacing:.1em;
  text-transform:uppercase; color:var(--muted); font-weight:640}
nav.toc ul{list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:.15rem}
nav.toc a{display:flex; gap:.6rem; text-decoration:none; color:var(--ink);
  font-size:.88rem; padding:.15rem .25rem; border-radius:2px}
nav.toc a:hover{background:var(--accent-soft); color:var(--accent)}
nav.toc .n{color:var(--muted); font-size:.78rem;
  font-variant-numeric:tabular-nums; padding-top:.08rem}
a:focus-visible{outline:2px solid var(--accent); outline-offset:2px}
.slide{margin:0; display:flex; flex-direction:column; gap:.55rem; scroll-margin-top:4.5rem}
.cap{display:flex; gap:.85rem; align-items:baseline}
.idx{font-size:.72rem; letter-spacing:.06em; color:var(--muted);
  font-variant-numeric:tabular-nums; padding-top:.15rem; flex:0 0 auto}
.cap-text{display:flex; flex-direction:column; gap:.1rem; min-width:0}
.cap-text .t{font-weight:620; font-size:1rem; letter-spacing:-.005em; text-wrap:balance}
.cap-text .k{color:var(--muted); font-size:.88rem; text-wrap:pretty}
.slide img{display:block; width:100%; height:auto; border:1px solid var(--rule);
  border-radius:2px; background:#fff; box-shadow:0 1px 3px var(--shadow)}
.break{display:flex; flex-direction:column; gap:.35rem; padding-top:1rem;
  border-top:2px solid var(--accent)}
.break-label{font-size:.72rem; letter-spacing:.11em; text-transform:uppercase; color:var(--accent)}
.break-note{margin:0; color:var(--muted); font-size:.9rem; max-width:60ch}
footer{border-top:1px solid var(--rule); color:var(--muted); font-size:.82rem;
  padding-top:1rem; max-width:70ch}
footer code{font-size:.95em; color:var(--ink)}
@media (prefers-reduced-motion:reduce){*{animation:none!important; transition:none!important}}
"""


def main():
    items = headings(MD)
    with tempfile.TemporaryDirectory() as tmp:
        pages = render(tmp)
        if len(pages) != len(items):
            sys.exit(f"deck_view: {len(pages)} rendered pages vs "
                     f"{len(items)} headings in deck_e2e6m.md")
        E = html.escape
        cards, nav_main, nav_back = [], [], []
        backup = False
        n_main = n_back = 0
        for i, ((kind, title, kick), png) in enumerate(zip(items, pages), 1):
            if kind == "divider":
                backup = True
                cards.append(
                    '<div class="break" role="separator">'
                    '<span class="break-label">Backup slides</span>'
                    '<p class="break-note">Diagnostics, negative results and '
                    'reproduction notes. Everything above is the success path.'
                    '</p></div>')
            label = ("Title" if kind == "title"
                     else "Divider" if kind == "divider" else f"{i:02d}")
            b64 = base64.b64encode(open(png, "rb").read()).decode()
            cards.append(
                f'<figure class="slide" id="s{i}">\n'
                f'  <figcaption class="cap"><span class="idx">{E(label)}</span>'
                f'<span class="cap-text"><span class="t">{E(title)}</span>'
                + (f'<span class="k">{E(kick)}</span>' if kick else "")
                + '</span></figcaption>\n'
                f'  <img src="data:image/png;base64,{b64}" '
                f'alt="Slide {i}: {E(title)}" loading="lazy">\n</figure>')
            if kind != "divider":
                row = (f'<li><a href="#s{i}"><span class="n">{E(label)}</span>'
                       f'{E(title)}</a></li>')
                if backup:
                    nav_back.append(row); n_back += 1
                else:
                    nav_main.append(row); n_main += 1

    body = f"""<header class="bar">
  <h1>e2e6m — a 6 m unobscured coronagraph, end to end</h1>
  <span class="tag">Draft</span>
  <span class="meta">{len(items)} pages · {n_main - 1} main + {n_back} backup · generator-built</span>
</header>
<div class="wrap">
  <div class="lede">
    <p>Every number on these pages is parsed from a committed stage report —
    the generator stops rather than print a value it cannot find — and every
    figure is a committed stage artifact, cropped but never redrawn.</p>
    <p><strong>Not signed off.</strong> Rebuild with
    <code class="mono">python3 deck_e2e6m.py</code> after any stage re-runs,
    then <code class="mono">python3 deck_view.py</code> for this page.</p>
  </div>
  <nav class="toc" aria-label="Slides">
    <div><h2>Main path</h2><ul>{''.join(nav_main)}</ul></div>
    <div><h2>Backup</h2><ul>{''.join(nav_back)}</ul></div>
  </nav>
  {''.join(cards)}
  <footer><p>Rendered from <code>deck_e2e6m.pptx</code> at {DPI} dpi.
  Source of truth is <code>deck_e2e6m.py</code>; neither the .pptx nor this
  page is ever hand-edited.</p></footer>
</div>"""

    open(HTML, "w", encoding="utf-8").write(
        f"<title>e2e6m draft deck</title>\n<style>{CSS}</style>\n{body}\n")
    print(f"wrote {PDF}  ({os.path.getsize(PDF)/1e6:.2f} MB)")
    print(f"wrote {HTML}  ({os.path.getsize(HTML)/1e6:.2f} MB, "
          f"{len(items)} pages inlined)")


if __name__ == "__main__":
    main()
