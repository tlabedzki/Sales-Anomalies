"""
Build docs/img/app-demo.svg and app-demo-dark.svg, the animated demo at the top of the README.

    python scripts/make_demo_svg.py [--shots DIR] [--reuse] [--quality 72] [--res 1.25]

The demo is an illustration made of screenshots of the real app, not a recording. One loop of 14 s: the view rests
on the timeline of anomalies (anomaly A in copper), the cursor clicks the dot of anomaly B, the page scrolls to B's
note and the figures of its change, then down to the bridge of its causes; it glides back to the timeline and the
cursor clicks A again, which is the first frame.

The script starts the Dash app on 127.0.0.1 and takes full-page screenshots 960 CSS px wide at device scale factor 2
(headless Chrome or Edge on a free DevTools port, scripts/browser.py) of the two states, set from the address
(/?anomaly=A and /?anomaly=B). The headings and the number of anomalies must be what sales_anomalies computes, or
the script stops. The cursor, the clicks and the scrollbar are vectors; the dots' positions come from Plotly in the
page. Only CSS animations of transform and opacity, no script and no external file (GitHub shows README images
through <img>); with prefers-reduced-motion the image shows the end state. The two files differ only in the browser
frame (light and dark GitHub theme). --shots keeps the screenshots and geometry; --reuse builds from them; two builds
from the same screenshots give the same bytes. The chosen anomalies and the alt texts go to docs/img/demo.json.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from browser import Browser, stop_tree, wait_for  # noqa: E402

from sales_anomalies import notes, report  # noqa: E402
from sales_anomalies import theme as look  # noqa: E402

OUT_DIR = ROOT / "docs" / "img"
DEMO_JSON = OUT_DIR / "demo.json"
CSS_W, DSF = 960, 2
VIEW = 640
W = 1000
K = W / CSS_W
BAR = 36
LIMIT = 500 * 1024
EASE = "cubic-bezier(.65,0,.35,1)"
EASE_OUT = "cubic-bezier(.22,1,.36,1)"
INK = look.INK
CURSOR = "M0 0V17.2L4.3 13.3L7.2 20.1L10.3 18.8L7.5 12.2H13.2Z"
FRAMES = {
    "light": {"bar": "#f6f8fa", "border": "#d0d7de", "dot": "#d0d7de", "pill": "#ffffff", "pill_text": "#57606a"},
    "dark": {"bar": "#161b22", "border": "#30363d", "dot": "#484f58", "pill": "#0d1117", "pill_text": "#8b949e"},
}
HIDDEN = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
PREFERRED = ("return_wave", "campaign_stop", "commission_error", "carrier_surcharge", "checkout_failure")

# Timeline (seconds)
T = 14.0
CLICK_B, CLICK_A = 2.0, 11.6
LAG = 0.25
TO_NOTE = (2.5, 3.7)
TO_BRIDGE = (6.0, 7.0)
BACK = (9.2, 10.6)


def choose(r: report.Results) -> tuple[int, int]:
    """A: the anomaly the app opens with; B: an injected one with the right cause, of another metric than A and of a
    preferred kind, so the demo shows two different stories."""
    first = report.default_anomaly(r)
    right = set(r.causes.loc[r.causes["right"] & (r.causes["source"] == "scenario"), "anomaly_id"])
    candidates = []
    for anomaly_id in right - {first}:
        kinds = r.events_of(anomaly_id)["kind"].tolist()
        rank = min((PREFERRED.index(k) for k in kinds if k in PREFERRED), default=len(PREFERRED))
        same = r.breakdowns[anomaly_id].metric == r.breakdowns[first].metric
        candidates.append((same, rank, -abs(r.breakdowns[anomaly_id].change), anomaly_id))
    if not candidates:
        sys.exit("No second anomaly to show: the run has no injected anomaly with the right cause.")
    return first, sorted(candidates)[0][-1]


def heading(r: report.Results, anomaly_id: int) -> str:
    sheet = notes.facts(r.breakdowns[anomaly_id], "en")
    return f"{sheet['segment']} · {sheet['metric']} · {sheet['days']}"


def alt_text(r: report.Results, second: int, lang: str) -> str:
    b = r.breakdowns[second]
    sheet = notes.facts(b, lang)
    cause = sheet["causes"][0] if sheet["causes"] else {"cause": sheet["main_cause"], "amount": sheet["change"]}
    count = r.summary["anomalies"]
    if lang == "pl":
        return (f"Aplikacja Sales Anomalies w ilustracji: oś czasu z {count} anomaliami z lat 2023–2024; kliknięcie "
                f"kropki {sheet['segment']} ({sheet['days']}) otwiera notatkę dla kierownika i rozbicie zmiany "
                f"{sheet['change']} na przyczyny, z główną: {cause['cause']} ({cause['amount']}); potem powrót na oś "
                f"czasu.")
    return (f"Sales Anomalies app, illustrated: a timeline of {count} anomalies in 2023–2024; a click on the dot "
            f"{sheet['segment']} ({sheet['days']}) opens the note for the manager and the change of {sheet['change']} "
            f"split into causes, the main one {cause['cause']} ({cause['amount']}); then back to the timeline.")


# ---- 1. screenshots ----------------------------------------------------------------------------------------------

GEOMETRY_JS = """(ids) => {
  const box = e => { const r = e.getBoundingClientRect();
    return {x: r.left, y: r.top + scrollY, w: r.width, h: r.height}; };
  const gd = document.querySelector('#timeline-graph .js-plotly-plot');
  const fl = gd._fullLayout, g = gd.getBoundingClientRect();
  const dots = {};
  gd.calcdata.forEach((points, t) => { const ids = gd._fullData[t].customdata || [];
    points.forEach(pt => { dots[Number(ids[pt.i])] = {x: fl.xaxis.c2p(pt.x) + fl.xaxis._offset + g.left,
                                                      y: fl.yaxis.c2p(pt.y) + fl.yaxis._offset + g.top + scrollY};
    }); });
  const anomaly = document.getElementById('anomaly-section');
  return JSON.stringify({
    timeline_section: box(document.getElementById('timeline-section')),
    timeline: box(document.getElementById('timeline')),
    anomaly: box(anomaly), note: box(anomaly.querySelector('.sa-note')),
    bridge: box(anomaly.querySelectorAll('.dg-chart')[0]),
    heading: anomaly.querySelector('h3').textContent,
    title: document.querySelector('#timeline .dg-chart-title').textContent,
    dots: {first: dots[ids[0]], second: dots[ids[1]]},
    page_bg: getComputedStyle(document.querySelector('.dg-page')).backgroundColor,
    height: document.documentElement.scrollHeight});
}"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def capture(port: int, shots: Path, first: int, second: int, headings: dict[int, str]) -> None:
    geometry = {}
    async with Browser() as page:
        await page.viewport(CSS_W, 900, DSF)
        for name, anomaly_id in (("base", first), ("final", second)):
            await page.open(f"http://127.0.0.1:{port}/?anomaly={anomaly_id}")
            wanted = json.dumps(headings[anomaly_id])
            await page.until("!!document.querySelector('#timeline-graph .js-plotly-plot .scatterlayer') && "
                             "!!document.querySelector('#anomaly h3') && "
                             f"document.querySelector('#anomaly h3').textContent === {wanted}")
            await page.js("document.fonts.ready.then(() => true)")
            await page.js("document.activeElement && document.activeElement.blur(); window.scrollTo(0, 0)")
            await page.send("Input.dispatchMouseEvent", type="mouseMoved", x=2, y=2)
            await asyncio.sleep(1.5)
            height = await page.js("document.documentElement.scrollHeight")
            shot = await page.send("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                                   clip={"x": 0, "y": 0, "width": CSS_W, "height": height, "scale": 1})
            (shots / f"{name}.png").write_bytes(base64.b64decode(shot["data"]))
            geometry[name] = json.loads(await page.js(f"({GEOMETRY_JS})({json.dumps([first, second])})"))
            print(name, height, "|", geometry[name]["heading"])
    (shots / "geometry.json").write_text(json.dumps(geometry, indent=1), encoding="utf-8")


def take_screenshots(shots: Path, first: int, second: int, headings: dict[int, str]) -> None:
    port = free_port()
    env = {**os.environ, "MPLBACKEND": "Agg", "SA_PORT": str(port)}
    app = subprocess.Popen([sys.executable, "-m", "ui.app"], cwd=ROOT, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, **HIDDEN)
    try:
        wait_for(f"http://127.0.0.1:{port}/")
        asyncio.run(capture(port, shots, first, second, headings))
    finally:
        stop_tree(app)


def check(geo: dict, r: report.Results, first: int, second: int) -> None:
    if set(geo) != {"base", "final"}:
        sys.exit("The screenshots are not the states of this script: run without --reuse.")
    for name, anomaly_id in (("base", first), ("final", second)):
        if geo[name]["heading"] != heading(r, anomaly_id):
            sys.exit(f"The app shows {geo[name]['heading']!r} for {name}: retake the screenshots.")
        if not geo[name]["title"].startswith(f"{r.summary['anomalies']} anomalies"):
            sys.exit(f"The timeline says {geo[name]['title']!r}, the run has {r.summary['anomalies']} anomalies.")
    for key in ("timeline_section", "anomaly"):
        if abs(geo["base"][key]["y"] - geo["final"][key]["y"]) > 2:
            sys.exit(f"{key} moves between the states: the overlays would not line up.")


# ---- 2. animation ------------------------------------------------------------------------------------------------

class Motion:
    """CSS keyframes on one timeline. Each track: (seconds, value, easing to the next stop)."""

    def __init__(self, total: float):
        self.total = total
        self.css: list[str] = []
        self.count = 0

    def track(self, prop: str, stops: list, rest) -> str:
        self.count += 1
        name = f"m{self.count}"
        stops = sorted(stops, key=lambda s: s[0])
        if stops[0][0] > 0:
            stops.insert(0, (0, stops[0][1], "linear"))
        if stops[-1][0] < self.total:
            stops.append((self.total, stops[-1][1], "linear"))
        frames = "".join(f"{s / self.total * 100:.3f}%{{{prop}:{value};animation-timing-function:{ease}}}"
                         for s, value, ease in stops)
        self.css.append(f"@keyframes {name}{{{frames}}}.{name}{{{prop}:{rest};"
                        f"animation:{name} {self.total:g}s linear 0s infinite both}}")
        return name


def tr(x: float, y: float) -> str:
    return f"translate({x:.1f}px,{y:.1f}px)"


def css_color(value: str) -> str:
    parts = [int(float(p)) for p in value[value.index("(") + 1:value.index(")")].split(",")[:3]]
    return "#{:02x}{:02x}{:02x}".format(*parts)


def build(shots: Path, quality: int, res: float, r: report.Results, first: int, second: int) -> dict[str, str]:
    geo = json.loads((shots / "geometry.json").read_text(encoding="utf-8"))
    check(geo, r, first, second)
    base, final = geo["base"], geo["final"]
    images = {name: Image.open(shots / f"{name}.png").convert("RGB") for name in geo}
    for name, im in images.items():
        if im.width != CSS_W * DSF:
            sys.exit(f"{name}.png is not {CSS_W} CSS px at device scale factor {DSF}: run without --reuse.")
    page_bg = css_color(base["page_bg"])
    start = base["timeline_section"]["y"] - 16
    note = final["anomaly"]["y"] - 16
    bridge = max(note, final["bridge"]["y"] + final["bridge"]["h"] + 24 - VIEW)
    height = max(base["height"], final["height"])

    def image(name: str, x0: float, y0: float, x1: float, y1: float) -> str:
        crop = images[name].crop((round(x0 * DSF), round(y0 * DSF), round(x1 * DSF), round(y1 * DSF)))
        size = (max(1, round((x1 - x0) * K * res)), max(1, round((y1 - y0) * K * res)))
        buffer = io.BytesIO()
        crop.resize(size, Image.LANCZOS).save(buffer, "WEBP", quality=quality, method=6)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return (f'<image x="{x0 * K:.2f}" y="{y0 * K:.2f}" width="{(x1 - x0) * K:.2f}" height="{(y1 - y0) * K:.2f}" '
                f'preserveAspectRatio="none" href="data:image/webp;base64,{encoded}"/>')

    m = Motion(T)
    page: list[str] = []
    view: list[str] = []
    shown = [(0, 0, "linear"), (CLICK_B + LAG, 0, EASE), (CLICK_B + LAG + 0.2, 1, "linear"),
             (CLICK_A + LAG, 1, "linear"), (CLICK_A + LAG + 0.01, 0, "linear")]
    on = m.track("opacity", shown, 1)
    bottom = max(bridge + VIEW + 8, final["anomaly"]["y"] + final["anomaly"]["h"])
    page.append(f'<rect width="{CSS_W * K:.2f}" height="{height * K:.2f}" fill="{page_bg}"/>')
    page.append(image("base", 0, max(0, start - 8), CSS_W, min(base["height"], bottom + 40)))
    t = final["timeline"]
    a = final["anomaly"]
    page.append(f'<g class="{on}">{image("final", 0, t["y"] - 4, CSS_W, t["y"] + t["h"] + 4)}'
                f'{image("final", 0, a["y"] - 6, CSS_W, min(final["height"], bottom + 40))}</g>')

    scroll = [(0, start), (TO_NOTE[0], start), (TO_NOTE[1], note), (TO_BRIDGE[0], note), (TO_BRIDGE[1], bridge),
              (BACK[0], bridge), (BACK[1], start), (T, start)]
    stops = [(s, v, EASE if i + 1 < len(scroll) and scroll[i + 1][1] != v else "linear")
             for i, (s, v) in enumerate(scroll)]
    scrolling = m.track("transform", [(s, tr(0, -v * K), e) for s, v, e in stops], tr(0, -start * K))
    thumb_h = VIEW * VIEW / height
    room = VIEW - thumb_h - 6
    scrollbar = m.track("transform", [(s, tr(0, v / (height - VIEW) * room * K), e) for s, v, e in stops],
                        tr(0, start / (height - VIEW) * room * K))
    view.append(f'<rect class="{scrollbar}" x="{(CSS_W - 7) * K:.2f}" y="{3 * K:.2f}" width="{4 * K:.2f}" '
                f'height="{thumb_h * K:.2f}" rx="{2 * K:.2f}" fill="{INK}" opacity="0.28"/>')

    rest = (820, 560)
    dot_b = (final["dots"]["second"]["x"], final["dots"]["second"]["y"] - start)
    dot_a = (base["dots"]["first"]["x"], base["dots"]["first"]["y"] - start)
    path = [(0, rest, "linear"), (0.6, rest, EASE), (CLICK_B - 0.15, dot_b, "linear"), (TO_NOTE[0] + 0.1, dot_b, EASE),
            (TO_NOTE[1] + 0.4, rest, "linear"), (BACK[1], rest, EASE), (CLICK_A - 0.15, dot_a, "linear"),
            (CLICK_A + 0.5, dot_a, EASE), (CLICK_A + 1.6, rest, "linear")]
    cursor_move = m.track("transform", [(s, tr(px * K, py * K), e) for s, (px, py), e in path],
                          tr(rest[0] * K, rest[1] * K))
    press = [(0, "scale(1)", "linear")]
    for down in (CLICK_B, CLICK_A):
        press += [(down - 0.08, "scale(1)", EASE_OUT), (down, "scale(.88)", "linear"),
                  (down + 0.12, "scale(.88)", EASE_OUT), (down + 0.3, "scale(1)", "linear")]
    cursor_press = m.track("transform", press, "scale(1)")
    for t0, (px, py) in ((CLICK_B, dot_b), (CLICK_A, dot_a)):
        ring = m.track("transform", [(t0, "scale(.2)", EASE_OUT), (t0 + 0.5, "scale(1)", "linear"),
                                     (t0 + 0.51, "scale(.2)", "linear")], "scale(1)")
        fade = m.track("opacity", [(t0 - 0.01, 0, "linear"), (t0, 0.45, EASE_OUT), (t0 + 0.5, 0, "linear")], 0)
        view.append(f'<g transform="translate({px * K:.1f} {py * K:.1f})"><g class="{fade}"><circle class="{ring}" '
                    f'r="18" fill="{look.ACCENT}" fill-opacity="0.35"/></g></g>')
    view.append(f'<g class="{cursor_move}"><g class="{cursor_press}"><path d="{CURSOR}" transform="scale(1.15)" '
                f'fill="{INK}" stroke="#ffffff" stroke-width="1.3" stroke-linejoin="round" filter="url(#shadow)"/>'
                f'</g></g>')

    alt = alt_text(r, second, "pl")
    vh = VIEW * K
    H = BAR + vh
    css = "".join(m.css) + "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
    content = (f'<g clip-path="url(#view)"><g transform="translate(0 {BAR})"><g class="{scrolling}">{"".join(page)}'
               f'</g>{"".join(view)}</g></g>')
    out = {}
    for theme, colors in FRAMES.items():
        dots = "".join(f'<circle cx="{18 + i * 16}" cy="{BAR / 2}" r="5" fill="{colors["dot"]}"/>' for i in range(3))
        host, pill_w = "127.0.0.1:8050", 150
        out[theme] = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H:.0f}" width="{W}" height="{H:.0f}" '
            f'role="img" aria-label="{alt}"><title>{alt}</title><defs><style>{css}</style>'
            f'<clipPath id="frame"><rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1:.1f}" rx="10"/></clipPath>'
            f'<clipPath id="view"><rect x="0" y="{BAR}" width="{W}" height="{vh:.1f}"/></clipPath>'
            f'<filter id="shadow" x="-50%" y="-50%" width="200%" height="200%"><feDropShadow dx="0" dy="1.2" '
            f'stdDeviation="1.4" flood-color="#000" flood-opacity="0.35"/></filter></defs>'
            f'<g clip-path="url(#frame)"><rect width="{W}" height="{H:.1f}" fill="{page_bg}"/>{content}'
            f'<rect width="{W}" height="{BAR}" fill="{colors["bar"]}"/>'
            f'<line x1="0" y1="{BAR - 0.5}" x2="{W}" y2="{BAR - 0.5}" stroke="{colors["border"]}"/>{dots}'
            f'<rect x="{W / 2 - pill_w / 2}" y="{BAR / 2 - 11}" width="{pill_w}" height="22" rx="11" '
            f'fill="{colors["pill"]}" stroke="{colors["border"]}"/>'
            f'<text x="{W / 2}" y="{BAR / 2 + 4.3}" text-anchor="middle" font-size="12.5" '
            f'fill="{colors["pill_text"]}" style="font-family:ui-monospace,SFMono-Regular,Consolas,monospace">'
            f'{host}</text></g>'
            f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1:.1f}" rx="10" fill="none" '
            f'stroke="{colors["border"]}"/></svg>')
    return out


def make(shots: Path | None = None, reuse: bool = False, quality: int = 72, res: float = 1.25) -> dict:
    r = report.load()
    first, second = choose(r)
    shots = shots or ROOT / "output" / "demo-shots"
    shots.mkdir(parents=True, exist_ok=True)
    if not (reuse and (shots / "geometry.json").exists()):
        take_screenshots(shots, first, second, {first: heading(r, first), second: heading(r, second)})
    svgs = build(shots, quality, res, r, first, second)
    for theme, svg in svgs.items():
        path = OUT_DIR / ("app-demo.svg" if theme == "light" else "app-demo-dark.svg")
        path.write_text(svg, encoding="utf-8", newline="\n")
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB, loop {T:g} s")
        if path.stat().st_size > LIMIT:
            sys.exit(f"{path.name} is over {LIMIT // 1024} KB: lower --quality or --res.")
    demo = {"first": first, "second": second, "alt_pl": alt_text(r, second, "pl"),
            "alt_en": alt_text(r, second, "en")}
    DEMO_JSON.write_text(json.dumps(demo, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shots", type=Path, help="keep the screenshots and geometry in this folder")
    parser.add_argument("--reuse", action="store_true", help="build from --shots without starting the app")
    parser.add_argument("--quality", type=int, default=72, help="WebP quality of the screenshots")
    parser.add_argument("--res", type=float, default=1.25, help="image pixels per SVG pixel")
    args = parser.parse_args()
    make(args.shots, args.reuse, args.quality, args.res)


if __name__ == "__main__":
    main()
