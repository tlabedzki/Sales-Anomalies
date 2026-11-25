"""
README images and numbers in one run: the charts (the same Plotly code as the app, drawn in headless Chrome with
Inter from the repository) and every <!-- generated:... --> block of README.md, from output/ (python main.py) and
the saved notes (data/explanations.json). The animated hero is scripts/make_demo_svg.py.

    python scripts/make_charts.py               # README in Polish (the repository is private)
    python scripts/make_charts.py --lang en     # README in English (before it goes public)

Needs Chrome or Edge and requirements-dev.txt (websockets, pillow).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from browser import Browser  # noqa: E402

from sales_anomalies import figures, i18n, notes, report  # noqa: E402
from sales_anomalies import theme as t  # noqa: E402

IMG = ROOT / "docs" / "img"
README = ROOT / "README.md"
CACHE = ROOT / ".cache" / "charts"
FONTS = ROOT / "ui" / "assets" / "fonts"
CHART_WIDTH, PAD, SCALE = 712, 24, 2  # CSS pixels; the PNG is (712 + 2 * 24) * 2 = 1520 px wide
DEMO = ROOT / "docs" / "img" / "demo.json"   # which anomalies the hero shows (written by make_demo_svg.py)


# ---------------------------------------------------------------------------------------------------------------- #
# Numbers

def featured(r: report.Results) -> int:
    """The anomaly the README walks through: the one the hero opens, else the default of the app."""
    if DEMO.exists():
        return int(json.loads(DEMO.read_text(encoding="utf-8"))["second"])
    return report.default_anomaly(r)


def kind_label(kind: str, lang: str) -> str:
    pl = {"discount": "Dodatkowy rabat w kategorii i kanale", "price_error": "Zbyt niska cena hitów",
          "return_wave": "Fala zwrotów wadliwej partii", "carrier_surcharge": "Przewoźnik podnosi stawkę",
          "competitor_undercut": "Konkurent tnie ceny", "campaign_stop": "Kampania przestaje działać",
          "checkout_failure": "Płatność nie działa na telefonach", "seo_drop": "Spada ruch z wyszukiwarki",
          "commission_error": "Wyższa prowizja afiliacyjna", "fraud_ring": "Oszustwo (przejęte konto)",
          "shop_outage": "Awaria sklepu na cały dzień", "return_spike": "Wadliwa partia jednego produktu"}
    return pl.get(kind, kind) if lang == "pl" else report.KIND_LABELS.get(kind, kind)


def recall_data(r: report.Results, lang: str) -> pd.DataFrame:
    rows = report.recall_rows(r)
    rows["label"] = [kind_label(k, lang) for k in rows["kind"]]
    rows["share"] = rows["found"] / rows["events"]
    return rows


METRIC_OF = {"margin_sold": "margin", "return_cost": "returns", "costs": "costs", "sales": "sales",
             "orders": "conversion"}


def size_data(r: report.Results, lang: str) -> pd.DataFrame:
    """
    Signal of every scenario: its true change over the noise of its segment and metric in its days (the square root
    of the summed daily variance of the model, widened by the weekly inflation for more than one day), as z.
    """
    events = r.events[r.events["source"] == "scenario"].copy()
    scores = r.scores
    signals = []
    for e in events.itertuples():
        metric = METRIC_OF[e.primary]
        segment = e.segments.split(",")[0]
        days = scores[(scores["segment"] == segment) & (scores["metric"] == metric)
                      & scores["date"].between(e.start, e.end)]
        spread = float(np.sqrt(days["variance"].sum()))
        if len(days) > 1:
            spread *= float(days["k7_up"].mean())
        change = e.primary_change  # PLN, or orders for conversion (the variance is in orders too)
        signals.append(abs(change) / spread if spread > 0 else np.nan)
    events["signal"] = signals
    events["label"] = [kind_label(k, lang) for k in events["kind"]]
    return events


def note_summary(r: report.Results) -> dict:
    """The note of every anomaly in both languages, each through the number check."""
    checked = [notes.check(r.note(i, lang), notes.facts(b, lang), lang)
               for i, b in r.breakdowns.items() for lang in ("pl", "en")]
    return {"notes": len(checked), "passed": sum(c["ok"] for c in checked),
            "invented": sum(len(c["invented"]) for c in checked)}


def numbers(lang: str) -> dict:
    r = report.load()
    n = {"r": r, "lang": lang, "featured": featured(r), "recall": recall_data(r, lang), "sizes": size_data(r, lang),
         "notes": note_summary(r)}
    n["titles"] = titles(n)
    return n


# ---------------------------------------------------------------------------------------------------------------- #
# Charts

def timeline_figure(r: report.Results, chosen: int, lang: str) -> go.Figure:
    table = report.anomaly_table(r)
    table["days"] = [i18n.period(a.start, a.end, lang) for a in r.anomalies.set_index("anomaly_id").loc[
        table["anomaly_id"]].itertuples()]
    fig = figures.timeline(table, chosen, lang, height=320)
    real = set(r.anomalies.loc[r.anomalies["verdict"] != "false", "anomaly_id"])
    for trace in fig.data:  # real anomalies in teal, false alarms in mist, the one discussed in copper
        ids = list(trace.customdata)
        if ids and ids[0] != chosen:
            trace.marker.color = [t.PRIMARY if i in real else t.MIST for i in ids]
    return fig


def recall_figure(rows: pd.DataFrame, lang: str) -> go.Figure:
    rows = rows.sort_values(["share", "events"], ascending=[False, True])
    labels = [f"{f} z {e}" if lang == "pl" else f"{f} of {e}" for f, e in zip(rows["found"], rows["events"],
                                                                              strict=True)]
    fig = go.Figure()
    fig.add_bar(y=rows["label"], x=[1.0] * len(rows), orientation="h", marker={"color": t.LINE}, hoverinfo="skip")
    fig.add_bar(y=rows["label"], x=rows["share"], orientation="h", marker={"color": t.PRIMARY}, text=labels,
                textposition="outside", cliponaxis=False, textfont=figures._num(13, t.INK), hoverinfo="skip")
    fig.update_layout(barmode="overlay")
    fig.update_yaxes(autorange="reversed", showgrid=False, ticks="", tickfont={"color": t.TEXT}, fixedrange=True,
                     ticklabelstandoff=8)
    fig.update_xaxes(visible=False, range=[0, 1.22], fixedrange=True)
    return figures._layout(fig, 25 * len(rows) + 20, bargap=0.28, margin={"l": 8, "r": 8, "t": 8, "b": 8})


def size_figure(events: pd.DataFrame, lang: str) -> go.Figure:
    order = events.groupby("label")["signal"].max().sort_values(ascending=False).index.tolist()
    fig = go.Figure()
    for found, color, name in ((True, t.PRIMARY, "wykryte" if lang == "pl" else "found"),
                               (False, t.ACCENT, "przeoczone" if lang == "pl" else "missed")):
        part = events[events["found"] == found]
        fig.add_scatter(x=part["signal"], y=part["label"], mode="markers", name=name,
                        marker={"color": color, "size": 11, "line": {"color": t.PAGE, "width": 1.5}},
                        hoverinfo="skip")
    fig.update_yaxes(categoryorder="array", categoryarray=order[::-1], showgrid=False, ticks="",
                     tickfont={"color": t.TEXT}, fixedrange=True, ticklabelstandoff=8)
    ticks = [0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    labels = [f"{v:g}".replace(".", ",") if lang == "pl" else f"{v:g}" for v in ticks]
    fig.update_xaxes(type="log", showgrid=True, gridcolor=t.LINE, zeroline=False, tickfont=figures._num(12, t.MUTED),
                     fixedrange=True, tickvals=ticks, ticktext=labels)
    fig = figures._layout(fig, 26 * len(order) + 64, margin={"l": 8, "r": 8, "t": 8, "b": 8})
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.18, "x": 0, "font": {"color": t.TEXT}})
    return fig


def font_faces() -> str:
    css = (ROOT / "ui" / "assets" / "fonts.css").read_text(encoding="utf-8")
    return css.replace("url('fonts/", f"url('{FONTS.as_uri()}/")


def keep_together(text: str) -> str:
    """No one-letter word at the end of a line (Polish typography): a non-breaking space after it."""
    return re.sub(r"(?<=\s)([aiouwzAIOUWZ]) ", lambda match: match.group(1) + " ", text)


def chart_page(title: str, subtitle: str, figure) -> str:
    title, subtitle = keep_together(title), keep_together(subtitle)
    figure.update_layout(width=CHART_WIDTH, autosize=False)
    body = figure.to_html(full_html=False, include_plotlyjs=True, config={"staticPlot": True})
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{font_faces()}
html, body {{ margin: 0; background: {t.PAGE}; }}
#chart {{ width: {CHART_WIDTH}px; padding: {PAD}px; background: {t.PAGE}; font-family: {t.SANS_STACK};
         -webkit-font-smoothing: antialiased; }}
.title {{ font-size: 21px; font-weight: 600; letter-spacing: -0.01em; color: {t.INK}; line-height: 1.3; }}
.sub {{ margin-top: 4px; font-size: 14px; color: {t.MUTED}; line-height: 1.45; }}
.graph {{ margin-top: 14px; }}
.graph text {{ font-feature-settings: {t.NUMBER_FEATURES}; }}
</style></head><body><div id="chart"><div class="title">{title}</div><div class="sub">{subtitle}</div>
<div class="graph">{body}</div></div></body></html>"""


def titles(n: dict) -> dict[str, tuple[str, str]]:
    r, lang = n["r"], n["lang"]
    s = r.summary
    b = r.breakdowns[n["featured"]]
    recall = n["recall"]
    full = recall[recall["found"] == recall["events"]]["label"].tolist()
    sizes = n["sizes"]
    missed_big = float(sizes.loc[~sizes["found"], "signal"].max()) if (~sizes["found"]).any() else 0.0
    found_small = float(sizes.loc[sizes["found"], "signal"].min()) if sizes["found"].any() else 0.0
    if lang == "pl":
        return {
            "timeline": (f"{s['anomalies']} anomalii w 2023–2024: {s['hits']} prawdziwych, {s['false_alarms']} "
                         f"fałszywych alarmów",
                         "Każda kropka to anomalia: o ile złotych zmieniła swoją metrykę wobec oczekiwań (skala "
                         "logarytmiczna, każda linia to dziesięć razy więcej); petrol prawdziwe, szare fałszywe "
                         "alarmy, miedziana omawiana niżej."),
            "recall": (f"W całości wykryte: {', '.join(x.lower() for x in full[:3])}",
                       "Ile zdarzeń każdego rodzaju znalazło wykrywanie (scenariusze wstrzyknięte i zdarzenia "
                       "oznaczone przez generator)."),
            "sizes": (f"Najsilniejszy przeoczony sygnał miał {i18n.number(missed_big, 'pl', 1)} odchylenia szumu, "
                      f"najsłabszy wykryty {i18n.number(found_small, 'pl', 1)}",
                      "Siła sygnału scenariusza: jego prawdziwa zmiana podzielona przez dzienny szum segmentu w tych "
                      "dniach (skala logarytmiczna); petrol wykryte, miedź przeoczone."),
            "bridge": report.bridge_title(b, "pl"),
        }
    return {
        "timeline": (f"{s['anomalies']} anomalies in 2023–2024: {s['hits']} real, {s['false_alarms']} false alarms",
                     "Each dot is an anomaly at the PLN it moved its metric by against what was expected (a log "
                     "scale: each line is ten times the one before); teal real, grey false, copper the one "
                     "discussed below."),
        "recall": (f"Found in full: {', '.join(x.lower() for x in full[:3])}",
                   "How many events of each kind the check found (injected scenarios and events labelled by the "
                   "generator)."),
        "sizes": (f"The strongest missed signal was {i18n.number(missed_big, 'en', 1)} noise deviations, the weakest "
                  f"found {i18n.number(found_small, 'en', 1)}",
                  "Signal of a scenario: its true change divided by the daily noise of its segment in its days (log "
                  "scale); teal found, copper missed."),
        "bridge": report.bridge_title(b, "en"),
    }


def charts(n: dict) -> list[tuple[str, str]]:
    r, lang = n["r"], n["lang"]
    title = titles(n)
    return [
        ("timeline", chart_page(*title["timeline"], timeline_figure(r, n["featured"], lang))),
        ("recall", chart_page(*title["recall"], recall_figure(n["recall"], lang))),
        ("sizes", chart_page(*title["sizes"], size_figure(n["sizes"], lang))),
        ("bridge", chart_page(*title["bridge"], figures.bridge(r.breakdowns[n["featured"]], lang))),
    ]


async def draw(pages: list[tuple[str, str]]) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    async with Browser() as page:
        await page.viewport(CHART_WIDTH + 2 * PAD, 900, SCALE)
        for name, html in pages:
            path = CACHE / f"{name}.html"
            path.write_text(html, encoding="utf-8")
            await page.open(path.as_uri())
            await page.until("document.fonts.status === 'loaded' && !!document.querySelector('.main-svg')")
            await page.js("document.fonts.ready.then(() => true)")
            await asyncio.sleep(0.5)
            await page.shot("#chart", IMG / f"{name}.png")
            print(f"Saved docs/img/{name}.png")


# ---------------------------------------------------------------------------------------------------------------- #
# README blocks

def wrap(text: str, width: int = 120) -> str:
    """Wrap prose lines and list items of a block at `width` characters; tables and images stay as they are."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"(?<=\d) (?=\d|mln|tys\.|zł|PLN)", " ", line)
        if line.lstrip().startswith(("|", "![", "<")) or len(line) <= width:
            out.append(line)
        elif line.startswith("- "):
            out.append(textwrap.fill(line, width, subsequent_indent="  ", break_long_words=False,
                                     break_on_hyphens=False))
        else:
            out.append(textwrap.fill(line, width, break_long_words=False, break_on_hyphens=False))
    return chr(10).join(out)


def write_readme(texts: dict[str, str]) -> None:
    readme = README.read_text(encoding="utf-8")
    for name, text in texts.items():
        text = wrap(text)
        pattern = re.compile(rf"(<!-- generated:{name} -->\n)(?:.*?\n)?(<!-- /generated:{name} -->)", re.S)
        if not pattern.search(readme):
            raise SystemExit(f"README.md has no block <!-- generated:{name} -->")
        readme = pattern.sub(lambda match, text=text: match.group(1) + text + "\n" + match.group(2), readme)
    README.write_text(readme, encoding="utf-8", newline="\n")


def blocks(n: dict) -> dict[str, str]:
    from readme_blocks import blocks as build  # the wording lives next to the README template

    return build(n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lang", choices=i18n.LANGS, default="pl", help="language of README.md")
    parser.add_argument("--no-charts", action="store_true", help="only the README blocks")
    args = parser.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    n = numbers(args.lang)
    if not args.no_charts:
        asyncio.run(draw(charts(n)))
    write_readme(blocks(n))
    print("Updated the generated blocks of README.md")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
