"""
Dash + Mantine app of Sales Anomalies, in the portfolio look of Data-Generator: every anomaly of two years on a
timeline; a click opens its breakdown into causes in PLN, the note written from those numbers and the answer key.

Run locally (from the repository root, after python main.py):
    python -m ui.app            # http://127.0.0.1:8050

The address can open an anomaly, so a view is a link: /?anomaly=12. The server listens on 127.0.0.1 only.
"""
from __future__ import annotations

import os
import sys
from functools import cache
from pathlib import Path
from urllib.parse import parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dash._dash_renderer  # noqa: E402
import dash_mantine_components as dmc  # noqa: E402
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update  # noqa: E402

from sales_anomalies import explain, figures, i18n, report  # noqa: E402
from ui import theme as t  # noqa: E402

dash._dash_renderer._set_react_version("19.2.4")  # Mantine 3 needs React 19; Dash 4.5 still defaults to 18

REPO_URL = "https://github.com/tlabedzki/Sales-Anomalies"
METRIC_FILTER = [("all", "All"), ("sales", "Sales"), ("margin", "Margin"), ("returns", "Returns"),
                 ("costs", "Costs"), ("conversion", "Conversion")]
TABLE_COLUMNS = [  # column, header, css class
    ("days", "Days", ""), ("segment", "Part of the shop", ""), ("metric", "Metric", ""),
    ("change", "Change", "sa-num"), ("change_percent", "vs expected", "sa-num"), ("cause", "Main cause", ""),
    ("verdict", "Answer key", ""),
]


@cache
def results() -> report.Results:
    return report.load()


@cache
def table():
    return report.anomaly_table(results())


def money(value: float) -> str:
    return i18n.money(value, "en", sign=True)


# ---------------------------------------------------------------------------------------------------------------- #
# Building blocks

def section(number, title, lead, body, section_id):
    return html.Section([
        html.Div([html.Span(f"{number:02d}", className="dg-step"), html.H2(title, className="dg-h2")],
                 className="dg-section-head"),
        dmc.Text(lead, className="dg-lead-small"),
        html.Div(body, id=f"{section_id}-body"),
    ], id=section_id, className="dg-section")


def stat(label, value, note="", extra_class=""):
    return html.Div([html.Div(value, className="dg-stat-value"), html.Div(label, className="dg-stat-label"),
                     html.Div(note, className="dg-stat-note")], className=f"dg-stat {extra_class}".strip())


def chart_block(text, figure, graph_id=None, extra_class=""):
    title, subtitle = text
    graph = {"id": graph_id} if graph_id else {}
    return html.Figure([
        html.Figcaption([html.Div(title, className="dg-chart-title"), html.Div(subtitle, className="dg-chart-sub")]),
        dcc.Graph(figure=figure, config=figures.CONFIG, className="dg-graph",
                  style={"height": f"{figure.layout.height}px"}, **graph),
    ], className=f"dg-chart {extra_class}".strip())


def scroll_table(header, rows, label):
    body = html.Table([html.Thead(html.Tr(header)), html.Tbody(rows)], className="dg-table sa-table")
    return html.Div(body, className="dg-table-scroll", tabIndex="0", role="region", **{"aria-label": label})


# ---------------------------------------------------------------------------------------------------------------- #
# Views (pure functions of the results, tested without a browser)

def summary_stats(r: report.Results) -> html.Div:
    s = r.summary
    return html.Div([
        stat("anomalies in two years", f"{s['anomalies']:,}", f"{s['false_per_month']:.2f} false alarms a month"),
        stat("of them real", i18n.percent(s["precision"], "en", sign=False), "precision against the answer key"),
        stat("injected anomalies found", f"{s['scenarios_found']}/{s['scenarios_total']}",
             i18n.percent(s["recall_scenarios"], "en", sign=False) + " recall"),
        stat("causes add up", f"{s['breakdowns_exact_to_grosz']}/{s['breakdowns']}", "to the grosz"),
        stat("main cause right", f"{s['main_cause_right']}/{s['main_cause_graded']}", "of the real anomalies"),
    ], className="dg-stats")


def timeline_view(r: report.Results, metric: str, selected: int | None) -> html.Figure:
    data = table() if metric in (None, "all") else table()[table()["metric"] == i18n.METRICS["en"][metric]]
    count = len(data)
    real = int((data["verdict"] != report.VERDICTS["false"]).sum())
    title = (f"{count} anomalies in 2023-2024, {real} of them real" if count else "No anomaly of this metric",
             "Each dot is an anomaly at the PLN it moved its metric by, against what was expected (a log scale: "
             "each line is ten times the one before). Click one to open it.")
    return chart_block(title, figures.timeline(data, selected), graph_id="timeline-graph")


def note_block(r: report.Results, anomaly_id: int) -> html.Div:
    note = r.note(anomaly_id, "en")
    if note["source"] == "model":
        badge = f"Written by {explain.MODEL} from the numbers below; every number checked by code"
        if len(note.get("attempts", [])) > 1:
            badge += " (second try)"
    else:
        badge = "Written by code: no saved note from the model for this anomaly"
    return html.Div([html.P(note["text"], className="sa-note-text"), html.Div(badge, className="sa-note-badge")],
                    className="sa-note")


def anomaly_view(r: report.Results, anomaly_id: int) -> list:
    b = r.breakdowns[anomaly_id]
    row = r.row(anomaly_id)
    sheet = explain.facts(b, "en")
    heading = f"{sheet['segment']} · {sheet['metric']} · {sheet['days']}"
    percent = report.change_percent(b)
    metric_unit = "orders" if b.metric == "conversion" else "PLN"
    stats = html.Div([
        stat("change against expected", money(b.change), i18n.percent(percent, "en") + " of expected",
             "sa-negative" if b.change < 0 else ""),
        stat("expected", i18n.money(b.expected, "en"), "from the model, before the day was seen"),
        stat("actual", i18n.money(b.observed, "en"), "what the shop booked"),
        stat("main cause", i18n.CAUSES["en"][b.main_cause].capitalize(), sheet["suggested_check"]),
    ], className="dg-stats sa-stats-4")
    evidence = html.Ul([html.Li(item) for item in sheet["evidence"]], className="sa-evidence") \
        if sheet["evidence"] else None
    window = report.series_window(r, anomaly_id)
    blocks = [
        html.H3(heading, className="dg-chart-title"),
        note_block(r, anomaly_id),
        stats,
        chart_block(report.bridge_title(b), figures.bridge(b), extra_class="sa-block"),
        evidence,
        chart_block((f"{sheet['metric'].capitalize()} of {sheet['segment']} around it",
                     f"Daily {metric_unit}: actual (teal) and expected (dotted); the anomaly's days are shaded."),
                    figures.series(window, b.metric), extra_class="sa-block"),
        html.P(report.truth_line(r, anomaly_id),
               className=f"sa-truth {'sa-false' if row['verdict'] == 'false' else ''}".strip()),
    ]
    return [block for block in blocks if block is not None]


def table_view(r: report.Results) -> html.Div:
    rows = []
    for record in table().to_dict("records"):
        cells = []
        for column, _, css in TABLE_COLUMNS:
            value = record[column]
            if column == "days":
                content = html.Button(value, id={"type": "open-anomaly", "index": int(record["anomaly_id"])},
                                      className="sa-open", title="Open this anomaly")
            elif column == "change":
                content, css = money(value), f"{css} {'sa-loss' if value < 0 else 'sa-gain'}"
            elif column == "change_percent":
                content = i18n.percent(value, "en")
            else:
                content = value
            cells.append(html.Td(content, className=css.strip() or None))
        rows.append(html.Tr(cells, className="sa-false-row" if record["verdict"] == report.VERDICTS["false"]
                            else None))
    header = [html.Th(name, className=css or None) for _, name, css in TABLE_COLUMNS]
    return scroll_table(header, rows, "All anomalies, scrolls in both directions")


def truth_view(r: report.Results) -> html.Div:
    rows = [html.Tr([html.Td(record["label"]),
                     html.Td("injected" if record["source"] == "scenario" else "labelled by the generator"),
                     html.Td(f"{record['found']} of {record['events']}", className="sa-num")])
            for record in report.recall_rows(r).to_dict("records")]
    header = [html.Th("Event"), html.Th("Source"), html.Th("Found", className="sa-num")]
    return scroll_table(header, rows, "Recall by kind of event")


def start_anomaly(r: report.Results, requested: str | None) -> int:
    try:
        value = int(requested) if requested else None
    except ValueError:
        value = None
    return value if value in r.breakdowns else report.default_anomaly(r)


# ---------------------------------------------------------------------------------------------------------------- #
# Page

def page():
    header = html.Header(html.Div([
        html.Span(["Sales Anomalies", html.Span(".", className="dg-dot")], className="dg-wordmark"),
        html.A("Source on GitHub", href=REPO_URL, target="_blank", className="dg-link"),
    ], className="dg-wrap dg-topline"))
    hero = html.Div([
        html.P("Lumo · daily anomaly check", className="dg-eyebrow"),
        html.H1(["Every unusual day, ", html.Em("explained"), " in złoty."], className="dg-h1"),
        html.P("Code finds the days and weeks when sales, margin, returns, costs or conversion left their expected "
               "band, splits each change into causes that add up to the grosz, and a language model writes one or "
               "two sentences from those numbers only.", className="dg-lead"),
    ], className="dg-hero")
    try:
        r = results()
    except FileNotFoundError as error:
        body = [hero, section(1, "Timeline", "The results are missing.", html.P(str(error), className="dg-placeholder"),
                              "timeline")]
    else:
        selected = report.default_anomaly(r)
        body = [
            hero,
            summary_stats(r),
            section(1, "When did something move?", f"Two years of daily checks on {r.summary['segments']} parts of "
                    "the shop and five metrics.", [
                        dmc.SegmentedControl(id="metric", data=[{"value": v, "label": label}
                                                                for v, label in METRIC_FILTER],
                                             value="all", size="sm", className="sa-filter"),
                        html.Div(timeline_view(r, "all", selected), id="timeline", className="sa-block"),
                    ], "timeline-section"),
            section(2, "Why did it move?", "The change split into causes, the note for the manager and what the "
                    "answer key says.", html.Div(anomaly_view(r, selected), id="anomaly"), "anomaly-section"),
            section(3, "All anomalies", "Newest first; open one to see its causes.", table_view(r), "table-section"),
            section(4, "Against the answer key", "How many of the injected and labelled events the check found.",
                    truth_view(r), "truth-section"),
            dmc.Text(f"Data: Lumo, a fictional shop from Data-Generator v1.7.1 (seed {r.summary['seed']}), "
                     f"{r.summary['period'][0]} to {r.summary['period'][1]}; checked from "
                     f"{r.summary['detection_start']}. Amounts in PLN without VAT.", className="dg-caption sa-block"),
        ]
    footer = html.Footer([
        html.P("All orders, visits and costs are generated by Data-Generator: a fictional shop. The anomalies are "
               "injected by this project or labelled by the generator, so every alert can be graded."),
        html.A(REPO_URL.removeprefix("https://"), href=REPO_URL, target="_blank", className="dg-link"),
    ], className="dg-footer")
    return dmc.MantineProvider(theme=t.MANTINE_THEME, children=html.Div([
        dcc.Location(id="url", refresh=False), dcc.Store(id="selected"), dcc.Store(id="scroll-to"),
        dcc.Store(id="scroll-done"),
        header,
        html.Main(html.Div([*body, footer], className="dg-wrap")),
    ], className="dg-page"))


# ---------------------------------------------------------------------------------------------------------------- #
# App and callbacks

app = Dash(__name__, title="Sales Anomalies", assets_folder=str(Path(__file__).parent / "assets"),
           suppress_callback_exceptions=True)
app.layout = page


@app.callback(Output("selected", "data"), Input("url", "search"))
def apply_address(search):
    """The anomaly from the address when the page opens (/?anomaly=12), else the biggest real one."""
    query = {key: values[0] for key, values in parse_qs((search or "").lstrip("?")).items()}
    return start_anomaly(results(), query.get("anomaly"))


@app.callback(
    Output("selected", "data", allow_duplicate=True), Output("scroll-to", "data"),
    Input("timeline-graph", "clickData"), Input({"type": "open-anomaly", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def choose(click, opened):
    """A dot on the timeline or a row of the table opens that anomaly."""
    chosen = clicked_anomaly(ctx.triggered_id, click, ctx.triggered[0]["value"] if ctx.triggered else None)
    return (chosen, chosen) if chosen else (no_update, no_update)


def clicked_anomaly(triggered_id, click, value) -> int | None:
    if triggered_id == "timeline-graph" and click and click.get("points"):
        return int(click["points"][0]["customdata"])
    if isinstance(triggered_id, dict) and value:
        return int(triggered_id["index"])
    return None


@app.callback(Output("anomaly", "children"), Output("timeline", "children"), Input("selected", "data"),
              State("metric", "value"))
def show(selected, metric):
    if not selected:
        return no_update, no_update
    r = results()
    return anomaly_view(r, int(selected)), timeline_view(r, metric, int(selected))


@app.callback(Output("timeline", "children", allow_duplicate=True), Input("metric", "value"),
              State("selected", "data"), prevent_initial_call=True)
def filter_metric(metric, selected):
    return timeline_view(results(), metric, selected)


app.clientside_callback(
    """function (anomalyId) {
        if (anomalyId) {
            setTimeout(function () {
                var target = document.getElementById('anomaly-section');
                if (target) { target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
            }, 150);
        }
        return window.dash_clientside.no_update;
    }""",
    Output("scroll-done", "data"), Input("scroll-to", "data"), prevent_initial_call=True,
)


def main():
    app.run(host="127.0.0.1", port=int(os.environ.get("SA_PORT", 8050)), debug=False)


if __name__ == "__main__":
    main()
