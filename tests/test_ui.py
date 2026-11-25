"""The look (Inter from the repo, Data-Generator colours) and the app's views, without a browser."""
import json
import re

import pandas as pd
import pytest

from sales_anomalies import data, figures, report, theme
from sales_anomalies.decompose import Breakdown, Part
from ui import theme as ui_theme

ASSETS = data.ROOT / "ui" / "assets"


def test_inter_comes_from_files_in_the_repo_and_no_font_service():
    css = (ASSETS / "fonts.css").read_text(encoding="utf-8")
    urls = re.findall(r"url\('([^']+)'\)", css)
    assert len(urls) == 2 and all((ASSETS / url).is_file() for url in urls)
    assert (ASSETS / "fonts" / "OFL-inter.txt").is_file()
    for path in [*ASSETS.rglob("*.css"), *(data.ROOT / "ui").glob("*.py")]:
        text = path.read_text(encoding="utf-8")
        assert "googleapis" not in text and "gstatic" not in text, path


def test_style_sheet_tokens_match_the_shared_colours():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    tokens = dict(re.findall(r"--dg-([a-z-]+):\s*([^;]+);", css))
    assert tokens["teal"] == theme.PRIMARY == "#006e79"
    assert tokens["copper"] == theme.ACCENT == "#c7692c"
    assert tokens["ink"] == theme.INK and tokens["muted"] == theme.MUTED and tokens["line"] == theme.LINE
    assert "'tnum' 1" in tokens["num-features"]
    assert ui_theme.MANTINE_THEME["colors"]["petrol"][ui_theme.MANTINE_THEME["primaryShade"]] == theme.PRIMARY


def a_breakdown(anomaly_id=1, change_grosz=-500000):
    parts = [Part("volume_traffic", "campaign", -4000.0, grosz=-400000), Part("discount", "discount", -900.0,
                                                                                   grosz=-90000),
             Part("prices", "prices", -100.0, grosz=-10000)]
    return Breakdown(anomaly_id, "*|social", "sales", "down", "2024-02-01", "2024-02-04", 15000.0, 20000.0,
                     change_grosz, parts, "campaign", {"sessions": 900.0, "expected_sessions": 2000.0,
                                                        "campaigns": ["Social 2024-01-31"], "lines": 50.0,
                                                        "expected_lines": 70.0})


@pytest.fixture()
def results(tmp_path, monkeypatch):
    """A tiny output/ folder like the one main.py writes."""
    b = a_breakdown()
    (tmp_path / "breakdowns.json").write_text(json.dumps([b.as_dict()], default=str), encoding="utf-8")
    pd.DataFrame([{"anomaly_id": 1, "segment": "*|social", "level": "channel", "start": "2024-02-01",
                   "end": "2024-02-04", "metrics": "sales", "main_metric": "sales", "direction": "down",
                   "z": -6.0, "impact": -5000.0, "verdict": "scenario", "events": "s01"}]).to_csv(
        tmp_path / "anomalies.csv", index=False)
    pd.DataFrame([{"event_id": "s01", "source": "scenario", "kind": "campaign_stop", "cause": "campaign",
                   "start": "2024-01-31", "end": "2024-02-04", "segments": "*|social", "metrics": "sales,margin",
                   "impact": -1500.0, "description": "campaign Social 2024-01-31 stops", "found": True,
                   "anomalies": "1"}]).to_csv(tmp_path / "events.csv", index=False)
    pd.DataFrame([{"anomaly_id": 1, "found": "campaign", "truth": "campaign", "right": True,
                   "source": "scenario"}]).to_csv(tmp_path / "causes.csv", index=False)
    days = pd.date_range("2024-01-01", "2024-03-01")
    pd.DataFrame({"date": days, "segment": "*|social", "metric": "sales", "observed": 5000.0,
                  "expected": 5000.0}).to_parquet(tmp_path / "scores.parquet")
    summary = {"seed": 42, "period": ["2021-01-01", "2024-12-31"], "detection_start": "2023-01-01", "anomalies": 1,
               "false_per_month": 0.0, "precision": 1.0, "recall_scenarios": 1.0, "scenarios_found": 1,
               "scenarios_total": 1, "breakdowns_exact_to_grosz": 1, "breakdowns": 1, "main_cause_right": 1,
               "main_cause_graded": 1, "segments": 112}
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr("sales_anomalies.explain.ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    return report.load(tmp_path)


def test_results_load_and_the_answer_key_reads_right(results):
    assert report.default_anomaly(results) == 1
    assert report.truth_line(results, 1).startswith("Answer key: injected")
    assert "named right" in report.truth_line(results, 1)
    table = report.anomaly_table(results)
    assert table.loc[0, "cause"] == "a campaign" and table.loc[0, "verdict"] == "injected"


def test_bridge_starts_at_expected_and_ends_at_actual():
    b = a_breakdown()
    rows = figures.bridge_data(b)
    assert rows["label"].iloc[0] == "Expected" and rows["label"].iloc[-1] == "Actual"
    assert rows["end"].iloc[-2] == pytest.approx(rows["end"].iloc[-1])   # the last part lands on the actual value
    fig = figures.bridge(b)
    colors = list(fig.data[0].marker.color)
    assert colors.count(theme.ACCENT) == 1                                # one copper mark: the main cause


def test_app_views_build(results, monkeypatch):
    from ui import app

    monkeypatch.setattr(app, "results", lambda: results)
    monkeypatch.setattr(app, "table", lambda: report.anomaly_table(results))
    assert app.start_anomaly(results, "1") == 1 and app.start_anomaly(results, "999") == 1
    assert app.anomaly_view(results, 1)
    assert app.timeline_view(results, "all", 1)
    assert app.table_view(results) and app.truth_view(results)
    assert app.clicked_anomaly("timeline-graph", {"points": [{"customdata": 1}]}, None) == 1
    assert app.clicked_anomaly({"type": "open-anomaly", "index": 1}, None, 1) == 1
    assert app.clicked_anomaly({"type": "open-anomaly", "index": 1}, None, None) is None
