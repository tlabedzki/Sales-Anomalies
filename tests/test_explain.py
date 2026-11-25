"""Code checks the model: every number in a note must come from the sheet of facts."""
import json

import pytest

from sales_anomalies import explain, i18n
from sales_anomalies.decompose import Breakdown, Part


def breakdown(cause="discount", metric="margin"):
    parts = [Part("discount", cause, -3100.0, grosz=-310000), Part("prices", "prices", -321.0, grosz=-32100),
             Part("volume_traffic", "traffic", 100.0, grosz=10000)]
    return Breakdown(7, "Clothing|direct", metric, "down", "2024-03-12", "2024-03-14", 8000.0, 11321.0,
                     -332100, parts, cause, {"lines": 120.0, "expected_lines": 118.0, "discount_rate": 0.24,
                                             "expected_discount_rate": 0.05})


def test_number_formats():
    assert i18n.money(-4321, "pl", sign=True) == "−4 321 zł"
    assert i18n.money(12345.6, "en") == "PLN 12,346"
    assert i18n.percent(-0.293, "pl") == "−29%"
    assert i18n.period(__import__("pandas").Timestamp("2024-03-12"), __import__("pandas").Timestamp("2024-03-14"),
                       "pl") == "12–14 marca 2024"


def test_numbers_are_read_as_values_in_both_languages():
    assert explain.numbers("spadek o 4 321 zł (−29%) 12–14 marca 2024", "pl") == {4321, 29, 12, 14, 2024}
    assert explain.numbers("a drop of PLN 4,321.5", "en") == {4321.5}
    assert explain.numbers("3,5 tys.", "pl") == {3.5}


def test_check_rejects_an_invented_number_and_accepts_the_sheet():
    sheet = explain.facts(breakdown(), "pl")
    good = explain.template(sheet, "pl")
    assert explain.check(good, sheet, "pl")["ok"], explain.check(good, sheet, "pl")
    bad = good.replace(sheet["change"], "−3 400 zł")
    verdict = explain.check(bad, sheet, "pl")
    assert not verdict["ok"] and -3400 in verdict["invented"]   # a drop written with another value
    rounded = good + " To około 3,3 tys. zł."
    assert not explain.check(rounded, sheet, "pl")["ok"]


def test_check_limits_sentences_and_words():
    sheet = explain.facts(breakdown(), "en")
    assert not explain.check("One. Two. Three.", sheet, "en")["ok"]
    assert not explain.check(" ".join(["word"] * 60), sheet, "en")["ok"]


def test_sheet_holds_the_strings_the_note_may_use():
    sheet = explain.facts(breakdown(), "pl")
    assert sheet["segment"] == "Odzież · wejścia bezpośrednie"
    assert sheet["change"] == "−3 321 zł"
    assert sheet["main_cause"] == "rabaty"
    assert sheet["causes"][0]["amount"] == "−3 100 zł"
    assert any("24%" in item for item in sheet["evidence"])


class FakeWriter:
    """Stands in for the model: first an invented number, then a good note."""

    def __init__(self, texts):
        self.texts = list(texts)

    def write(self, sheet, lang, feedback=None):
        return self.texts.pop(0), {"input_tokens": 1000, "output_tokens": 100}


def test_note_retries_once_then_falls_back_to_the_template():
    sheet = explain.facts(breakdown(), "pl")
    good = explain.template(sheet, "pl")
    fixed = explain.note(FakeWriter(["Spadek o 9 999 zł.", good]), sheet, "pl")
    assert fixed["source"] == "model" and len(fixed["attempts"]) == 2
    assert fixed["cost_usd"] == pytest.approx(2 * (1000 * 0.10 + 100 * 0.50) / 1e6)
    failed = explain.note(FakeWriter(["Spadek o 9 999 zł.", "Spadek o 8 888 zł."]), sheet, "pl")
    assert failed["source"] == "template" and failed["text"] == good


def test_describe_without_a_key_uses_the_template_and_saves_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(explain, "ENV_FILE", tmp_path / ".env")
    path = tmp_path / "explanations.json"
    notes = explain.describe([breakdown()], path=path)
    assert all(n["source"] == "template" for n in notes.values())
    assert not path.exists()


def test_saved_notes_are_reused_when_the_facts_match(tmp_path, monkeypatch):
    monkeypatch.setattr(explain, "ENV_FILE", tmp_path / ".env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    b = breakdown()
    sheet = explain.facts(b, "en")
    key = f"{explain.anomaly_key(b)}|en"
    cache = {"model": explain.MODEL, "prompt_version": explain.PROMPT_VERSION, "items": {key: {
        "text": "saved", "source": "model", "facts_hash": explain.facts_hash(sheet),
        "prompt_version": explain.PROMPT_VERSION, "attempts": [{}], "usage": {"input_tokens": 1, "output_tokens": 1},
        "cost_usd": 0.0}}}
    path = tmp_path / "explanations.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    assert explain.describe([b], langs=("en",), path=path)[key]["text"] == "saved"


def test_api_key_is_read_from_the_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("# local\nANTHROPIC_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setattr(explain, "ENV_FILE", env)
    assert explain.api_key() == "test-key"


def test_eval_graders_pass_the_template_and_catch_a_wrong_direction():
    from evals.run import grade

    b = breakdown()
    case = {"main_cause": "discount", "direction": "down",
            "sheets": {lang: explain.facts(b, lang) for lang in ("pl", "en")}}
    for lang in ("pl", "en"):
        result = grade(explain.template(case["sheets"][lang], lang), case, lang)
        assert result["passed"], (lang, result)
    wrong = grade("Marża w Odzież · wejścia bezpośrednie wzrosła o −3 321 zł przez rabaty.", case, "pl")
    assert not wrong["grades"]["direction"]


def test_a_flipped_sign_is_an_invented_number():
    sheet = explain.facts(breakdown(), "pl")
    good = explain.template(sheet, "pl")
    flipped = good.replace(sheet["change"], sheet["change"].replace("−", "+"))
    assert not explain.check(flipped, sheet, "pl")["ok"]
    unsigned = good.replace(sheet["change"], sheet["change"].replace("−", ""))
    assert explain.check(unsigned, sheet, "pl")["ok"]   # a number without a sign may be the sheet's


def test_polish_plurals_in_the_readme():
    import sys

    sys.path.insert(0, str(explain.ROOT / "scripts"))
    from readme_blocks import plural_pl

    forms = ("alarm", "alarmy", "alarmów")
    assert [plural_pl(n, *forms) for n in (1, 2, 5, 12, 22, 25)] == \
        ["alarm", "alarmy", "alarmów", "alarmów", "alarmy", "alarmów"]


def test_main_runs_from_the_repository_root_without_pythonpath(tmp_path):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(explain.ROOT / "main.py"), "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
