"""The notes: every number in a note comes from the sheet of facts, with its sign."""

from sales_anomalies import i18n, notes
from sales_anomalies.data import ROOT
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
    assert notes.numbers("spadek o 4 321 zł (−29%) 12–14 marca 2024", "pl") == {4321, 29, 12, 14, 2024}
    assert notes.numbers("a drop of PLN 4,321.5", "en") == {4321.5}
    assert notes.numbers("3,5 tys.", "pl") == {3.5}


def test_check_rejects_an_invented_number_and_accepts_the_sheet():
    sheet = notes.facts(breakdown(), "pl")
    good = notes.template(sheet, "pl")
    assert notes.check(good, sheet, "pl")["ok"], notes.check(good, sheet, "pl")
    bad = good.replace(sheet["change"], "−3 400 zł")
    verdict = notes.check(bad, sheet, "pl")
    assert not verdict["ok"] and -3400 in verdict["invented"]   # a drop written with another value
    rounded = good + " To około 3,3 tys. zł."
    assert not notes.check(rounded, sheet, "pl")["ok"]


def test_check_limits_sentences_and_words():
    sheet = notes.facts(breakdown(), "en")
    assert not notes.check("One. Two. Three.", sheet, "en")["ok"]
    assert not notes.check(" ".join(["word"] * (notes.MAX_WORDS + 1)), sheet, "en")["ok"]


def test_sheet_holds_the_strings_the_note_may_use():
    sheet = notes.facts(breakdown(), "pl")
    assert sheet["segment"] == "Odzież · wejścia bezpośrednie"
    assert sheet["change"] == "−3 321 zł"
    assert sheet["main_cause"] == "rabaty"
    assert sheet["causes"][0]["amount"] == "−3 100 zł"
    assert any("24%" in item for item in sheet["evidence"])


def test_a_flipped_sign_is_an_invented_number():
    sheet = notes.facts(breakdown(), "pl")
    good = notes.template(sheet, "pl")
    flipped = good.replace(sheet["change"], sheet["change"].replace("−", "+"))
    assert not notes.check(flipped, sheet, "pl")["ok"]
    unsigned = good.replace(sheet["change"], sheet["change"].replace("−", ""))
    assert notes.check(unsigned, sheet, "pl")["ok"]   # a number without a sign may be the sheet's


def test_polish_plurals_in_the_readme():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from readme_blocks import plural_pl

    forms = ("alarm", "alarmy", "alarmów")
    assert [plural_pl(n, *forms) for n in (1, 2, 5, 12, 22, 25)] == \
        ["alarm", "alarmy", "alarmów", "alarmów", "alarmy", "alarmów"]


def test_main_runs_from_the_repository_root_without_pythonpath(tmp_path):
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(ROOT / "main.py"), "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_a_note_does_not_repeat_the_metric_as_its_cause():
    parts = [Part("conversion_mobile", "conversion", -3283.14, grosz=-328314),
             Part("conversion_desktop", "conversion", 28.12, grosz=2812)]
    b = Breakdown(5, "*|*", "conversion", "down", "2023-08-10", "2023-08-11", 3514.85, 6769.87, -325502, parts,
                  "conversion", {"orders": 485.0, "expected_orders": 934.0})
    for lang, part in (("en", "orders on mobile"), ("pl", "głównie zamówienia z telefonów")):
        sheet = notes.facts(b, lang)
        note = notes.template(sheet, lang)
        assert part in note and sheet["main_part"]["amount"] in note
        assert note.lower().count(sheet["metric"].lower()) == 1, note
        assert notes.check(note, sheet, lang)["ok"]
        assert note.count(":") == 1, note   # only after the days: no "głównie: … : …"


def test_notes_cover_every_breakdown_in_both_languages():
    written = notes.notes([breakdown()])
    assert set(written) == {f"{notes.anomaly_key(breakdown())}|{lang}" for lang in ("pl", "en")}
    assert all(notes.check(text, notes.facts(breakdown(), key[-2:]), key[-2:])["ok"]
               for key, text in written.items())
