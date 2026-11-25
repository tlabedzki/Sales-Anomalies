import sales_anomalies


def test_version_matches_pyproject():
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert sales_anomalies.__version__ == pyproject["project"]["version"]
