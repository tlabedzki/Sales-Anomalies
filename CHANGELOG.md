# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [1.0.1] - 2026-11-27

### Added

- README section "Why these tools": what each key tool does in the project, why it fits how the data is read and
  rebuilt from a fresh clone, and which alternative was considered and why it was dropped.

### Fixed

- The version in `pyproject.toml` and in the package now follows the releases (both still said 0.1.0).

## [1.0.0] - 2026-11-25

### Added

- Daily anomaly check of the fictional shop Lumo (Data-Generator v1.7.1, 2021-2024): expected values from a Poisson
  regression fitted only on the past, daily and 7-day z-scores on 112 segments and five metrics, thresholds chosen on
  a second seed.
- Injected scenarios of nine kinds with their exact impact, and grading against them and `anomaly_data`.
- Breakdown of every anomaly into causes in PLN that add up to the grosz (margin bridge, LMDI for volume).
- A one-sentence note for the manager on every anomaly, built from the computed facts, every number checked by
  code.
- Dash + Mantine app in the portfolio look: anomaly timeline, cause bridge, note and answer key.
- README as a case study in English, with an animated hero of the app (light and dark) and four charts.

[1.0.1]: https://github.com/tlabedzki/Sales-Anomalies/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/tlabedzki/Sales-Anomalies/releases/tag/v1.0.0
