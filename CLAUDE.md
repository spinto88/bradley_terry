# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that fits a Bradley-Terry model (with a three-outcome extension for ties) to Argentine Liga
Profesional football results — Torneo Apertura 2026 and Torneo Clausura 2026 combined, **group stage
("fase de grupos") only, playoffs excluded** — and publishes an interactive ranking/simulator, with
bootstrap-based uncertainty bands, as a static site via GitHub Pages (`docs/`).

## Pipeline / data flow

```
src/scraping/wikipedia_source.py   →  data/matches_apertura_2026.json
        (requests + BeautifulSoup)     data/matches_clausura_2026.json
        (manual step, not in CI)       (one JSON list per tournament; each match is a dict with
                                         season/fecha/zona/local/visitante/goles_*/resultado)

src/build_site_data.py             →  docs/data/infered_score.json  (point-estimate scores, both models)
        (orchestrator, run by CI)      docs/data/bootstrap.json      (per-team mean/std/percentiles)
                                        docs/data/results.json        (combined match tuples, both tournaments)

docs/index.html                       reads all three JSON files via fetch() and renders the ranking
        (Plotly, vanilla JS)           chart (with error bars), the mathematical explanation, and the
                                        head-to-head win-probability simulator
```

- `data/matches_<season>.json` are the raw scraped inputs, one file per tournament — this lets the
  in-progress Clausura be re-scraped and overwritten independently of the already-finished Apertura.
  `src/config.load_all_matches()` concatenates all files in `MATCH_FILES` for downstream use.
- Scraping is a **manual, local step** — it is deliberately not part of the CI workflow, since the
  Wikipedia Clausura page is edited live while the tournament is in progress. Re-run the scraper by hand,
  review the diff on `data/matches_*.json`, then push (which triggers CI to refit).
- `src/build_site_data.py` is idempotent: rerun it after editing `data/matches_*.json` to refresh all
  three `docs/data/*.json` outputs.
- `TorneoArgentino.ipynb` is the original exploratory notebook the pipeline was extracted from (predates
  the Wikipedia source and the package split). Not kept in sync — treat `src/` as the source of truth.

## Data source: Wikipedia, not the league's own site

The scraper (`src/scraping/wikipedia_source.py`) reads the Spanish Wikipedia "Anexo" pages for each
tournament (`Anexo:Torneo Apertura 2026 (Argentina)`, `Anexo:Torneo Clausura 2026 (Argentina)`), listed
in `SOURCES`. These pages are static HTML with a consistent per-tournament template: one `<table
class="wikitable ...">` per "Fecha N", containing `Zona A` / `Zona B` / `Interzonal(es)` sub-sections and
match rows `[local, "N - M", visitante, estadio, fecha, hora]`. Group-stage tables are identified by
`_is_fecha_table()` (first row matches `Fecha \d+`) — this is what excludes the playoff bracket, which
Wikipedia renders with a structurally different table type. Rows with a non-numeric result (`"-"`) are
unplayed fixtures and are skipped. Adding a future tournament (e.g. Apertura 2027) is just adding an entry
to `SOURCES`, since all tournaments share the same table template.

The previous scraper (Selenium against `ligaprofesional.ar`) was replaced because this Wikipedia format
needs no browser/chromedriver, and separates group stage from playoffs structurally instead of by parsing
ad hoc status strings.

## Model (`src/modeling/`)

Two variants are fit on every run, both via `scipy.optimize.minimize` (L-BFGS-B) on a Gaussian-prior
log-posterior over per-team strength scores. The log-posteriors (`bradley_terry.py`) are vectorized over
numpy arrays of match indices (not a Python loop per match) — this matters because the bootstrap refits
the model hundreds of times per build, and the loop-based version made that impractically slow (~3s/fit →
~25 min for 500 replicas; vectorized: ~0.1s/fit).

- **Basic model** (`fit_basic` / `log_posterior_basic`): one score per team. A draw is modeled as a third
  "virtual" outcome with strength `exp(0.5*(s_i+s_j))`, so win/draw/loss probabilities for `i` vs `j` are
  `exp(s_i)/Z`, `exp(s_j)/Z`, `exp(0.5*(s_i+s_j))/Z`.
- **Home-advantage model** (`fit_home` / `log_posterior_home`): adds a non-negative per-team home-advantage
  term `h_i` (constrained via `bounds`), added to the local team's effective score.

Team indices (`src/modeling/infer.py::infer_scores`) are derived by sorting the unique team names
alphabetically (`sorted(set(...))`), not by any external ID — keep this in mind if comparing runs or
diffing output team order.

Output JSON shape (`docs/data/infered_score.json`, unchanged from before the reorg):
```json
{"results": [{"team": ..., "score": ...}, ...],
 "results_home": [{"team": ..., "score": ..., "h_home": ...}, ...],
 "metadata": {"date_of_creation": "..."}}
```

## Uncertainty (`src/uncertainty/bootstrap.py`)

`run_bootstrap()` measures how sensitive each team's score is to sampling noise, via a **per-team
stratified bootstrap** (`bootstrap_replicate()`): for each replica, for each team, resample with
replacement as many matches as that team actually played, drawn from that team's own match list (where it
appears as either local or visitante). This deliberately does not deduplicate a match that gets picked
independently via both of its teams' lists — each appearance is a valid data point for the log-likelihood,
and deduplicating would break the "each team contributes ~N matches" property. This is what corrects the
bias a naive whole-pool bootstrap would have (over/under-representing teams that have played more or fewer
matchdays so far, e.g. Apertura's 16 vs. Clausura's partial count).

Output (`docs/data/bootstrap.json`): `{"n_replicas", "seed", "results": {team: {mean, std, p5, p50, p95}},
"results_home": {team: {"score": {...}, "h_home": {...}}}, "raw_scores": {team: [500 floats]}}`.
`raw_scores` holds the basic model's per-replica score for every team, **aligned by replica index across
teams** (every team appears in every replica since `bootstrap_replicate` always draws >=1 of its own
matches) — this is what lets the frontend simulator pair up team A's and team B's score from the *same*
replica instead of just plugging in each team's marginal mean (see below).

## Frontend (`docs/index.html`)

Single self-contained HTML file (no build step, no bundler) using Plotly via CDN:
- Fetches `data/infered_score.json`, `data/results.json`, and `data/bootstrap.json` client-side (relative
  paths — the page must be served/hosted from `docs/`, e.g. via GitHub Pages, not opened as a bare
  `file://` URL, or the `fetch()` calls will fail under CORS).
- Renders a horizontal area chart of team scores (`chart-basico`, sorted ascending) with asymmetric
  Plotly `error_x` bars driven by each team's bootstrap `p5`/`p95`/`mean`.
- Implements a head-to-head simulator (`calcProbs()`) that, for each bootstrap replica, computes
  win/draw/loss probabilities from that replica's paired `(score_i, score_j)` and averages across
  replicas — deliberately not a single plug-in estimate from each team's point/mean score, so a team's
  score uncertainty propagates into the simulated probability instead of being averaged away first. Shows
  the resulting mean plus a 90% range under each percentage, and historical head-to-head results, across
  both tournaments (`data/results.json` is the concatenation of both).
- The explanatory text is intentionally written to stay generic (no hardcoded team names/results) because
  the Clausura is in progress and the ranking changes on every rebuild — don't reintroduce hardcoded
  "team X is the strongest" prose here.

When editing this file, changes are purely client-side JS/HTML/CSS edits — reload the page (served,
not `file://`) to verify.

## Commands

```bash
pip install -r requirements.txt        # numpy, scipy, requests, beautifulsoup4

python -m src.scraping.wikipedia_source   # manual step: re-scrapes both Wikipedia "Anexo" pages,
                                           # overwrites data/matches_apertura_2026.json and
                                           # data/matches_clausura_2026.json

python -m src.build_site_data             # fits both models + runs the bootstrap (~500 replicas,
                                           # ~1 min), writes all three docs/data/*.json files
                                           # (this is what CI runs — run it after editing data/matches_*.json)
```

There is no test suite, linter, or build step in this repo.

## CI (`.github/workflows/build.yml`)

On every push to `main` that touches `data/**` or `src/**`, GitHub Actions installs
`requirements.txt` and runs `python -m src.build_site_data`, then auto-commits the regenerated
`docs/data/infered_score.json`, `docs/data/bootstrap.json`, and `docs/data/results.json` back to `main`
as `github-actions`. **Scraping is not part of this workflow** — see the manual step above.
