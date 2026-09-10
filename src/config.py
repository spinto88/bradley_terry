import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DATA_DIR = BASE_DIR / "docs" / "data"

MATCH_FILES = [
    DATA_DIR / "matches_apertura_2026.json",
    DATA_DIR / "matches_clausura_2026.json",
]

SCORES_PATH = DOCS_DATA_DIR / "infered_score.json"
BOOTSTRAP_PATH = DOCS_DATA_DIR / "bootstrap.json"
RESULTS_COPY_PATH = DOCS_DATA_DIR / "results.json"
NETWORK_PATH = DOCS_DATA_DIR / "network.json"
STANDINGS_PATH = DOCS_DATA_DIR / "standings.json"


def load_all_matches(files=MATCH_FILES):
    """Concatena los partidos de fase de grupos de todos los torneos listados."""
    matches = []
    for path in files:
        with open(path) as fp:
            matches += json.load(fp)
    return matches


def as_bt_tuples(matches):
    """Convierte partidos (dicts enriquecidos) al formato (local, visitante, resultado)
    que consumen fit_basic/fit_home."""
    return [(m["local"], m["visitante"], m["resultado"]) for m in matches]


def write_json(path, data):
    with open(path, "w") as fp:
        fp.write(json.dumps(data, ensure_ascii=False))
