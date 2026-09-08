from collections import defaultdict

import numpy as np

from ..modeling.infer import infer_scores


def bootstrap_replicate(matches, rng):
    """Bootstrap estratificado por equipo: para cada equipo, resamplea con
    reemplazo tantos partidos como jugó realmente, tomados de su propia lista
    de partidos (donde aparece como local o visitante).

    Un partido A-vs-B puede terminar apareciendo 0, 1 o 2+ veces en la réplica
    (elegido independientemente desde la lista de A y desde la de B) — es
    intencional: deduplicar rompería la propiedad buscada de que cada equipo
    aporte ~N partidos, y es justamente lo que corrige el sesgo de un bootstrap
    naive por-partido (que sobre-representaría a los equipos con más fechas
    jugadas)."""
    by_team = defaultdict(list)
    for m in matches:
        by_team[m[0]].append(m)
        by_team[m[1]].append(m)

    replicate = []
    for team_matches in by_team.values():
        n = len(team_matches)
        idx = rng.integers(0, n, size=n)
        replicate += [team_matches[i] for i in idx]
    return replicate


def _percentiles(values):
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def run_bootstrap(matches, n_replicas=500, seed=42):
    """matches: lista de (local, visitante, resultado). Devuelve un resumen
    (media/desvío/percentiles) por equipo y por modelo, a partir de n_replicas
    réplicas del modelo Bradley-Terry ajustado sobre bootstrap_replicate()."""
    rng = np.random.default_rng(seed)

    basic_scores = defaultdict(list)
    home_scores = defaultdict(list)
    home_adv = defaultdict(list)

    for _ in range(n_replicas):
        replicate = bootstrap_replicate(matches, rng)
        results_basic, results_home = infer_scores(replicate)
        for r in results_basic:
            basic_scores[r["team"]].append(r["score"])
        for r in results_home:
            home_scores[r["team"]].append(r["score"])
            home_adv[r["team"]].append(r["h_home"])

    results = {team: _percentiles(vals) for team, vals in basic_scores.items()}
    results_home = {
        team: {
            "score": _percentiles(home_scores[team]),
            "h_home": _percentiles(home_adv[team]),
        }
        for team in home_scores
    }

    # Cada equipo aparece exactamente una vez por réplica (ver bootstrap_replicate:
    # todo equipo aporta siempre >=1 de sus propios partidos), así que estas listas
    # están alineadas por índice de réplica entre equipos. Se exponen crudas (no solo
    # resumidas) para que el simulador de partidos pueda calcular P(gana/empate/pierde)
    # réplica por réplica, usando el par de scores (i, j) de la MISMA réplica en vez de
    # solo el score puntual — así la incertidumbre de cada equipo se propaga a la
    # probabilidad simulada en lugar de perderse al promediar los scores primero.
    raw_scores = {team: [float(v) for v in vals] for team, vals in basic_scores.items()}

    return {
        "n_replicas": n_replicas,
        "seed": seed,
        "results": results,
        "results_home": results_home,
        "raw_scores": raw_scores,
    }
