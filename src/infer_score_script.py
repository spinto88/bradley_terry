import json
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "results.json"
OUTPUT_PATH = BASE_DIR / "docs" / "data" / "infered_score.json"


# ── Modelo básico (sin localía) ──────────────────────────────────────────────

def log_posterior_basic(scores, matches, sigma=1.0):
    ll = 0.0
    for i, j, y in matches:
        si, sj = scores[i], scores[j]
        zi = np.exp(si)
        zj = np.exp(sj)
        zt = np.exp(0.5 * (si + sj))
        Z = zi + zj + zt
        if y == 1:
            ll += np.log(zi / Z)
        elif y == -1:
            ll += np.log(zj / Z)
        else:
            ll += np.log(zt / Z)
    prior = -0.5 * np.sum(scores ** 2) / sigma ** 2
    return ll + prior


def fit_basic(matches, N, sigma=1.0):
    def objective(x):
        return -log_posterior_basic(x, matches, sigma)
    res = minimize(objective, np.zeros(N), method="L-BFGS-B")
    return res.x


# ── Modelo con ventaja de local ──────────────────────────────────────────────

def log_posterior_home(theta, matches, N, sigma=1.0):
    s = theta[:N]
    h = theta[N:2 * N]
    ll = 0.0
    for i, j, y in matches:
        si = s[i] + h[i]   # score efectivo del local
        sj = s[j]           # score efectivo del visitante
        a = np.exp(si)
        b = np.exp(sj)
        c = np.exp(0.5 * (si + sj))
        Z = a + b + c
        if y == 1:
            ll += np.log(a / Z)
        elif y == -1:
            ll += np.log(b / Z)
        else:
            ll += np.log(c / Z)
    prior = -0.5 * np.sum(theta ** 2) / sigma ** 2
    return ll + prior


def fit_home(matches, N, sigma=1.0):
    def objective(x):
        return -log_posterior_home(x, matches, N, sigma)
    res = minimize(objective, np.zeros(2 * N), method="L-BFGS-B")
    return res.x


# ── Pipeline principal ───────────────────────────────────────────────────────

def infer_scores(list_of_matches):
    teams = sorted(set(item for match in list_of_matches for item in match[:2]))
    N = len(teams)
    name_to_idx = {name: i for i, name in enumerate(teams)}

    matches_num = [
        (name_to_idx[m[0]], name_to_idx[m[1]], m[2])
        for m in list_of_matches
    ]

    # Modelo básico
    scores_basic = fit_basic(matches_num, N)
    results_basic = [
        {"team": t, "score": scores_basic[name_to_idx[t]]}
        for t in teams
    ]

    # Modelo con localía
    theta_home = fit_home(matches_num, N)
    scores_home = theta_home[:N]
    h_home = theta_home[N:2 * N]
    results_home = [
        {
            "team": t,
            "score": scores_home[name_to_idx[t]],
            "h_home": h_home[name_to_idx[t]],
        }
        for t in teams
    ]

    return results_basic, results_home


if __name__ == "__main__":
    list_of_matches = json.load(open(DATA_PATH, "r"))

    results_basic, results_home = infer_scores(list_of_matches)

    output = {
        "results": results_basic,
        "results_home": results_home,
        "metadata": {
            "date_of_creation": str(datetime.today()),
        },
    }

    with open(OUTPUT_PATH, "w") as fp:
        fp.write(json.dumps(output))

    print(f"Done. {len(results_basic)} equipos procesados.")
