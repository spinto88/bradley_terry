from .bradley_terry import fit_basic, fit_home


def infer_scores(list_of_matches):
    """list_of_matches: lista de (local, visitante, resultado) con resultado en {1,-1,0}.
    Devuelve (results_basic, results_home)."""
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
