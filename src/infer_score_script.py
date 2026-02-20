import json
import numpy as np
from scipy.optimize import minimize
from datetime import datetime

def log_likelihood(scores, matches):
    ll = 0.0
    for i, j, y in matches:
        si = scores[i]
        sj = scores[j]

        zi = np.exp(si)
        zj = np.exp(sj)
        zt = np.exp(0.5 * (si + sj))
        Z = zi + zj + zt

        if y == 1:
            ll += np.log(zi / Z)
        elif y == -1:
            ll += np.log(zj / Z)
        else:  # draw
            ll += np.log(zt / Z)
    return ll

def log_posterior(scores, matches, sigma=1.0):
    
    ll = log_likelihood(scores, matches)
    prior = -0.5 * np.sum(scores**2) / sigma**2
    
    return ll + prior

def fit_scores(matches, N, sigma=1.0):
    
    x0 = np.zeros(N)
    
    def objective(x):
        return -log_posterior(x, matches, sigma)

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B"
    )

    return res.x

def infer_score(list_of_matches):

    # List of teams
    teams = set([item for sublist in list_of_matches for item in sublist[:2]])
    N = len(teams)

    name_to_index = {name: i for i, name in enumerate(teams)}
    index_to_name = {value: key for key, value in name_to_index.items()}

    list_of_matches_num = [(name_to_index[m[0]], name_to_index[m[1]], m[2]) for m in list_of_matches]

    scores_hat = fit_scores(list_of_matches_num, N)

    ans = [{"team": team, "score": scores_hat[name_to_index[team]]} for team in teams]

    return ans

if __name__ == "__main__":

    list_of_matches = json.load(open("../data/results.json","r"))

    infered_score = infer_score(list_of_matches)

    ans = {"results": infered_score, "metadata": {"date_of_creation": str(datetime.today())}}

    with open("../data/infered_score.json", "w") as fp:
        fp.write(json.dumps(ans))




