import numpy as np
from scipy.optimize import minimize


# ── Modelo básico (sin localía) ──────────────────────────────────────────────
#
# log_posterior_basic/home reciben los índices de partido ya vectorizados
# (idx_i, idx_j, y como arrays de numpy) en vez de iterar la lista de tuplas
# partido a partido: L-BFGS-B evalúa el objetivo decenas de veces por ajuste
# (una por diferencia finita de cada parámetro en cada iteración), así que un
# loop en Python por partido hacía que el bootstrap (cientos de reajustes)
# fuera impracticablemente lento. La matemática es idéntica, solo cambia cómo
# se computa.

def log_posterior_basic(scores, idx_i, idx_j, y, sigma=1.0):
    si, sj = scores[idx_i], scores[idx_j]
    zi = np.exp(si)
    zj = np.exp(sj)
    zt = np.exp(0.5 * (si + sj))
    Z = zi + zj + zt

    ll = np.where(y == 1, np.log(zi / Z), np.where(y == -1, np.log(zj / Z), np.log(zt / Z)))
    prior = -0.5 * np.sum(scores ** 2) / sigma ** 2
    return ll.sum() + prior


def fit_basic(matches, N, sigma=1.0):
    idx_i = np.array([m[0] for m in matches])
    idx_j = np.array([m[1] for m in matches])
    y = np.array([m[2] for m in matches])

    def objective(x):
        return -log_posterior_basic(x, idx_i, idx_j, y, sigma)

    res = minimize(objective, np.zeros(N), method="L-BFGS-B")
    return res.x


# ── Modelo con ventaja de local ──────────────────────────────────────────────

def log_posterior_home(theta, idx_i, idx_j, y, N, sigma=1.0):
    s = theta[:N]
    h = theta[N:2 * N]
    si = s[idx_i] + h[idx_i]   # score efectivo del local
    sj = s[idx_j]               # score efectivo del visitante
    a = np.exp(si)
    b = np.exp(sj)
    c = np.exp(0.5 * (si + sj))
    Z = a + b + c

    ll = np.where(y == 1, np.log(a / Z), np.where(y == -1, np.log(b / Z), np.log(c / Z)))
    prior = -0.5 * np.sum(s ** 2) / sigma ** 2
    prior += -np.sum(h)
    return ll.sum() + prior


def fit_home(matches, N, sigma=1.0):
    idx_i = np.array([m[0] for m in matches])
    idx_j = np.array([m[1] for m in matches])
    y = np.array([m[2] for m in matches])

    def objective(x):
        return -log_posterior_home(x, idx_i, idx_j, y, N, sigma)

    bounds = [(None, None)] * N + [(0.0, None)] * N  # s libre, h >= 0
    x0 = np.concatenate([np.zeros(N), np.full(N, 1.00)])
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    return res.x
