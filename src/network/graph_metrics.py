from datetime import datetime

import networkx as nx


def build_display_edges(matches_dicts):
    """Una entrada por partido jugado (máx. 2 por par de equipos): para una victoria, dirigida
    perdedor -> ganador; para un empate, sin dirección de dominancia real (type="draw"),
    el frontend la dibuja sin flecha."""
    edges = []
    for m in matches_dicts:
        local, visitante, resultado = m["local"], m["visitante"], m["resultado"]
        if resultado == 1:
            source, target = visitante, local
        elif resultado == -1:
            source, target = local, visitante
        else:
            source, target = local, visitante
        edges.append({
            "source": source,
            "target": target,
            "type": "draw" if resultado == 0 else "win",
            "season": m.get("season"),
            "zona": m.get("zona"),
        })
    return edges


def _add_weight(G, u, v, w):
    if G.has_edge(u, v):
        G[u][v]["weight"] += w
    else:
        G.add_edge(u, v, weight=w)


def build_weighted_digraph(matches_dicts):
    """Grafo dirigido ponderado para PageRank: victoria = peso 1 perdedor->ganador, empate =
    peso 0.5 en cada sentido (mismo tratamiento simétrico que el modelo Bradley-Terry le da al
    empate). Pesos acumulados si dos equipos vuelven a cruzarse con el mismo ganador."""
    G = nx.DiGraph()
    for m in matches_dicts:
        local, visitante, resultado = m["local"], m["visitante"], m["resultado"]
        G.add_node(local)
        G.add_node(visitante)
        if resultado == 1:
            _add_weight(G, visitante, local, 1.0)
        elif resultado == -1:
            _add_weight(G, local, visitante, 1.0)
        else:
            _add_weight(G, local, visitante, 0.5)
            _add_weight(G, visitante, local, 0.5)
    return G


def compute_network_data(matches_dicts, results_basic):
    """Arma el JSON de la visualización de red: por equipo, récord V-E-D, in-degree ponderado
    (wins + 0.5*draws), PageRank (análogo a eigenvector centrality para grafos dirigidos) y el
    score Bradley-Terry ya calculado, más la lista de aristas para dibujar."""
    bt_scores = {d["team"]: d["score"] for d in results_basic}
    teams = sorted(bt_scores)

    wins = {t: 0 for t in teams}
    draws = {t: 0 for t in teams}
    losses = {t: 0 for t in teams}
    for m in matches_dicts:
        local, visitante, resultado = m["local"], m["visitante"], m["resultado"]
        if resultado == 1:
            wins[local] += 1
            losses[visitante] += 1
        elif resultado == -1:
            wins[visitante] += 1
            losses[local] += 1
        else:
            draws[local] += 1
            draws[visitante] += 1

    G = build_weighted_digraph(matches_dicts)
    pagerank = nx.pagerank(G, alpha=0.85, weight="weight")

    nodes = [
        {
            "team": t,
            "wins": wins[t],
            "draws": draws[t],
            "losses": losses[t],
            "weighted_indegree": wins[t] + 0.5 * draws[t],
            "pagerank": pagerank.get(t, 0.0),
            "bt_score": bt_scores[t],
        }
        for t in teams
    ]

    return {
        "nodes": nodes,
        "edges": build_display_edges(matches_dicts),
        "metadata": {"date_of_creation": str(datetime.today())},
    }
