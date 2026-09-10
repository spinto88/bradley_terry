from datetime import datetime


def compute_standings(matches_dicts):
    """Tabla de posiciones clásica (victoria=3, empate=1, derrota=0) sobre la fase de grupos de
    Apertura + Clausura combinadas. Desempate: puntos, diferencia de gol, goles a favor, nombre."""
    stats = {}
    for m in matches_dicts:
        local, visitante = m["local"], m["visitante"]
        gl, gv = m["goles_local"], m["goles_visitante"]
        for t in (local, visitante):
            stats.setdefault(t, {
                "played": 0, "wins": 0, "draws": 0, "losses": 0,
                "goals_for": 0, "goals_against": 0,
            })
        stats[local]["played"] += 1
        stats[visitante]["played"] += 1
        stats[local]["goals_for"] += gl
        stats[local]["goals_against"] += gv
        stats[visitante]["goals_for"] += gv
        stats[visitante]["goals_against"] += gl
        if gl > gv:
            stats[local]["wins"] += 1
            stats[visitante]["losses"] += 1
        elif gl < gv:
            stats[visitante]["wins"] += 1
            stats[local]["losses"] += 1
        else:
            stats[local]["draws"] += 1
            stats[visitante]["draws"] += 1

    table = []
    for team, s in stats.items():
        points = 3 * s["wins"] + s["draws"]
        goal_diff = s["goals_for"] - s["goals_against"]
        table.append({"team": team, **s, "goal_diff": goal_diff, "points": points})

    table.sort(key=lambda r: (-r["points"], -r["goal_diff"], -r["goals_for"], r["team"]))
    for i, row in enumerate(table, start=1):
        row["rank"] = i

    return {
        "table": table,
        "metadata": {"date_of_creation": str(datetime.today())},
    }
