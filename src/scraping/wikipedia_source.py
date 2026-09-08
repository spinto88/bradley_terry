import json
import re

import requests
from bs4 import BeautifulSoup

from ..config import DATA_DIR

FECHA_HEADER_RE = re.compile(r"^Fecha\s+(\d+)")
RESULT_RE = re.compile(r"^(\d+)\s*-\s*(\d+)$")

HEADERS = {"User-Agent": "Mozilla/5.0 (bradley-terry data collection)"}

SOURCES = [
    {
        "season": "apertura_2026",
        "url": "https://es.wikipedia.org/wiki/Anexo:Torneo_Apertura_2026_(Argentina)",
        "output": DATA_DIR / "matches_apertura_2026.json",
    },
    {
        "season": "clausura_2026",
        "url": "https://es.wikipedia.org/wiki/Anexo:Torneo_Clausura_2026_(Argentina)",
        "output": DATA_DIR / "matches_clausura_2026.json",
    },
]


def _is_fecha_table(table):
    """Las tablas de fase de grupos son wikitables cuya primera fila es 'Fecha N'.
    Los playoffs usan un tipo de tabla distinto (sin class "wikitable"), y otras
    wikitables de la página (goleadores, entrenadores) no empiezan con 'Fecha N'."""
    first_row = table.find("tr")
    if first_row is None:
        return False
    text = first_row.get_text(" ", strip=True)
    return FECHA_HEADER_RE.match(text) is not None


def _parse_fecha_table(table, season):
    fecha_actual = None
    zona_actual = None
    matches = []

    for tr in table.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if not cells:
            continue

        if len(cells) == 1:
            m = FECHA_HEADER_RE.match(cells[0])
            if m:
                fecha_actual = int(m.group(1))
            else:
                zona_actual = cells[0]
            continue

        if cells[0] == "Local" and cells[1] == "Resultado":
            continue  # fila de encabezado de columnas

        if len(cells) < 3:
            continue

        local, resultado, visitante = cells[0], cells[1], cells[2]
        m = RESULT_RE.match(resultado)
        if not m:
            continue  # partido todavía no jugado ("-")

        goles_local, goles_visitante = int(m.group(1)), int(m.group(2))
        resultado_num = 1 if goles_local > goles_visitante else (-1 if goles_local < goles_visitante else 0)

        matches.append({
            "season": season,
            "fecha": fecha_actual,
            "zona": zona_actual,
            "local": local,
            "visitante": visitante,
            "goles_local": goles_local,
            "goles_visitante": goles_visitante,
            "resultado": resultado_num,
        })

    return matches


def fetch_group_stage(url, season):
    """Descarga una página "Anexo:Torneo X (Argentina)" de Wikipedia y devuelve
    únicamente los partidos de fase de grupos (zonas + interzonales) ya jugados,
    excluyendo la fase de playoffs."""
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    matches = []
    for table in soup.find_all("table", class_="wikitable"):
        if _is_fecha_table(table):
            matches += _parse_fecha_table(table, season)
    return matches


def main():
    for source in SOURCES:
        print(f"Scraping {source['url']} ...")
        matches = fetch_group_stage(source["url"], source["season"])
        print(f"  Partidos de fase de grupos encontrados: {len(matches)}")
        with open(source["output"], "w") as fp:
            fp.write(json.dumps(matches, ensure_ascii=False))
        print(f"  Guardado en {source['output']}")


if __name__ == "__main__":
    main()
