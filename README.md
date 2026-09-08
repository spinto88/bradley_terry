# Fútbol Argentino — Ranking Bradley-Terry

Ranking de fuerza de los equipos de la Liga Profesional Argentina (Torneo Apertura + Clausura 2026,
solo fase de grupos), estimado con un modelo de Bradley-Terry con empates y bandas de incertidumbre
por bootstrap. El resultado se publica como un sitio estático con GitHub Pages.

## Cómo funciona

1. **Scraping** (`src/scraping/`): descarga los resultados de las páginas "Anexo" de Wikipedia de cada
   torneo, quedándose solo con los partidos de fase de grupos (sin playoffs).
2. **Modelado** (`src/modeling/`): ajusta el modelo Bradley-Terry (con y sin ventaja de localía) para
   obtener un score de fortaleza por equipo.
3. **Incertidumbre** (`src/uncertainty/`): mide cuánto podría variar cada score con un bootstrap
   estratificado por equipo.
4. **Sitio** (`docs/`): página estática (Plotly + JS vanilla) con el ranking, sus bandas de
   incertidumbre, y un simulador de partidos entre dos equipos.

## Uso rápido

```bash
pip install -r requirements.txt

python -m src.scraping.wikipedia_source   # actualiza los datos scrapeados (manual)
python -m src.build_site_data             # reajusta el modelo y el bootstrap

cd docs && python3 -m http.server 8000    # previsualizar el sitio en http://localhost:8000
```

Ver [`CLAUDE.md`](CLAUDE.md) para el detalle de la arquitectura y el pipeline completo.
