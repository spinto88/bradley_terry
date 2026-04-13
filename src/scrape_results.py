import json
import pandas as pd
from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL = "https://www.ligaprofesional.ar/torneo-apertura-2026/"
OUTPUT_PATH = "../data/results.json"
N_FECHAS = 16


def get_page_source(url):
    driver = webdriver.Chrome()
    try:
        driver.get(url)
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, "table")))
        return driver.page_source
    finally:
        driver.quit()


def output_match(match_dict):
    if match_dict["goles_local"] > match_dict["goles_visitante"]:
        return (match_dict["local"], match_dict["visitante"], 1)
    elif match_dict["goles_local"] < match_dict["goles_visitante"]:
        return (match_dict["local"], match_dict["visitante"], -1)
    else:
        return (match_dict["local"], match_dict["visitante"], 0)


def parse_results(html_content, n_fechas):
    soup = BS(html_content, "html.parser")
    tables = soup.find_all("table")

    all_results = []
    for fecha in range(n_fechas):
        try:
            results = pd.read_html(str(tables[fecha]))[0]
            results = results[results[0] == "TC"].copy()
            results.dropna(axis=1, how="all", inplace=True)
            results.rename(
                columns={1: "local", 2: "goles_local", 4: "goles_visitante", 5: "visitante"},
                inplace=True,
            )
            results.reset_index(drop=True, inplace=True)
            all_results += [output_match(results.iloc[i].to_dict()) for i in range(results.shape[0])]
        except Exception:
            pass

    return all_results


def main():
    print(f"Scraping {URL} ...")
    html_content = get_page_source(URL)
    all_results = parse_results(html_content, N_FECHAS)
    print(f"Partidos encontrados: {len(all_results)}")

    with open(OUTPUT_PATH, "w") as fp:
        fp.write(json.dumps(all_results))

    print(f"Resultados guardados en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
