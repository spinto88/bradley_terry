from datetime import datetime

from .config import (
    BOOTSTRAP_PATH,
    NETWORK_PATH,
    RESULTS_COPY_PATH,
    SCORES_PATH,
    as_bt_tuples,
    load_all_matches,
    write_json,
)
from .modeling.infer import infer_scores
from .network.graph_metrics import compute_network_data
from .uncertainty.bootstrap import run_bootstrap


def main():
    matches_dicts = load_all_matches()
    matches = as_bt_tuples(matches_dicts)

    results_basic, results_home = infer_scores(matches)
    write_json(SCORES_PATH, {
        "results": results_basic,
        "results_home": results_home,
        "metadata": {"date_of_creation": str(datetime.today())},
    })

    bootstrap_summary = run_bootstrap(matches, n_replicas=500)
    write_json(BOOTSTRAP_PATH, bootstrap_summary)

    write_json(RESULTS_COPY_PATH, matches)

    network_data = compute_network_data(matches_dicts, results_basic)
    write_json(NETWORK_PATH, network_data)

    print(f"Done. {len(results_basic)} equipos procesados, {len(matches)} partidos combinados.")


if __name__ == "__main__":
    main()
