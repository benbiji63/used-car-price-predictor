"""Generate notebook 07, the cross-model comparison and analysis notebook."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


SETUP = '''from pathlib import Path
import sys, warnings, json
PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from src import config, data, metrics, plots, training, comparison

plots.set_style()
FIGURES_DIR = config.ARTIFACTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f"Loaded comparison helpers | seed={config.SEED}")'''


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        """# 07 — Model Comparison, Ensembles, Ablations, and Generalization

**Goal.** Compare all six saved models on the untouched test set, verify that metrics are reproducible, evaluate simple ensembles, quantify the leakage effect of `Original_Purchase_Price_INR`, test generalization to unseen car models and future model years, check the Delhi-NCR penalty, and select the dashboard winner.

The test set is used only for final evaluation and comparison. All tuning and refitting used train/validation data only.
"""
    ))
    cells.append(nbf.v4.new_code_cell(SETUP))
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Load data, test split, and saved metrics\n\nThe canonical split indices ensure the comparison uses exactly the same test rows as notebooks 01–06."
    ))
    cells.append(nbf.v4.new_code_cell(
        "raw = data.load_raw_data()\n"
        "data.validate_data(raw)\n"
        "train, val, test = data.split_frames(raw)\n"
        "saved_metrics = training.load_metrics()\n"
        "print('Saved model keys:', sorted(k for k in saved_metrics if k in config.MODEL_NAMES))"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Recompute test predictions and verify metrics.json\n\nEach saved pipeline is reloaded and evaluated independently. Any mismatch with `metrics.json` raises an error before the analysis proceeds."
    ))
    cells.append(nbf.v4.new_code_cell(
        "pred_frame, recomputed = comparison.recompute_test_predictions(config.MODEL_NAMES, test)\n"
        "check = comparison.assert_metrics_match(recomputed, saved_metrics)\n"
        "display(check[check['Metric'].isin(['r2_log', 'rmse_log', 'mape'])].pivot(index='Model', columns='Metric', values='abs_diff'))\n"
        "print('Maximum absolute difference:', float(check['abs_diff'].max()))\n"
        "pred_frame.to_parquet(config.TEST_PREDICTIONS_PATH, index=False)\n"
        "print('Saved test predictions to', config.TEST_PREDICTIONS_PATH)"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Leaderboard\n\nLower log-RMSE, MAE, RMSE, MAPE, and median APE are better; higher log-R² is better. The best value per metric is highlighted."
    ))
    cells.append(nbf.v4.new_code_cell(
        "leaderboard = comparison.leaderboard(pred_frame, saved_metrics)\n"
        "display(\n"
        "    leaderboard.drop(columns=['key']).style\n"
        "    .format({'r2_log': '{:.5f}', 'rmse_log': '{:.5f}', 'mae_inr': '₹{:,.0f}',\n"
        "             'rmse_inr': '₹{:,.0f}', 'mape': '{:.2f}%', 'median_ape': '{:.2f}%',\n"
        "             'train_time_s': '{:.1f}s', 'inference_ms_per_1k': '{:.2f}ms', 'size_mb': '{:.2f}MB'})\n"
        "    .highlight_min(subset=['rmse_log', 'mae_inr', 'rmse_inr', 'mape', 'median_ape'], color='#d4edda')\n"
        "    .highlight_max(subset=['r2_log'], color='#d4edda')\n"
        ")"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Ensembles\n\nThe all-model and top-3 averages are computed in log space, which is a geometric mean of prices and matches the multiplicative generator."
    ))
    cells.append(nbf.v4.new_code_cell(
        "all_names = list(config.MODEL_NAMES)\n"
        "top3_names = leaderboard.head(3)['key'].tolist()\n"
        "ensemble_all = comparison.evaluate_ensemble(pred_frame, all_names)\n"
        "ensemble_top3 = comparison.evaluate_ensemble(pred_frame, top3_names)\n"
        "print('All-model ensemble:', json.dumps(ensemble_all, indent=2))\n"
        "print('Top-3 ensemble:', json.dumps(ensemble_top3, indent=2))\n"
        "print('Top-3 members:', top3_names)\n"
        "training.append_metrics('ensemble_average', {'model_label': 'Ensemble (all)', 'members': all_names, 'test': ensemble_all})\n"
        "training.append_metrics('ensemble_top3', {'model_label': 'Ensemble (top-3)', 'members': top3_names, 'test': ensemble_top3})"
    ))
    cells.append(nbf.v4.new_code_cell(
        "ensemble_table = pd.DataFrame([\n"
        "    {'Ensemble': 'All 6 models', **ensemble_all},\n"
        "    {'Ensemble': 'Top 3 models', **ensemble_top3},\n"
        "])\n"
        "display(ensemble_table.style.format({'r2_log': '{:.5f}', 'rmse_log': '{:.5f}', 'mae_inr': '₹{:,.0f}', 'rmse_inr': '₹{:,.0f}', 'mape': '{:.2f}%', 'median_ape': '{:.2f}%'}))"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Comparison plots"
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig = plots.plot_metric_bars(leaderboard)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_metric_bars.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "ape = comparison.ape_frame(pred_frame, sample_size=20_000)\n"
        "fig = plots.plot_ape_box(ape, 'Absolute percentage error by model (test sample)')\n"
        "fig.savefig(FIGURES_DIR / 'comparison_ape_box.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "preds = {config.MODEL_LABELS[name]: pred_frame[name].to_numpy(float) for name in config.MODEL_NAMES}\n"
        "fig = plots.plot_small_multiples(preds, pred_frame['y_true_inr'].to_numpy(float), 'Predicted vs actual — small multiples')\n"
        "fig.savefig(FIGURES_DIR / 'comparison_small_multiples.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "segments = comparison.segment_error_frame(pred_frame, test, sample_size=30_000)\n"
        "fig = plots.plot_error_by_segments_all(segments)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_error_segments.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig = plots.plot_train_time_vs_error(leaderboard)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_train_vs_error.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()\n"
        "fig = plots.plot_time_size(leaderboard)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_time_size.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_code_cell(
        "corr = comparison.residual_correlation(pred_frame)\n"
        "fig = plots.plot_residual_correlation(corr)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_residual_corr.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Ablation: Original_Purchase_Price_INR\n\nThe default models exclude this generator input because real listings rarely provide it. Refitting the best model with it quantifies the leakage effect. This ablation is never eligible to become the dashboard winner."
    ))
    cells.append(nbf.v4.new_code_cell(
        "best_row = comparison.choose_best(leaderboard)\n"
        "best_name = str(best_row['key'])\n"
        "print('Best default model:', config.MODEL_LABELS[best_name])\n"
        "with open(config.params_path(best_name), encoding='utf-8') as handle:\n"
        "    best_params = json.load(handle)['params']\n"
        "ablation = comparison.run_ablation(best_name, best_params, train, test, best_row.to_dict())\n"
        "ablation_table = pd.DataFrame([\n"
        "    {'Setting': 'Without original price', 'RMSE(log)': best_row['rmse_log'], 'MAPE': best_row['mape']},\n"
        "    {'Setting': 'With original price', 'RMSE(log)': ablation['test']['rmse_log'], 'MAPE': ablation['test']['mape']},\n"
        "])\n"
        "display(ablation_table)"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 7. Generalization tests\n\n(a) Hold out every training row for 15 randomly chosen car models and evaluate on their test rows. (b) Hold out `Manufacturing_Year >= 2024` from training and evaluate on 2024–2026 test rows."
    ))
    cells.append(nbf.v4.new_code_cell(
        "car_deg, year_deg, holdout_models = comparison.run_generalization(train, test, pred_frame)\n"
        "print('Held-out car models:', holdout_models)\n"
        "display(car_deg.style.format({c: '{:.4f}' for c in ['baseline_rmse_log', 'holdout_rmse_log', 'degradation_pct', 'baseline_mape', 'holdout_mape']}))\n"
        "display(year_deg.style.format({c: '{:.4f}' for c in ['baseline_rmse_log', 'holdout_rmse_log', 'degradation_pct', 'baseline_mape', 'holdout_mape']}))"
    ))
    cells.append(nbf.v4.new_code_cell(
        "fig = plots.plot_generalization(car_deg, 'Generalization: 15 held-out car models')\n"
        "fig.savefig(FIGURES_DIR / 'comparison_generalization_cars.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()\n"
        "fig = plots.plot_generalization(year_deg, 'Generalization: Manufacturing_Year >= 2024')\n"
        "fig.savefig(FIGURES_DIR / 'comparison_generalization_years.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 8. Delhi-NCR restriction check\n\nCounterfactual predictions move the same diesel car between Delhi and Pune for age 9 (not restricted) and age 11 (restricted). Actual data ratios are shown for context."
    ))
    cells.append(nbf.v4.new_code_cell(
        "delhi = comparison.delhi_rule_check(test, config.MODEL_NAMES)\n"
        "display(delhi.pivot(index='Series', columns='Age', values='ratio').style.format('{:.3f}'))\n"
        "fig = plots.plot_delhi_rule(delhi)\n"
        "fig.savefig(FIGURES_DIR / 'comparison_delhi_rule.png', dpi=110, bbox_inches='tight')\n"
        "plt.show()"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 9. Paired bootstrap comparison of the top two models\n\nA paired bootstrap resamples test rows and estimates the 95% confidence interval for the RMSE(log) difference. A CI entirely above zero means the second model is worse."
    ))
    cells.append(nbf.v4.new_code_cell(
        "top2 = leaderboard.head(2)['key'].tolist()\n"
        "ci = metrics.bootstrap_rmse_difference(\n"
        "    pred_frame['y_true_inr'].to_numpy(float),\n"
        "    pred_frame[top2[0]].to_numpy(float),\n"
        "    pred_frame[top2[1]].to_numpy(float),\n"
        "    n_bootstrap=2000,\n"
        ")\n"
        "print('Top two:', [config.MODEL_LABELS[x] for x in top2])\n"
        "print(json.dumps(ci, indent=2))"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## 10. Final recommendation and winner artifact"
    ))
    cells.append(nbf.v4.new_code_cell(
        "best_payload = comparison.write_best_model(best_row, ensemble_all, ensemble_top3)\n"
        "print(json.dumps(best_payload, indent=2))\n"
        "print('Wrote', config.BEST_MODEL_PATH)"
    ))
    cells.append(nbf.v4.new_markdown_cell(
        "## Findings\n\n"
        "The leaderboard and bootstrap interval determine the winner. The all-model ensemble is reported alongside the individual models; it is available in the dashboard with any subset of models. "
        "The ablation quantifies how much the excluded original-price column would have leaked, and the generalization tables show whether performance survives unseen car models and future model years. "
        "The Delhi-NCR check verifies that the learned penalty drops between age 9 and age 11 for diesel cars. Because the data is synthetic, these results validate the pipeline and the generator's rules rather than the real used-car market."
    ))
    nb["cells"] = cells
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / "07_model_comparison.ipynb"
    nbf.write(nb, path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build()
