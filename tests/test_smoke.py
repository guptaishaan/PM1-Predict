"""
Smoke tests: imports, baseline + extensions, expected artifacts exist.
Run from repo root:
  PYTHONPATH=. python3 -m pytest tests/test_smoke.py -v
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "extract_0.csv"


def _assert_png(path: Path):
    assert path.exists(), f"missing {path}"
    with open(path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path}"


def test_imports():
    sys.path.insert(0, str(REPO))
    from src.bootstrap_analysis import run_bootstrap
    from src.map_models import ridge_val_mae_path
    from src.distribution_fits import fit_normal
    assert run_bootstrap is not None


def test_pipeline_outputs():
    if not DATA.exists():
        import pytest
        pytest.skip("data/extract_0.csv not present")
    out = REPO / "outputs_test_run"
    cmd = [
        sys.executable, "-m", "src.run_pipeline",
        "--data_path", str(DATA),
        "--output_dir", str(out),
        "--run_bootstrap", "--bootstrap_B", "20",
        "--run_map",
        "--run_distribution_fit",
    ]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout

    figs = out / "figures"
    expected = [
        "timeseries_week.png",
        "parity_plot.png",
        "interval_coverage.png",
        "residual_vs_fitted.png",
        "residual_qq_normal.png",
        "bootstrap_mae_hist.png",
        "bootstrap_mse_hist.png",
        "bootstrap_coef_intervals.png",
        "map_vs_mle_bar.png",
        "ridge_coef_shrinkage.png",
        "ridge_val_mae_vs_alpha.png",
        "hourly_pm1_distribution_fit.png",
        "daily_pm1_distribution_fit.png",
        "residual_distribution_fit.png",
        "hourly_pm1_aic_comparison.png",
        "daily_pm1_aic_comparison.png",
        "residuals_aic_comparison.png",
    ]
    for name in expected:
        _assert_png(figs / name)

    for name in [
        "metrics.json", "best_model.json", "bootstrap_metrics.json",
        "map_vs_mle_comparison.csv", "best_ridge_model.json",
        "distribution_fits_hourly_pm1.csv",
    ]:
        assert (out / name).exists(), f"missing {name}"
