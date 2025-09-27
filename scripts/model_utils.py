# scripts/model_utils.py

from pathlib import Path
from typing import Dict, Tuple

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr


# --------------------
# I/O for InferenceData
# --------------------

def save_idata(idata: az.InferenceData, idata_dir: str | Path, tag: str) -> Path:
    """Save InferenceData to <idata_dir>/<tag>.nc (creates dir if needed)."""
    idata_dir = Path(idata_dir)
    idata_dir.mkdir(parents=True, exist_ok=True)
    path = idata_dir / f"{tag}.nc"
    az.to_netcdf(idata, path)
    return path


def load_idata(tag: str, idata_dir: str | Path) -> az.InferenceData:
    """Load InferenceData from <idata_dir>/<tag>.nc."""
    return az.from_netcdf(Path(idata_dir) / f"{tag}.nc")


# --------------------
# Coverage utilities
# --------------------

def posterior_coverage(
    idata: az.InferenceData,
    var: str,
    q_pairs: Tuple[Tuple[float, float], ...] = ((0.025, 0.975), (0.1, 0.9), (0.25, 0.75)),
    dims: Tuple[str, str] = ("obs_idx", "λ_aφ"),
) -> pd.DataFrame:
    """
    Compute achieved coverage of central posterior predictive intervals.

    Parameters
    ----------
    idata : az.InferenceData
        Must contain observed_data[var] with dims (obs, band) and
        posterior_predictive[var] with dims (chain, draw, obs, band).
    var : str
        Variable name in both observed_data and posterior_predictive.
    q_pairs : tuple of (q_lo, q_hi)
        Central quantile bounds to evaluate (e.g., 95%, 80%, 50%).
    dims : (obs_dim, band_dim)
        Names of the observation and band dimensions.

    Returns
    -------
    DataFrame with columns: interval, band, coverage, nominal
    """
    # Use xarray to align by name (safer than assuming order)
    y_obs_da: xr.DataArray = idata.observed_data[var]
    y_ppc_da: xr.DataArray = idata.posterior_predictive[var]

    # Ensure dims exist
    assert dims[0] in y_obs_da.dims and dims[1] in y_obs_da.dims, "Observed dims mismatch"
    assert all(d in y_ppc_da.dims for d in ("chain", "draw", *dims)), "PPC dims mismatch"

    # Stack chain/draw to samples axis
    draws = y_ppc_da.stack(sample=("chain", "draw"))  # (sample, obs, band)

    rows = []
    for qlo, qhi in q_pairs:
        lo = draws.quantile(qlo, dim="sample")
        hi = draws.quantile(qhi, dim="sample")
        covered = ((y_obs_da >= lo) & (y_obs_da <= hi)).mean(dim=dims[0])  # per band
        for band, cov in zip(covered[dims[1]].values, covered.values):
            rows.append({
                "interval": f"{int((qhi - qlo) * 100)}%",
                "band": int(band),
                "coverage": float(cov),
            })

    out = pd.DataFrame(rows)
    out["nominal"] = out["interval"].map({"95%": 0.95, "80%": 0.8, "50%": 0.5})
    return out


# --------------------
# Model comparison
# --------------------

def compare_and_save(
    idatas: Dict[str, az.InferenceData],
    results_dir: str | Path,
    tag: str = "model_compare",
    ic: str = "loo",
    scale: str = "log",
    method: str = "stacking",
):
    """
    Run az.compare, save CSV and a LaTeX table (Table 1).

    Returns (cmp_df, out_csv_path, out_tex_path)
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cmp_df = az.compare(idatas, ic=ic, scale=scale, method=method)
    out_csv = results_dir / f"{tag}.csv"
    cmp_df.to_csv(out_csv)

    # Basic LaTeX export
    out_tex = results_dir / "table1_model_compare.tex"
    cmp_df.to_latex(out_tex, float_format="%.2f")
    return cmp_df, out_csv, out_tex


# -------------- OPTIONAL niceties --------------

def format_compare_for_paper(cmp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional: prettify az.compare output for manuscript.
    Renames columns and keeps common essentials.
    """
    rename = {
        "rank": "Rank",
        "loo": "ELPD",
        "p_loo": "p_LOO",
        "d_loo": "ΔELPD",
        "weight": "Weight",
        "se": "SE",
    }
    keep = [c for c in rename if c in cmp_df.columns]
    pretty = cmp_df[keep].rename(columns=rename).copy()
    # ΔELPD: best model should be 0.00
    if "ΔELPD" in pretty.columns:
        pretty.loc[pretty["Rank"] == 0, "ΔELPD"] = 0.0
    # Round a bit for neatness
    return pretty.round({"ELPD": 2, "ΔELPD": 2, "SE": 2, "p_LOO": 2, "Weight": 2})

def save_compare_table(pretty_df: pd.DataFrame, results_dir: str | Path, filename: str = "table1_model_compare.tex") -> Path:
    """Save the prettified compare table as LaTeX."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_tex = results_dir / filename
    pretty_df.to_latex(out_tex, index=True)
    return out_tex


# -------------- DIAGNOSTICS --------------

def find_influential_observations(
    idata: az.InferenceData,
    threshold: float = 0.7,
    save_csv: str | Path | None = None,
):
    """
    Return indices of observations with high Pareto-k in PSIS-LOO.

    Parameters
    ----------
    idata : az.InferenceData
        Must contain `log_likelihood` (you said it’s already computed).
    threshold : float
        Cutoff for flagging influential points (common: 0.5 warn, 0.7 high).
    save_csv : path-like or None
        If provided, writes a CSV with columns: obs_idx, pareto_k, influential.

    Returns
    -------
    influential_idx : np.ndarray
        1D array of observation indices where pareto_k > threshold.
    k_values : np.ndarray
        Pareto-k values (length = n_obs).
    loo_result : az.ELPDData
        Full ArviZ LOO result (with pointwise info).
    """
    loo_result = az.loo(idata, pointwise=True)
    # robust extraction across ArviZ versions
    k_da = getattr(loo_result, "pareto_k", None)
    k_values = k_da.values if hasattr(k_da, "values") else np.asarray(k_da)
    k_values = np.ravel(k_values)
    influential_idx = np.flatnonzero(k_values > threshold)

    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "obs_idx": np.arange(k_values.size, dtype=int),
            "pareto_k": k_values,
            "influential": k_values > threshold,
        }).to_csv(save_csv, index=False)

    return influential_idx, k_values, loo_result

