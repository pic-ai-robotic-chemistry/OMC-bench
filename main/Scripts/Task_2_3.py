import json
import os
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import simpson
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------
# Penalty values (used to fill None/NaN metrics)
# --------------------------------------------------------------

PENALTY_VALUES = {
    "omega_max": 20.0,  # THz
    "DOS_WD": 50.0,  # THz (Wasserstein Distance)
    "S_MAE": 200.0,  # J/mol/K
    "A_MAE": 100.0,  # kJ/mol (unit consistency is user-defined)
    "Cv_MAE": 100.0,  # J/mol/K
}


# --------------------------------------------------------------
# Utility: normalize structure names
# --------------------------------------------------------------

def normalize_name(s):
    """
    Normalize material names by removing suffixes such as '_opt'.
    Example:
        'OXAMID04_opt' -> 'OXAMID04'
    """
    return s.split('_')[0]


def load_csv(path):
    """
    Load phonon summary CSV and parse list‐type columns (DOS and DOS frequencies).

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    DataFrame indexed by Material name.
    """
    print(path)
    df = pd.read_csv(path, converters={"DOS_freq": literal_eval, "DOS": literal_eval})
    df["Material"] = df["Material"].apply(normalize_name)
    return df.set_index("Material")


# --------------------------------------------------------------
# Scalar property metrics (MAE, RMSE, R²)
# --------------------------------------------------------------

def calc_scalar_metrics(ref, pred, col):
    """
    Compute MAE, RMSE, and R² for a scalar phonon/thermo property.
    MAE/RMSE are computed after applying per-structure penalties.
    R² is computed only on valid (finite) pairs without penalties.
    """
    y_true = ref[col].copy()
    y_pred = pred[col].copy()

    # Normalize thermodynamic quantities by atom count Z (if available)
    if col in {"Entropy_300K(J/mol/K)", "Free_energy_300K(meV)", "Cv_300K(J/mol/K)"}:
        z_col = None
        for candidate in ("Z", "n_mol"):
            if candidate in ref.columns:
                z_col = candidate
                break
        if z_col:
            z_vals = ref[z_col].replace(0, np.nan)
            y_true = y_true / z_vals
            y_pred = y_pred / z_vals

    # Raw per-structure absolute error
    raw_err = (y_true - y_pred).abs()

    # Penalty mapping
    if col == "w_max(THz)":
        penalty = PENALTY_VALUES["omega_max"]
    elif col == "Entropy_300K(J/mol/K)":
        penalty = PENALTY_VALUES["S_MAE"]
    elif col == "Free_energy_300K(meV)":
        penalty = PENALTY_VALUES["A_MAE"]
    elif col == "Cv_300K(J/mol/K)":
        penalty = PENALTY_VALUES["Cv_MAE"]
    else:
        penalty = None

    if penalty is not None:
        raw_err = raw_err.fillna(penalty)

    mae = raw_err.mean()
    rmse = np.sqrt((raw_err ** 2).mean())

    # R² uses only valid pairs (no penalties)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.any():
        r2 = r2_score(y_true[mask], y_pred[mask])
    else:
        r2 = np.nan

    return mae, rmse, r2


# --------------------------------------------------------------
# DOS metrics (MAE, RMSE, Wasserstein distance)
# --------------------------------------------------------------

def calc_dos_metrics(ref, pred, n_points=1000, sigma=10):
    """
    Compute DOS comparison metrics using interpolation on a shared frequency grid.
    Metrics:
        - MAE (normalized DOS)
        - RMSE (normalized DOS)
        - Wasserstein distance

    Parameters
    ----------
    n_points : int
        Number of points in the interpolated frequency grid.

    Returns
    -------
    avg_mae, avg_rmse, avg_wd : floats
        Mean DOS metrics across all materials.
    """
    dos_mae = {}
    dos_rmse = {}
    dos_wd = {}

    for m in ref.index:
        if m not in pred.index:
            continue

        try:
            freq_ref = np.asarray(ref.loc[m, "DOS_freq"])
            dos_ref = np.asarray(ref.loc[m, "DOS"])

            freq_pred = np.asarray(pred.loc[m, "DOS_freq"])
            dos_pred = np.asarray(pred.loc[m, "DOS"])

            if freq_ref.ndim == 0 or freq_pred.ndim == 0:
                continue

            # Filter invalid values
            mask_ref = np.isfinite(freq_ref) & (freq_ref >= 1e-2)
            freq_ref, dos_ref = freq_ref[mask_ref], dos_ref[mask_ref]

            mask_pred = np.isfinite(freq_pred) & (freq_pred >= 1e-2)
            freq_pred, dos_pred = freq_pred[mask_pred], dos_pred[mask_pred]

            if len(freq_ref) < 2 or len(freq_pred) < 2:
                continue

            # Determine unified frequency domain
            start = min(freq_ref[0], freq_pred[0])
            end = max(freq_ref[-1], freq_pred[-1])
            end = min(end, 4000)
            if end - start < 1e-5:
                continue

            freq_common = np.linspace(start, end, n_points)

            interp_ref = interp1d(freq_ref, dos_ref, kind="linear",
                                  fill_value=0, bounds_error=False)
            interp_pred = interp1d(freq_pred, dos_pred, kind="linear",
                                   fill_value=0, bounds_error=False)

            y_ref = np.maximum(interp_ref(freq_common), 0.0)
            y_pred = np.maximum(interp_pred(freq_common), 0.0)

            if sigma and sigma > 0:
                y_ref = gaussian_filter1d(y_ref, sigma)
                y_pred = gaussian_filter1d(y_pred, sigma)

            area_ref = simpson(y_ref, x=freq_common)
            area_pred = simpson(y_pred, x=freq_common)

            if area_ref < 1e-6 or area_pred < 1e-6:
                continue

            y_ref_norm = y_ref / area_ref
            y_pred_norm = y_pred / area_pred

            dos_mae[m] = mean_absolute_error(y_ref_norm, y_pred_norm)
            dos_rmse[m] = np.sqrt(mean_squared_error(y_ref_norm, y_pred_norm))
            dos_wd[m] = wasserstein_distance(freq_common, freq_common, y_ref_norm, y_pred_norm)

        except Exception:
            continue

    return pd.Series(dos_mae), pd.Series(dos_rmse), pd.Series(dos_wd)


# --------------------------------------------------------------
# Main evaluation pipeline
# --------------------------------------------------------------

def main(ref_csv, pred_csv):
    # Load data
    ref = load_csv(ref_csv)
    pred = load_csv(pred_csv)

    # Compare only overlapping materials
    common = ref.index.intersection(pred.index)
    ref, pred = ref.loc[common], pred.loc[common]

    print(f"Number of comparable materials: {len(common)}\n")

    # Optional: invalidate rows based on JSON steps
    json_dir = getattr(main, "json_dir", None)
    if json_dir and os.path.isdir(json_dir):
        invalid_count = 0
        for mat_name in ref.index:
            json_path = os.path.join(json_dir, f"{mat_name}.json")
            should_drop = False
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        content = json.load(f)
                    steps = content.get("steps", 3000)
                    if steps is None:
                        steps = 3000
                    if steps >= 3000:
                        should_drop = True
                except Exception:
                    should_drop = True
            else:
                should_drop = True

            if should_drop:
                ref.loc[mat_name] = np.nan
                pred.loc[mat_name] = np.nan
                invalid_count += 1

        print(f"Invalidated rows (steps>=3000 or missing JSON): {invalid_count}")

    # Scalar properties to evaluate
    columns = [
        "w_max(THz)",
        "Entropy_300K(J/mol/K)",
        "Free_energy_300K(meV)",
        "Cv_300K(J/mol/K)"
    ]

    results = []

    # Evaluate scalar metrics
    for col in columns:
        mae, rmse, r2 = calc_scalar_metrics(ref, pred, col)
        print(f"{col:30s}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

        results.append({
            "Metric": col,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Wasserstein": np.nan   # Wasserstein applies only to DOS
        })

    # Evaluate DOS metrics
    dos_mae_s, dos_rmse_s, dos_wd_s = calc_dos_metrics(ref, pred)

    dos_mae = dos_mae_s.mean() if not dos_mae_s.empty else np.nan
    dos_rmse = dos_rmse_s.mean() if not dos_rmse_s.empty else np.nan

    if dos_wd_s.empty:
        dos_wd = np.nan
    else:
        dos_wd = dos_wd_s.reindex(ref.index).fillna(PENALTY_VALUES["DOS_WD"]).mean()
    print(f"{'DOS(normed)':30s}: MAE={dos_mae:.4f}, RMSE={dos_rmse:.4f}, Wasserstein={dos_wd:.4f}")

    results.append({
        "Metric": "DOS(normed)",
        "MAE": dos_mae,
        "RMSE": dos_rmse,
        "R2": np.nan,
        "Wasserstein": dos_wd
    })

    # Save summary
    out_csv = "results/task_2/metrics_summary.csv"
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(out_csv, index=False)

    print(f"\nAll metrics have been saved to: {out_csv}")


# --------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) not in (3, 4):
        print("Usage: python Task_2_3.py <ref_csv> <pred_csv> [json_dir]")
        exit(1)

    main.json_dir = sys.argv[3] if len(sys.argv) == 4 else "results/task_2/individual_results"
    main(sys.argv[1], sys.argv[2])
