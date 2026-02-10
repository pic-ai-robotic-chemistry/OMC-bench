import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.integrate import simpson
from scipy.stats import entropy
from scipy.interpolate import interp1d


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

    Parameters
    ----------
    ref, pred : DataFrame
        Reference and predicted data.
    col : str
        The column name to evaluate.

    Returns
    -------
    mae, rmse, r2 : floats
    """
    y_true = ref[col]
    y_pred = pred[col]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


# --------------------------------------------------------------
# DOS metrics (MAE, RMSE, KL divergence)
# --------------------------------------------------------------

def calc_dos_metrics(ref, pred, n_points=1000):
    """
    Compute DOS comparison metrics using interpolation on a shared frequency grid.
    Metrics:
        - MAE (normalized DOS)
        - RMSE (normalized DOS)
        - KL divergence

    Parameters
    ----------
    n_points : int
        Number of points in the interpolated frequency grid.

    Returns
    -------
    avg_mae, avg_rmse, avg_kl : floats
        Mean DOS metrics across all materials.
    """
    maes, rmses, kls = [], [], []

    for m in ref.index:
        if m not in pred.index:
            continue

        freq_ref = np.array(ref.loc[m, "DOS_freq"])
        dos_ref = np.array(ref.loc[m, "DOS"])

        freq_pred = np.array(pred.loc[m, "DOS_freq"])
        dos_pred = np.array(pred.loc[m, "DOS"])

        # Determine overlapping frequency domain
        start = max(freq_ref[0], freq_pred[0])
        end = min(freq_ref[-1], freq_pred[-1])
        if end - start < 1e-5:
            continue

        freq_common = np.linspace(start, end, n_points)

        # Linear interpolation (use 'cubic' if higher smoothness is desired)
        interp_ref = interp1d(freq_ref, dos_ref, kind="linear",
                              fill_value=0, bounds_error=False)
        interp_pred = interp1d(freq_pred, dos_pred, kind="linear",
                               fill_value=0, bounds_error=False)

        dos_ref_interp = interp_ref(freq_common)
        dos_pred_interp = interp_pred(freq_common)

        # Normalize DOS using Simpson integration
        area_ref = simpson(dos_ref_interp, x=freq_common)
        area_pred = simpson(dos_pred_interp, x=freq_common)

        if area_ref == 0 or area_pred == 0:
            continue

        dos_ref_norm = dos_ref_interp / area_ref
        dos_pred_norm = dos_pred_interp / area_pred

        # MAE and RMSE
        maes.append(mean_absolute_error(dos_ref_norm, dos_pred_norm))
        rmses.append(np.sqrt(mean_squared_error(dos_ref_norm, dos_pred_norm)))

        # KL divergence (add epsilon for stability)
        epsilon = 1e-10
        p = dos_ref_norm + epsilon
        q = dos_pred_norm + epsilon

        p /= p.sum()
        q /= q.sum()

        kl = entropy(p, q)
        kls.append(kl)

    if maes:
        return np.mean(maes), np.mean(rmses), np.mean(kls)
    else:
        return np.nan, np.nan, np.nan


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
            "KL": np.nan   # KL applies only to DOS
        })

    # Evaluate DOS metrics
    dos_mae, dos_rmse, dos_kl = calc_dos_metrics(ref, pred)
    print(f"{'DOS(normed)':30s}: MAE={dos_mae:.4f}, RMSE={dos_rmse:.4f}, KL={dos_kl:.4f}")

    results.append({
        "Metric": "DOS(normed)",
        "MAE": dos_mae,
        "RMSE": dos_rmse,
        "R2": np.nan,
        "KL": dos_kl
    })

    # Save summary
    out_csv = "results/task_4/metrics_summary.csv"
    df_metrics = pd.DataFrame(results)
    df_metrics.to_csv(out_csv, index=False)

    print(f"\nAll metrics have been saved to: {out_csv}")


# --------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python compare_metrics_phonon.py <ref_csv> <pred_csv>")
        exit(1)

    main(sys.argv[1], sys.argv[2])
