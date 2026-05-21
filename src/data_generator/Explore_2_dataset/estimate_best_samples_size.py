"""Optimal spatio-temporal window analysis for exponential distribution fitting.

For each (window_size, time_window) config, find which discharge threshold
(expressed as a class quantile) maximises the fraction of samples following Exp.

Run from: cd notebooks && python dev.py
"""
import importlib.util
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from joblib import Parallel, delayed
from pathlib import Path
from tqdm import tqdm

# ── Paths & settings ──────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[3]
NC_PATH     = "/home/mgallet/Documents/Dataset/RIVER_DISCHARGES/c7491e060d94c97212f0fe7ebcff57f0/data_version-5.nc"
CSV_PATH    = REPO_ROOT / "src/data_generator/Explore_2_dataset/stations_regimes_explore2.csv"
MODULE_PATH = REPO_ROOT / "src/data_generator/Explore_2_dataset/build_classification_dataset.py"
OUT_DIR     = Path(__file__).resolve().parent

CONFIGS          = [(5,4),(3,4),(4,6), (2,28), (1, 28), (2, 14)]
    # (5,2), (4,3), (4,4),(3,6), (2,8), (3, 3)]
#(2, 4), (3, 3), (3, 8), (4, 8)]  # (window_size, time_window)
N_PER_CLASS      = 2000         # stations per class (increase for production)
N_QUANTILES      = 9
Q_LEVELS         = np.linspace(0.1, 0.50, N_QUANTILES)
ALPHA_KS         = 0.05
EVALUATE_PASSED  = 0.6        # minimum mean pass rate to consider a threshold valid
PALETTE          = {"NG": "#d62728", "NP": "#ff7f0e", "PC": "#2ca02c",
                    "PM": "#1f77b4", "PN": "#9467bd", "NV": "#e377c2"}
REGIMES          = {
    "PM": "Pluvial moderately contrasted",
    "PC": "Pluvial contrasted",
    "PN": "Pluvio-nival",
    "NP": "Nivo-pluvial",
    "NV": "Nival",
    "NG": "Nivo-glacial",
}

# ── Load helpers ──────────────────────────────────────────────────────────────
def _load_bcd():
    spec = importlib.util.spec_from_file_location("bcd", MODULE_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_bcd               = _load_bcd()
extract_window     = _bcd.extract_window
find_nearest_pixel = _bcd.find_nearest_pixel
lambert93_to_wgs84 = _bcd.lambert93_to_wgs84


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    ds = xr.open_dataset(NC_PATH, engine="netcdf4")
    return ds["dis06"].values, ds["latitude"].values, ds["longitude"].values


def select_stations(n=N_PER_CLASS):
    df = pd.read_csv(CSV_PATH).dropna(subset=["X_Lambert93", "Y_Lambert93", "regime_code"])
    return (df.groupby("regime_code", group_keys=False)
              .apply(lambda g: g.head(n))
              .reset_index(drop=True))


def station_pixel(row, lat_c, lon_c):
    lon, lat = lambert93_to_wgs84(row["X_Lambert93"], row["Y_Lambert93"])
    return find_nearest_pixel(lat, lon, lat_c, lon_c)


# ── KS test ───────────────────────────────────────────────────────────────────
def extract_station_windows(row_dict, configs, data, lat_c, lon_c):
    """Pre-extract all spatial windows for one station (no GIL issue, fast slicing)."""
    li, loni = station_pixel(row_dict, lat_c, lon_c)
    return {(ws, tw): extract_window(data, li, loni, ws, tw) for ws, tw in configs}


def ks_from_windows(row_dict, windows_by_config):
    """KS tests with per-sample quantile threshold — safe for process-based parallelism."""
    regime = row_dict["regime_code"]

    records = []
    for (ws, tw), window in windows_by_config.items():
        for q_lv in Q_LEVELS:
            passed, tested, p_vals, n_vals = 0, 0, [], []
            if window is not None:
                for t in range(window.shape[0]):
                    x = window[t].ravel()
                    x = x[np.isfinite(x)]
                    if len(x) < 5:
                        continue
                    thr = np.quantile(x, q_lv)
                    x = x[x > thr] - thr
                    if len(x) < 5:
                        continue
                    n_vals.append(len(x))
                    _, p = stats.kstest(x, "expon", args=(0, np.mean(x)))
                    tested += 1
                    if p > ALPHA_KS:
                        passed += 1
                        p_vals.append(p)
            records.append({
                "code_station": row_dict["code_station"],
                "regime":  regime,
                "ws": ws, "tw": tw,
                "q_level":   round(float(q_lv), 4),
                "pass_pct":  passed / tested if tested > 0 else np.nan,
                "mean_p":    float(np.mean(p_vals)) if p_vals else np.nan,
                "mean_n":    float(np.mean(n_vals)) if n_vals else np.nan,
                "tested":    tested,
            })
    return records


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_config(df, ws, tw, out_path=None):
    """Per-regime semilogx boxplots: pass_pct (left) and KS p-value (right) vs q_level."""
    sub     = df[(df["ws"] == ws) & (df["tw"] == tw)]
    regimes = sorted(sub["regime"].unique())
    n_cls   = len(regimes)

    fig, axes = plt.subplots(n_cls, 2, figsize=(12, 3 * n_cls), sharex="col")
    if n_cls == 1:
        axes = np.array([axes])          # ensure 2-D indexing
    fig.suptitle(f"KS exponential fit — ws={ws}, tw={tw}  (n/group≈{ws**2*tw})", fontsize=12, y=1.01)

    for row_i, regime in enumerate(regimes):
        color  = PALETTE.get(regime, "gray")
        r_sub  = sub[sub["regime"] == regime]
        rname  = REGIMES.get(regime, regime)

        for col_i, (metric, scale, ylabel) in enumerate([
            ("pass_pct", 100, "Pass rate (%)"),
            ("mean_p",     1, "KS p-value"),
        ]):
            ax = axes[row_i, col_i]

            for q_lv, grp in r_sub.groupby("q_level"):
                vals = grp[metric].dropna().values * scale
                if len(vals) == 0:
                    continue
                bp = ax.boxplot(vals, positions=[q_lv], widths=[0.02],
                                patch_artist=True, manage_ticks=False,
                                flierprops=dict(marker=".", ms=3, alpha=0.5))
                bp["boxes"][0].set(facecolor=color, alpha=0.65)
                for part in ["whiskers", "caps", "medians"]:
                    for line in bp[part]:
                        line.set_color(color)

            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3, which="both")
            if col_i == 0:
                ax.set_title(f"{regime} — {rname}", fontsize=9, loc="left")
            if row_i == n_cls - 1:
                ax.set_xlabel("Quantile level")

    plt.tight_layout()
    path = out_path or OUT_DIR / f"ks_ws{ws}_tw{tw}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── Summary: minimum threshold per regime × config ────────────────────────────
def make_summary(df, evaluate_passed=EVALUATE_PASSED):
    """For each (regime, ws, tw), find the minimum q_level achieving mean_pass >= evaluate_passed."""
    agg = (df.groupby(["regime", "ws", "tw", "q_level"])
             .agg(mean_pass=("pass_pct", "mean"),
                  mean_p=("mean_p", "mean"),
                  mean_n=("mean_n", "mean"))
             .reset_index())
    rows = []
    for (regime, ws, tw), grp in agg.groupby(["regime", "ws", "tw"]):
        passing = grp[grp["mean_pass"] >= evaluate_passed].sort_values("q_level")
        if passing.empty:
            rows.append({"regime": regime, "ws": ws, "tw": tw,
                         "q_level": np.nan,
                         "mean_pass": np.nan, "mean_p": np.nan, "mean_n": np.nan})
        else:
            r = passing.iloc[0]
            rows.append({"regime": regime, "ws": ws, "tw": tw,
                         "q_level":   r["q_level"],
                         "mean_pass": r["mean_pass"],
                         "mean_p":    r["mean_p"],
                         "mean_n":    r["mean_n"]})
    return pd.DataFrame(rows).sort_values(["regime", "ws", "tw"]).reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading NetCDF data...")
    data, lat_c, lon_c = load_data()

    df_sel = select_stations()
    print(f"Selected {len(df_sel)} stations: {df_sel['regime_code'].value_counts().to_dict()}")

    print("\nExtracting spatial windows (sequential, fast)...")
    rows_list = df_sel.to_dict("records")
    station_windows = [
        extract_station_windows(row, CONFIGS, data, lat_c, lon_c)
        for row in tqdm(rows_list, desc="Windows", unit="station")
    ]

    print("Running KS tests (parallel, process-based)...")
    results = Parallel(n_jobs=-1)(
        delayed(ks_from_windows)(row, windows)
        for row, windows in tqdm(zip(rows_list, station_windows),
                                 total=len(rows_list), desc="KS tests", unit="station")
    )
    all_records = [rec for sublist in results for rec in sublist]

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_DIR / "ks_results.csv", index=False)
    print(f"Saved {len(df)} records to ks_results.csv")

    print("\nGenerating plots...")
    for ws, tw in CONFIGS:
        plot_config(df, ws, tw)

    print(f"\n── Summary: minimum q_level for mean_pass ≥ {EVALUATE_PASSED} ──────────────")
    df_summary = make_summary(df)
    df_summary["regime_name"] = df_summary["regime"].map(REGIMES)
    df_summary.to_csv(OUT_DIR / "ks_summary.csv", index=False)
    print(df_summary[["regime", "regime_name", "ws", "tw",
                       "q_level", "mean_pass", "mean_p", "mean_n"]].round(3).to_string(index=False))
    print(f"Saved ks_summary.csv")


if __name__ == "__main__":
    main()