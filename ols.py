import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── constants ─────────────────────────────────────────────────────────────────
SESSION_START  = "09:15"
MORNING_END    = "11:00"   # first 90 minutes
EXIT_TIME      = "14:30"
SESSION_MINS   = 375
TRADING_DAYS   = 252
TRAIN_FRAC     = 0.60

MORNING_MINS   = 105       # 09:15 to 11:00
REMAINING_MINS = 210       # 11:00 to 14:30

DEFAULT_DATASETS = [
    ("data/NIFTY_50_minute_data.csv",   "NIFTY"),
    ("data/NIFTY_BANK_minute_data.csv", "BANKNIFTY"),
]

EXPIRY_REGIMES = {
    "NIFTY": [
        {"start": "2019-02-11", "end": "2025-08-31", "weekday": 3, "freq": "W"},
        {"start": "2025-09-01", "end": "2099-01-01", "weekday": 1, "freq": "W"},
    ],
    "BANKNIFTY": [
        {"start": "2016-05-27", "end": "2023-09-03", "weekday": 3, "freq": "W"},
        {"start": "2023-09-04", "end": "2024-11-13", "weekday": 2, "freq": "W"},
        {"start": "2024-11-14", "end": "2025-08-31", "weekday": 3, "freq": "M"},
        {"start": "2025-09-01", "end": "2099-01-01", "weekday": 1, "freq": "M"},
    ],
}

plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                     "grid.alpha": 0.3, "font.size": 9})


# ── data loading ──────────────────────────────────────────────────────────────
def load_minute_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed")
    elif {"date", "time"}.issubset(df.columns):
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str))
        df = df.drop(columns=["date", "time"])
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], format="mixed")
        df = df.drop(columns=["date"])
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], format="mixed")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = (df["datetime"].dt
                          .tz_convert("Asia/Kolkata").dt.tz_localize(None))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= SESSION_START) & (df["time"] <= "15:30")].copy()
    df["logp"] = np.log(df["close"].astype(float))
    df["ret"]  = df.groupby("date")["logp"].diff()
    df["ret2"] = df["ret"] ** 2
    return df


def load_vix(path):
    if not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    v.columns = [c.strip().lower() for c in v.columns]
    date_col  = next((c for c in ["date","datetime","timestamp"]
                      if c in v.columns), None)
    close_col = next((c for c in ["close","vix","india_vix"]
                      if c in v.columns), None)
    if not date_col or not close_col:
        return None
    v["date"] = pd.to_datetime(v[date_col], format="mixed").dt.date
    v = (v[["date", close_col]]
         .rename(columns={close_col: "vix"})
         .drop_duplicates("date")
         .sort_values("date")
         .reset_index(drop=True))
    v["vix_lag1"] = v["vix"].shift(1)   # prior day close — no lookahead
    return v[["date", "vix_lag1"]].dropna()


def infer_expiry_days(df, symbol):
    key = "BANKNIFTY" if "BANK" in symbol.upper() else "NIFTY"
    trading_dates = set(df["date"].unique())
    lo, hi = min(trading_dates), max(trading_dates)
    expiries = set()
    for reg in EXPIRY_REGIMES[key]:
        start = max(pd.Timestamp(reg["start"]).date(), lo)
        end   = min(pd.Timestamp(reg["end"]).date(),   hi)
        if start > end:
            continue
        if reg["freq"] == "W":
            cands = [c.date() for c in pd.date_range(start, end, freq="D")
                     if c.weekday() == reg["weekday"]]
        else:
            cands = []
            for me in pd.date_range(start, end, freq="ME"):
                c = me
                while c.weekday() != reg["weekday"]:
                    c -= pd.Timedelta(days=1)
                if start <= c.date() <= end:
                    cands.append(c.date())
        for c in cands:
            d = c
            for _ in range(5):
                if d in trading_dates:
                    expiries.add(d); break
                d -= pd.Timedelta(days=1)
    return expiries


def drop_bad_days(df, min_bars=300):
    counts = df.groupby("date")["close"].count()
    return df[df["date"].isin(counts[counts >= min_bars].index)].copy()


# ── feature building ──────────────────────────────────────────────────────────
def build_dataset(expiry_df, vix_df):

    rows = []
    for date, grp in expiry_df.groupby("date"):
        grp = grp.sort_values("time")

        morning  = grp[(grp["time"] >= SESSION_START) &
                       (grp["time"] <= MORNING_END)]["ret2"].dropna()
        after    = grp[(grp["time"] >  MORNING_END) &
                       (grp["time"] <= EXIT_TIME)]["ret2"].dropna()

        if len(morning) < 50 or len(after) < 100:
            continue

        rv_m = float(morning.sum())
        rv_r = float(after.sum())
        if rv_m <= 0 or rv_r <= 0:
            continue

        vix_row = None
        if vix_df is not None:
            match = vix_df[vix_df["date"] == date]
            if len(match):
                vix_row = float(match["vix_lag1"].values[0])

        rows.append({
            "date":             date,
            "rv_morning":       rv_m,
            "log_rv_morning":   np.log(rv_m),
            "vix_prev":         vix_row,
            "log_vix_prev":     np.log(vix_row) if vix_row else np.nan,
            "rv_remaining":     rv_r,
            "log_rv_remaining": np.log(rv_r),
        })

    return pd.DataFrame(rows).set_index("date").sort_index()


# ── OLS fit ───────────────────────────────────────────────────────────────────
def fit_ols(X, y):
    b, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return b, y - yhat, r2


def oos_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── walk-forward OOS ──────────────────────────────────────────────────────────
def walk_forward_ols(df, use_vix):
    df = df.dropna(subset=["log_rv_morning", "log_rv_remaining"] +
                          (["log_vix_prev"] if use_vix else [])).copy()
    n      = len(df)
    n_init = max(int(n * TRAIN_FRAC), 30)

    feature_cols = ["log_rv_morning"] + (["log_vix_prev"] if use_vix else [])

    preds = []
    for i in range(n_init, n):
        train = df.iloc[:i]
        test  = df.iloc[[i]]

        X_tr = np.column_stack([np.ones(len(train))] +
                               [train[c].values for c in feature_cols])
        y_tr = train["log_rv_remaining"].values

        X_te = np.column_stack([np.ones(1)] +
                               [test[c].values for c in feature_cols])

        b, _, _ = fit_ols(X_tr, y_tr)
        pred    = float((X_te @ b).flat[0])

        preds.append({
            "date":             test.index[0],
            "log_rv_remaining": float(test["log_rv_remaining"].values[0]),
            "rv_remaining":     float(test["rv_remaining"].values[0]),
            "pred":             pred,
            "residual":         float(test["log_rv_remaining"].values[0]) - pred,
        })

    return pd.DataFrame(preds).set_index("date")


def fit_ols_full(df, use_vix):
    """Full-sample OLS. Returns (alpha, beta1, [beta2], r2_insample)."""
    df = df.dropna(subset=["log_rv_morning", "log_rv_remaining"] +
                          (["log_vix_prev"] if use_vix else []))
    feature_cols = ["log_rv_morning"] + (["log_vix_prev"] if use_vix else [])
    X = np.column_stack([np.ones(len(df))] +
                        [df[c].values for c in feature_cols])
    y = df["log_rv_remaining"].values
    b, resid, r2 = fit_ols(X, y)
    return b, resid, r2


def fit_t(residuals):
    """Fit scaled-t to residuals, cap nu at 50."""
    try:
        nu, loc, scale = stats.t.fit(residuals,
                                     loc=residuals.mean(),
                                     scale=residuals.std(ddof=1))
    except Exception:
        nu, loc, scale = stats.t.fit(residuals)
    nu = float(np.clip(nu, 2.1, 50.0))
    return float(nu), float(loc), float(scale)


# ── annualised vol helper ─────────────────────────────────────────────────────
def ann_vol(rv_sum, horizon_mins):
    return float(np.sqrt(rv_sum / horizon_mins * SESSION_MINS * TRADING_DAYS) * 100)


# ── strike quantile helper ────────────────────────────────────────────────────
def strike_quantiles(wf, nu, loc, scale, quantiles=(0.025, 0.05, 0.95, 0.975)):
    """
    Given OOS walk-forward predictions and t-fit parameters,
    compute the median conditional return quantile at each tail level.
    """
    out = {}
    for q in quantiles:
        t_q = stats.t.ppf(q, df=nu, loc=loc, scale=scale)
        shifted_log_rv = wf["pred"].values + t_q
        shifted_rv     = np.exp(shifted_log_rv)
        sign           = 1.0 if q >= 0.5 else -1.0
        ret_pct        = sign * np.sqrt(shifted_rv) * 100
        out[f"q{q}"] = float(np.median(ret_pct))
    return out


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_model(df, wf_2f, wf_1f, symbol, outdir):
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # 1. scatter: log RV morning vs log RV remaining
    ax = axes[0]
    ax.scatter(df["log_rv_morning"], df["log_rv_remaining"],
               s=12, alpha=0.5, label="all expiry days")
    if "log_vix_prev" in df.columns:
        # colour by VIX tercile
        vix = df["log_vix_prev"].dropna()
        q33, q67 = vix.quantile(0.33), vix.quantile(0.67)
        for tercile, (lo, hi, col, lbl) in enumerate([
            (-np.inf, q33, "tab:green",  "low VIX"),
            (q33,     q67, "tab:orange", "mid VIX"),
            (q67,  np.inf, "tab:red",    "high VIX"),
        ]):
            mask = (df["log_vix_prev"] >= lo) & (df["log_vix_prev"] < hi)
            sub  = df[mask]
            ax.scatter(sub["log_rv_morning"], sub["log_rv_remaining"],
                       s=14, alpha=0.6, color=col, label=lbl)
    ax.set_xlabel("log(RV morning 09:15–11:00)")
    ax.set_ylabel("log(RV remaining 11:00–14:30)")
    ax.set_title(f"{symbol}: log RV scatter coloured by prior-day VIX tercile")
    ax.legend(fontsize=8)

    # 2. OOS walk-forward: 2-feature vs 1-feature
    ax = axes[1]
    ax.plot(wf_2f.index, wf_2f["log_rv_remaining"], lw=0.9, label="actual")
    ax.plot(wf_2f.index, wf_2f["pred"],             lw=0.9, alpha=0.8,
            label=f"2-feature pred (R²={oos_r2(wf_2f['log_rv_remaining'].values, wf_2f['pred'].values):.3f})")
    ax.plot(wf_1f.index, wf_1f["pred"],             lw=0.9, alpha=0.6,
            ls="--",
            label=f"1-feature pred (R²={oos_r2(wf_1f['log_rv_remaining'].values, wf_1f['pred'].values):.3f})")
    ax.set_title(f"{symbol}: OOS walk-forward — 2-feature vs 1-feature model")
    ax.set_ylabel("log(RV remaining)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=7)

    # 3. residual histogram + t fit
    ax = axes[2]
    resid = wf_2f["residual"].dropna().values
    nu, loc, scale = fit_t(resid)
    ax.hist(resid, bins=35, density=True, alpha=0.65, label="OOS residuals")
    xs = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(xs, stats.t.pdf(xs, df=nu, loc=loc, scale=scale),
            lw=1.3, color="tab:red", label=f"t(ν={nu:.1f}, loc={loc:.3f}, scale={scale:.3f})")
    ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std(ddof=1)),
            lw=1.0, ls="--", color="tab:green", label="normal")
    ax.set_title(f"{symbol}: OOS residual distribution (2-feature model)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(outdir, f"two_feature_model_{symbol.lower()}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  plot saved → {path}")


# ── Pine Script block generator ───────────────────────────────────────────────
def pine_coefficients_block(results):
    """
    Generate a ready-to-paste Pine Script variable block for both indices.
    """
    lines = [
        "// TWO-FEATURE RV MODEL COEFFICIENTS",
        "// Model: log(RV_remaining) = α + β₁·log(RV_morning) + β₂·log(VIX_prev)",
        "// RV_morning  : sum of squared 1-min log-returns, 09:15 → 11:00",
        "// VIX_prev    : prior trading day's India VIX closing value",
        "// RV_remaining: predicted sum of squared 1-min log-returns, 11:00 → 14:30",
        "// Fitted on NSE expiry-day minute data.  Walk-forward OOS validation.",
        "",
    ]
    for sym, r in results.items():
        b2f  = r["coef_2feature"]
        b1f  = r["coef_1feature"]
        nu   = r["t_nu"]
        loc_ = r["t_loc"]
        sc   = r["t_scale"]
        r2   = r["oos_r2_2feature"]
        r2_1 = r["oos_r2_1feature"]
        n    = r["n_days"]
        q    = r["strike_quantiles"]

        lines += [
            f"// ── {sym} ──────────────────────────────────────────────────",
            f"// Training data: {n} expiry days",
            f"// OOS R²  2-feature: {r2:.4f}   |   1-feature (RV only): {r2_1:.4f}",
            f"// Residual t-distribution: ν={nu:.2f}  loc={loc_:.4f}  scale={sc:.4f}",
            f"// Median conditional strike quantiles (11:00 anchor):",
            f"//   PUT  2.5% : {q.get('q0.025', float('nan')):+.4f}%  of spot",
            f"//   PUT  5.0% : {q.get('q0.05',  float('nan')):+.4f}%  of spot",
            f"//   CALL 95.0%: {q.get('q0.95',  float('nan')):+.4f}%  of spot",
            f"//   CALL 97.5%: {q.get('q0.975', float('nan')):+.4f}%  of spot",
            f"",
            f"// 2-feature model (use this)",
            f"float {sym.lower()}_alpha  = {b2f[0]:.6f}  // intercept",
            f"float {sym.lower()}_beta1  = {b2f[1]:.6f}  // log(RV_morning)",
            f"float {sym.lower()}_beta2  = {b2f[2]:.6f}  // log(VIX_prev)",
            f"",
            f"// Residual t-distribution parameters",
            f"float {sym.lower()}_t_nu    = {nu:.4f}",
            f"float {sym.lower()}_t_loc   = {loc_:.6f}",
            f"float {sym.lower()}_t_scale = {sc:.6f}",
            f"",
        ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    outdir = "two_feature_model"
    os.makedirs(outdir, exist_ok=True)

    vix_df = load_vix("data/INDIA_VIX_minute.csv")
    results = {}
    summary_rows = []

    for data_path, symbol in DEFAULT_DATASETS:
        if not os.path.exists(data_path):
            print(f"[skip] {data_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        df = load_minute_data(data_path)
        df = drop_bad_days(df)
        expiries = infer_expiry_days(df, symbol)
        edf = df[df["date"].isin(expiries)].copy()

        dataset = build_dataset(edf, vix_df)
        has_vix = (vix_df is not None and
                   dataset["log_vix_prev"].notna().sum() > 30)

        n_total = len(dataset)
        n_vix   = int(dataset["log_vix_prev"].notna().sum()) if has_vix else 0
        print(f"  expiry days: {n_total}  |  with VIX: {n_vix}")

        # ── full-sample OLS ──
        b2f, resid_full, r2_full = fit_ols_full(dataset, use_vix=has_vix)
        b1f, _,           r2_1f  = fit_ols_full(dataset, use_vix=False)

        print(f"\n  Full-sample OLS coefficients (2-feature):")
        print(f"    α  (intercept)    = {b2f[0]:+.6f}")
        print(f"    β₁ (log RV_morn)  = {b2f[1]:+.6f}")
        if has_vix:
            print(f"    β₂ (log VIX_prev) = {b2f[2]:+.6f}")
        print(f"    In-sample R²       = {r2_full:.4f}")
        print(f"\n  Full-sample OLS coefficients (1-feature, RV only):")
        print(f"    α  (intercept)    = {b1f[0]:+.6f}")
        print(f"    β₁ (log RV_morn)  = {b1f[1]:+.6f}")
        print(f"    In-sample R²       = {r2_1f:.4f}")

        # ── walk-forward OOS ──
        wf_2f = walk_forward_ols(dataset, use_vix=has_vix)
        wf_1f = walk_forward_ols(dataset, use_vix=False)

        r2_oos_2f  = oos_r2(wf_2f["log_rv_remaining"].values, wf_2f["pred"].values)
        r2_oos_1f  = oos_r2(wf_1f["log_rv_remaining"].values, wf_1f["pred"].values)
        rmse_2f    = rmse(wf_2f["log_rv_remaining"].values,    wf_2f["pred"].values)
        rmse_1f    = rmse(wf_1f["log_rv_remaining"].values,    wf_1f["pred"].values)
        vix_lift   = r2_oos_2f - r2_oos_1f

        print(f"\n  Walk-forward OOS (expanding window, {int(TRAIN_FRAC*100)}% initial train):")
        print(f"    2-feature  OOS R² = {r2_oos_2f:.4f}   RMSE = {rmse_2f:.4f}")
        print(f"    1-feature  OOS R² = {r2_oos_1f:.4f}   RMSE = {rmse_1f:.4f}")
        print(f"    VIX lift in OOS R² = {vix_lift:+.4f}  "
              f"({'meaningful' if vix_lift > 0.02 else 'marginal'})")

        # ── residual t-fit on 2-feature OOS residuals ──
        resid_oos = wf_2f["residual"].dropna().values
        nu, loc_, scale_ = fit_t(resid_oos)
        ks_stat, ks_p = stats.kstest(resid_oos, "t",
                                     args=(nu, loc_, scale_))
        print(f"\n  Residual t-fit (OOS):  ν={nu:.2f}  loc={loc_:.4f}  scale={scale_:.4f}")
        print(f"  KS test p={ks_p:.3f}  "
              f"({'good fit' if ks_p > 0.05 else 'some misfit — tails may be heavier'})")

        # ── bias stats ──
        bias_mean = resid_oos.mean()
        bias_med  = np.median(resid_oos)
        print(f"\n  Residual bias: mean={bias_mean:+.4f}  median={bias_med:+.4f}")
        if bias_mean < -0.05:
            print(f"  → Model overpredicts RV by ~{(1-np.exp(bias_mean))*100:.1f}% on average")
            print(f"    (VRP visible: realized vol < forecast vol on typical expiry day)")
        elif bias_mean > 0.05:
            print(f"  → Model underpredicts RV by ~{(np.exp(bias_mean)-1)*100:.1f}% on average")

        # ── median remaining vol ──
        med_rv_rem = float(np.median(dataset["rv_remaining"]))
        med_vol    = ann_vol(med_rv_rem, REMAINING_MINS)
        print(f"\n  Median remaining annualized vol (11:00→14:30): {med_vol:.2f}%")

        # ── conditional strike quantiles ──
        q_strikes = strike_quantiles(wf_2f, nu, loc_, scale_)
        print(f"\n  Median conditional strike quantiles (% of spot):")
        for qk, qv in q_strikes.items():
            side = "PUT " if qv < 0 else "CALL"
            print(f"    {side} {qk}: {qv:+.4f}%")

        # ── store ──
        results[symbol] = {
            "n_days":          n_total,
            "coef_2feature":   b2f,
            "coef_1feature":   b1f,
            "r2_insample":     r2_full,
            "oos_r2_2feature": r2_oos_2f,
            "oos_r2_1feature": r2_oos_1f,
            "oos_rmse":        rmse_2f,
            "vix_lift":        vix_lift,
            "t_nu":            nu,
            "t_loc":           loc_,
            "t_scale":         scale_,
            "strike_quantiles": q_strikes,
            "med_vol":         med_vol,
        }

        summary_rows.append({
            "symbol":          symbol,
            "n_days":          n_total,
            "alpha":           b2f[0],
            "beta1_log_rv":    b2f[1],
            "beta2_log_vix":   b2f[2] if has_vix else np.nan,
            "r2_insample":     r2_full,
            "oos_r2_2feature": r2_oos_2f,
            "oos_r2_1feature": r2_oos_1f,
            "vix_lift":        vix_lift,
            "t_nu":            nu,
            "t_loc":           loc_,
            "t_scale":         scale_,
        })

        plot_model(dataset, wf_2f, wf_1f, symbol, outdir)

    # ── summary table ──
    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        path = os.path.join(outdir, "model_coefficients.csv")
        summary.to_csv(path, index=False)
        print(f"\n\n{'='*60}")
        print("  SUMMARY TABLE")
        print(f"{'='*60}")
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda v: f"{v:.4f}")
        print(summary.to_string(index=False))
        print(f"\n  coefficients CSV → {path}")

    # Pine script for personal tradingview integration
    
    # if results:
    #     pine_block = pine_coefficients_block(results)
    #     pine_path  = os.path.join(outdir, "pine_coefficients.txt")
    #     with open(pine_path, "w") as f:
    #         f.write(pine_block)
    #     print(f"\n{'='*60}")
    #     print("  PINE SCRIPT COEFFICIENT BLOCK")
    #     print(f"{'='*60}")
    #     print(pine_block)
    #     print(f"  → also saved to {pine_path}")

if __name__ == "__main__":
    main()