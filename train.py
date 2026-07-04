import argparse
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
from catboost import CatBoostRegressor, Pool


SESSION_START   = "09:15"
EXIT_TIME       = "14:30"                                                               # position square off at 14:30 to avoid last hour volatility
SESSION_MINUTES = 375
TRADING_DAYS    = 252

ANCHORS = ["09:30", "10:00", "10:30", "11:00", "11:30", "12:00"]

QUANTILES = [0.005, 0.01, 0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.99, 0.995]


TRAIN_FRAC = 0.60

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


# ── utilities ────────────────────────────────────────────────────────────────
def minutes_between(t1: str, t2: str) -> int:
    h1, m1 = map(int, t1.split(":"))
    h2, m2 = map(int, t2.split(":"))
    return (h2 * 60 + m2) - (h1 * 60 + m1)


def ann_vol_pct(rv_sum, horizon_minutes):
    h = max(float(horizon_minutes), 1e-9)
    return float(np.sqrt(rv_sum / h * SESSION_MINUTES * TRADING_DAYS) * 100)


# ── data loading ─────────────────────────────────────────────
def load_minute_data(path: str) -> pd.DataFrame:
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
    else:
        raise ValueError("Need a datetime/date/timestamp column.")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = (df["datetime"].dt.tz_convert("Asia/Kolkata")
                          .dt.tz_localize(None))
    df = df.sort_values("datetime").reset_index(drop=True)
    df["date"] = df["datetime"].dt.date
    df["time"] = df["datetime"].dt.strftime("%H:%M")
    df = df[(df["time"] >= SESSION_START) & (df["time"] <= "15:30")].copy()
    df["logp"] = np.log(df["close"].astype(float))
    df["ret"]  = df.groupby("date")["logp"].diff()
    df["ret2"] = df["ret"] ** 2
    return df


def load_vix(path: str | None) -> pd.DataFrame | None:

    if path is None or not os.path.exists(path):
        return None
    v = pd.read_csv(path)
    v.columns = [c.strip().lower() for c in v.columns]
    date_col  = next((c for c in ["date","datetime","timestamp"] if c in v.columns), None)
    if date_col is None:
        return None
    v["date"] = pd.to_datetime(v[date_col], format="mixed").dt.date
    close_col = next((c for c in ["close","vix","india_vix"] if c in v.columns), None)
    if close_col is None:
        return None
    v = v[["date", close_col]].rename(columns={close_col: "vix"}).drop_duplicates("date")
    v = v.sort_values("date").reset_index(drop=True)
    # shift: row i's vix value becomes available on row i+1's date
    v["india_vix_lag1"] = v["vix"].shift(1)
    return v[["date", "india_vix_lag1"]].dropna()


def infer_expiry_days(df: pd.DataFrame, symbol: str) -> set:
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
                    expiries.add(d)
                    break
                d -= pd.Timedelta(days=1)
    return expiries


def drop_bad_days(df: pd.DataFrame, min_bars: int = 300) -> pd.DataFrame:
    counts = df.groupby("date")["close"].count()
    good = counts[counts >= min_bars].index
    return df[df["date"].isin(good)].copy()


# ── prior-expiry RV ───────────────────────────────────────────────────────────
def build_prior_expiry_rv(df: pd.DataFrame, expiry_dates_sorted: list) -> dict:
    full_rv = {}
    for d in expiry_dates_sorted:
        sub = df[df["date"] == d]["ret2"].dropna()
        full_rv[d] = float(sub.sum())
    prior = {}
    for i, d in enumerate(expiry_dates_sorted):
        prior[d] = full_rv[expiry_dates_sorted[i - 1]] if i > 0 else np.nan
    return prior


# ── feature engineering ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame,
                   anchor: str,
                   prior_rv: dict,
                   vix_df: pd.DataFrame | None) -> pd.DataFrame:
    
    anchor_mins    = minutes_between(SESSION_START, anchor)
    remaining_mins = minutes_between(anchor, EXIT_TIME)

    rows = []
    for date, grp in df.groupby("date"):
        grp = grp.sort_values("time")
        before = grp[grp["time"] <= anchor]
        after  = grp[(grp["time"] > anchor) & (grp["time"] <= EXIT_TIME)]

        if len(before) < 5 or len(after) < 3:
            continue

        rets2_before = before["ret2"].dropna().values
        rets_before  = before["ret"].dropna().values
        prices_before = before["close"].values
        times_before  = before["time"].values

        # ── target ──
        rv_rem = float(after["ret2"].dropna().sum())
        if rv_rem <= 0:
            continue

        # ── RV in fixed backward windows ──
        def rv_last_n(n_min):
            # last n_min rows that fall within anchor window
            sub = before[before["time"] >= _subtract_min(anchor, n_min)]
            return float(sub["ret2"].dropna().sum())

        rv_15  = rv_last_n(15)
        rv_30  = rv_last_n(30)
        rv_75  = rv_last_n(75)
        rv_all = float(rets2_before.sum())

        # ── RV acceleration: last-15 vs prior-15 ──
        rv_15_lag = rv_last_n(30) - rv_last_n(15)        # 15–30 min ago
        rv_accel  = (rv_15 - rv_15_lag) / (rv_15_lag + 1e-12)

        # ── first-minute return (overnight shock) ──
        first_row = grp[grp["time"] == grp["time"].min()]
        open_ret  = float(first_row["ret"].values[0]) if len(first_row) else 0.0

        # ── cumulative return open → anchor ──
        if len(prices_before) >= 2:
            cum_ret = float(np.log(prices_before[-1] / prices_before[0]))
        else:
            cum_ret = 0.0

        # ── max absolute 1-min return (jump detector) ──
        max_abs_ret = float(np.abs(rets_before).max()) if len(rets_before) else 0.0

        # ── time features ──
        time_remaining   = float(remaining_mins)
        hour_frac        = float(anchor_mins) / SESSION_MINUTES

        # ── prior expiry full-session RV ──
        prior_full_rv = prior_rv.get(date, np.nan)

        # ── VIX (prior-day close — no lookahead) ──
        vix_val = np.nan
        if vix_df is not None:
            row = vix_df[vix_df["date"] == date]
            if len(row):
                vix_val = float(row["india_vix_lag1"].values[0])

        rows.append({
            "date":           date,
            # HAR-style RV windows
            "rv_15min":       rv_15,
            "rv_30min":       rv_30,
            "rv_75min":       rv_75,
            "rv_morning":     rv_all,
            # momentum / vol state
            "rv_accel":       rv_accel,
            "cum_ret":        cum_ret,
            "open_ret":       open_ret,
            "max_abs_ret":    max_abs_ret,
            # time
            "time_remaining": time_remaining,
            "hour_frac":      hour_frac,
            # cross-expiry memory
            "prior_expiry_rv": prior_full_rv,
            # optional (prior-day close, no lookahead)
            "india_vix_lag1": vix_val,
            # targets
            "rv_rem":         rv_rem,
            "log_rv_rem":     np.log(rv_rem),
        })

    return pd.DataFrame(rows).set_index("date")


def _subtract_min(time_str: str, n: int) -> str:
    h, m = map(int, time_str.split(":"))
    total = h * 60 + m - n
    total = max(total, 0)
    return f"{total // 60:02d}:{total % 60:02d}"


FEATURE_COLS = [
    "rv_15min", "rv_30min", "rv_75min", "rv_morning",
    "rv_accel", "cum_ret", "open_ret", "max_abs_ret",
    "time_remaining", "hour_frac",
    "prior_expiry_rv", "india_vix_lag1",
]


# ── CatBoost walk-forward ─────────────────────────────────────────────────────
def catboost_params():
    return dict(
        iterations        = 500,
        learning_rate     = 0.05,
        depth             = 5,
        l2_leaf_reg       = 3.0,
        min_data_in_leaf  = 5,
        loss_function     = "RMSE",
        eval_metric       = "RMSE",
        random_seed       = 42,
        verbose           = False,
        allow_writing_files = False,
    )


def walk_forward(feat_df: pd.DataFrame,
                 anchor: str) -> pd.DataFrame:

    df = feat_df.copy().sort_index()
    n  = len(df)
    n_init = max(int(n * TRAIN_FRAC), 30)

    if n < n_init + 5:
        print(f"  [{anchor}] insufficient data ({n} days) for walk-forward")
        return pd.DataFrame()

    # drop VIX feature if all-NaN
    feats = [c for c in FEATURE_COLS
             if c in df.columns and df[c].notna().any()]

    preds = []
    for i in range(n_init, n):
        train = df.iloc[:i]
        test  = df.iloc[[i]]

        X_tr = train[feats].fillna(train[feats].median())
        y_tr = train["log_rv_rem"].values

        X_te = test[feats].fillna(train[feats].median())

        model = CatBoostRegressor(**catboost_params())
        model.fit(X_tr, y_tr, verbose=False)
        pred = float(model.predict(X_te)[0])

        preds.append({
            "date":          test.index[0],
            "log_rv_rem":    float(test["log_rv_rem"].values[0]),
            "pred_log_rv":   pred,
            "rv_rem":        float(test["rv_rem"].values[0]),
            "pred_rv":       float(np.exp(pred)),
        })

    out = pd.DataFrame(preds).set_index("date")
    out["residual"] = out["log_rv_rem"] - out["pred_log_rv"]
    return out


def fit_full_model(feat_df: pd.DataFrame,
                   anchor: str,
                   outdir: str) -> CatBoostRegressor:
    """Fit on ALL expiry days and save model."""
    feats = [c for c in FEATURE_COLS
             if c in feat_df.columns and feat_df[c].notna().any()]
    X = feat_df[feats].fillna(feat_df[feats].median())
    y = feat_df["log_rv_rem"].values
    model = CatBoostRegressor(**catboost_params())
    model.fit(X, y, verbose=False)
    model.save_model(os.path.join(outdir, f"model_{anchor.replace(':','')}.cbm"))
    return model, feats


# ── evaluation metrics ────────────────────────────────────────────────────────
def oos_r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


# ── Phase 3: scaled-t tail fit ────────────────────────────────────────────────
NU_MAX = 50.0  # cap on t df: avoids nu->inf (Gaussian) on quiet samples,
               # which underestimates tail risk on unscheduled-event days.

def fit_scaled_t(residuals: np.ndarray) -> tuple[float, float, float]:
    """
    MLE fit of a scaled (location-scale) t-distribution to OOS residuals.
    Returns (df, loc, scale). nu is clamped to [2.1, NU_MAX].
    """
    try:
        nu, loc, scale = stats.t.fit(residuals,
                                     loc=residuals.mean(),
                                     scale=residuals.std(ddof=1))
    except Exception:
        nu, loc, scale = stats.t.fit(residuals)

    nu = float(np.clip(nu, 2.1, NU_MAX))

    ks_stat, ks_p = stats.kstest(residuals, "t", args=(nu, loc, scale))
    if ks_p < 0.05:
        print(f"     [warn] t-fit KS p={ks_p:.3f} -- residuals may deviate from t")

    return float(nu), float(loc), float(scale)


def conditional_quantiles(wf: pd.DataFrame,
                          feat_df: pd.DataFrame,
                          anchor: str,
                          nu: float, loc: float, scale: float) -> dict:

    remaining_mins = minutes_between(anchor, EXIT_TIME)

    rows = []
    for date, row in wf.iterrows():
        pred_rv   = row["pred_rv"]
        pred_vol  = np.sqrt(pred_rv / remaining_mins)  # per-minute vol

        for q in QUANTILES:
            # residual quantile in log-RV space then back-transform
            # to return space: ret ~ N(0, pred_vol) times residual scaling
            # We model R_remaining | F_t ~ pred_vol * sqrt(remaining_mins) * t_epsilon
            # where epsilon ~ t(nu, loc, scale) on the standardised return
            t_q    = stats.t.ppf(q, df=nu, loc=loc, scale=scale)
            # log RV quantile → RV → annualised vol → de-annualise to raw return
            pred_log_rv_q = row["pred_log_rv"] + t_q   # residual added in log space
            pred_rv_q     = np.exp(pred_log_rv_q)
            # sign from return direction: negative RV quantile → left tail (put)
            # map RV quantile to return quantile via: |R| ~ sqrt(RV), sign from q
            ret_pct = (1 if q >= 0.5 else -1) * np.sqrt(abs(pred_rv_q)) * 100
            # more rigorous: use normal approximation for sign assignment
            # with the actual directional return distribution
            rows.append({"date": date, "q": q, "ret_pct": ret_pct,
                         "pred_rv": pred_rv, "pred_vol_ann": ann_vol_pct(pred_rv, remaining_mins)})

    if not rows:
        return {}, pd.DataFrame()

    df_q = pd.DataFrame(rows)
    summary = df_q.groupby("q")["ret_pct"].median().to_dict()
    return summary, df_q


def conditional_quantiles_direct(wf: pd.DataFrame, anchor: str,
                                 nu: float, loc: float, scale: float) -> pd.DataFrame:

    remaining_mins = minutes_between(anchor, EXIT_TIME)
    out_rows = []

    for q in QUANTILES:
        t_quantiles = stats.t.ppf(q, df=nu, loc=loc, scale=scale)
        # shift predicted log RV by this residual
        shifted_log_rvs = wf["pred_log_rv"].values + t_quantiles
        shifted_rvs     = np.exp(shifted_log_rvs)
        # convert RV sum → return magnitude, assign direction
        rets_pct = np.sqrt(shifted_rvs) * (1 if q >= 0.5 else -1) * 100
        out_rows.append({
            "q":                q,
            "median_ret_pct":   float(np.median(rets_pct)),
            "mean_ret_pct":     float(np.mean(rets_pct)),
            "p10_ret_pct":      float(np.percentile(rets_pct, 10)),
            "p90_ret_pct":      float(np.percentile(rets_pct, 90)),
        })
    return pd.DataFrame(out_rows)


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_feature_importance(model, feats, anchor, outdir, symbol):
    imp = pd.Series(model.get_feature_importance(), index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(feats) + 1.5))
    imp.plot.barh(ax=ax, color="steelblue")
    ax.set_title(f"{symbol} feature importance ({anchor})")
    ax.set_xlabel("CatBoost importance score")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"feature_importance_{anchor.replace(':','')}.png"))
    plt.close(fig)
    return imp


def plot_wf_predictions(all_wf: dict, outdir: str, symbol: str):
    anchors = [a for a in ANCHORS if a in all_wf and len(all_wf[a])]
    n = len(anchors)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, a in zip(axes, anchors):
        wf = all_wf[a].sort_index()
        ax.plot(wf.index, wf["log_rv_rem"],   lw=0.9, label="actual log RV")
        ax.plot(wf.index, wf["pred_log_rv"],  lw=0.9, label="predicted log RV", alpha=0.8)
        r2 = oos_r2(wf["log_rv_rem"].values, wf["pred_log_rv"].values)
        ax.set_title(f"{a}: OOS R²={r2:.3f}", fontsize=9)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    fig.suptitle(f"{symbol}: walk-forward predicted vs actual log(RV_remaining)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wf_predictions.png"))
    plt.close(fig)


def plot_wf_scatter(all_wf: dict, outdir: str, symbol: str):
    anchors = [a for a in ANCHORS if a in all_wf and len(all_wf[a])]
    n  = len(anchors)
    nc = 3
    nr = int(np.ceil(n / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4.5 * nr))
    axes = np.array(axes).flatten()
    for i, a in enumerate(anchors):
        wf  = all_wf[a]
        ax  = axes[i]
        r2  = oos_r2(wf["log_rv_rem"].values, wf["pred_log_rv"].values)
        rm  = rmse(wf["log_rv_rem"].values, wf["pred_log_rv"].values)
        ax.scatter(wf["pred_log_rv"], wf["log_rv_rem"], s=12, alpha=0.5)
        lims = [min(wf["pred_log_rv"].min(), wf["log_rv_rem"].min()) - 0.2,
                max(wf["pred_log_rv"].max(), wf["log_rv_rem"].max()) + 0.2]
        ax.plot(lims, lims, "r--", lw=0.8)
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_title(f"{a}: OOS R²={r2:.3f}  RMSE={rm:.3f}", fontsize=8)
        ax.set_xlabel("predicted log RV"); ax.set_ylabel("actual log RV")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"{symbol}: OOS scatter (log RV space)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "wf_scatter.png"))
    plt.close(fig)


def plot_residual_diagnostics(all_wf: dict, all_t_fits: dict,
                               outdir: str, symbol: str):
    anchors = [a for a in ANCHORS if a in all_wf and len(all_wf[a])]
    n  = len(anchors)
    fig, axes = plt.subplots(n, 3, figsize=(14, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, 3)

    for i, a in enumerate(anchors):
        res = all_wf[a]["residual"].dropna().values
        nu, loc, scale = all_t_fits[a]

        # histogram + t fit
        ax = axes[i, 0]
        ax.hist(res, bins=35, density=True, alpha=0.65, label="residuals")
        xs = np.linspace(res.min(), res.max(), 300)
        ax.plot(xs, stats.t.pdf(xs, df=nu, loc=loc, scale=scale),
                lw=1.2, color="tab:red", label=f"t(ν={nu:.1f})")
        ax.plot(xs, stats.norm.pdf(xs, res.mean(), res.std(ddof=1)),
                lw=1.0, ls="--", color="tab:green", label="normal")
        ax.set_title(f"{a}: residual distribution", fontsize=8)
        ax.legend(fontsize=7)

        # QQ vs fitted t
        ax = axes[i, 1]
        theoretical = stats.t.ppf(
            np.linspace(0.01, 0.99, len(res)), df=nu, loc=loc, scale=scale)
        empirical = np.sort(res)
        ax.scatter(theoretical, empirical, s=8, alpha=0.6)
        lims = [min(theoretical.min(), empirical.min()),
                max(theoretical.max(), empirical.max())]
        ax.plot(lims, lims, "r--", lw=0.8)
        ax.set_title(f"{a}: QQ vs t(ν={nu:.1f})", fontsize=8)
        ax.set_xlabel("theoretical t quantiles")
        ax.set_ylabel("empirical quantiles")

        # ACF of residuals (serial correlation check)
        ax = axes[i, 2]
        max_lag = min(20, len(res) // 5)
        acf_vals = [1.0] + [float(pd.Series(res).autocorr(lag=l))
                             for l in range(1, max_lag + 1)]
        ci = 1.96 / np.sqrt(len(res))
        ax.bar(range(len(acf_vals)), acf_vals, width=0.4, color="steelblue")
        ax.axhline(ci,  ls="--", color="red", lw=0.8)
        ax.axhline(-ci, ls="--", color="red", lw=0.8)
        ax.set_title(f"{a}: ACF of residuals", fontsize=8)
        ax.set_xlabel("lag"); ax.set_ylabel("autocorrelation")

    fig.suptitle(f"{symbol}: residual diagnostics (OOS)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "residual_diagnostics.png"))
    plt.close(fig)


def plot_feature_importance_combined(imp_by_anchor: dict, outdir: str, symbol: str):
    """Heatmap of feature importance across anchor times."""
    df = pd.DataFrame(imp_by_anchor).T   # index=anchor, cols=features
    df = df[df.mean().sort_values(ascending=False).index]  # sort by mean importance
    fig, ax = plt.subplots(figsize=(max(8, len(df.columns) * 0.8), 4))
    im = ax.imshow(df.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="importance")
    ax.set_title(f"{symbol}: feature importance across anchor times")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_importance.png"))
    plt.close(fig)
    df.to_csv(os.path.join(outdir, "feature_importance.csv"))


# ── main per-symbol runner ────────────────────────────────────────────────────
def run_symbol(data_path: str, symbol: str, args):
    outdir = args.outdir or f"phase2_{symbol.lower()}"
    os.makedirs(outdir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"  {symbol}: {data_path}")
    print(f"{'='*60}")

    df = load_minute_data(data_path)
    df = drop_bad_days(df, 300)

    if args.expiry_dates:
        with open(args.expiry_dates) as f:
            expiries = {pd.to_datetime(l.strip()).date()
                        for l in f if l.strip()}
    else:
        expiries = infer_expiry_days(df, symbol)

    expiry_df = df[df["date"].isin(expiries)].copy()
    expiry_dates_sorted = sorted(expiries & set(expiry_df["date"].unique()))
    n_days = len(expiry_dates_sorted)
    print(f"  {n_days} expiry days  ({expiry_dates_sorted[0]} … {expiry_dates_sorted[-1]})")

    vix_df = load_vix(args.vix)
    if vix_df is not None:
        print(f"  India VIX loaded: {len(vix_df)} rows (using prior-day close — no lookahead)")
    else:
        print("  India VIX: not supplied, feature skipped")

    prior_rv = build_prior_expiry_rv(expiry_df, expiry_dates_sorted)

    # ── per-anchor loop ──
    all_wf       = {}
    all_t_fits   = {}
    all_cq       = {}
    imp_by_anchor = {}
    summary_rows  = []

    for anchor in ANCHORS:
        print(f"\n  ── anchor {anchor} ──")
        feat_df = build_features(expiry_df, anchor, prior_rv, vix_df)
        if len(feat_df) < 40:
            print(f"     only {len(feat_df)} rows, skipping")
            continue

        # walk-forward OOS predictions
        wf = walk_forward(feat_df, anchor)
        if len(wf) < 10:
            continue
        all_wf[anchor] = wf

        r2   = oos_r2(wf["log_rv_rem"].values, wf["pred_log_rv"].values)
        rm   = rmse(wf["log_rv_rem"].values, wf["pred_log_rv"].values)
        n_oos = len(wf)
        print(f"     OOS days={n_oos}  R²={r2:.4f}  RMSE={rm:.4f}")

        # fit scaled-t to OOS residuals
        nu, loc, scale = fit_scaled_t(wf["residual"].dropna().values)
        all_t_fits[anchor] = (nu, loc, scale)
        print(f"     t-fit: ν={nu:.2f}  loc={loc:.4f}  scale={scale:.4f}")

        # conditional quantile table
        cq = conditional_quantiles_direct(wf, anchor, nu, loc, scale)
        all_cq[anchor] = cq
        cq.to_csv(os.path.join(outdir,
            f"conditional_quantiles_{anchor.replace(':','')}.csv"), index=False)

        # full-data model + feature importance
        full_model, feats = fit_full_model(feat_df, anchor, outdir)
        imp = plot_feature_importance(full_model, feats, anchor, outdir, symbol)
        imp_by_anchor[anchor] = imp.to_dict()

        summary_rows.append({
            "anchor":       anchor,
            "n_train_days": len(feat_df),
            "n_oos_days":   n_oos,
            "oos_r2":       r2,
            "oos_rmse":     rm,
            "t_nu":         nu,
            "t_loc":        loc,
            "t_scale":      scale,
        })

    if not summary_rows:
        print("  No anchors produced results. Check data.")
        return

    # ── aggregate outputs ──
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(outdir, "model_summary.csv"), index=False)

    # combined conditional quantile table (one row per anchor × quantile)
    cq_rows = []
    for anchor, cq in all_cq.items():
        for _, row in cq.iterrows():
            cq_rows.append({"anchor": anchor, **row.to_dict()})
    cq_all = pd.DataFrame(cq_rows)
    cq_all.to_csv(os.path.join(outdir, "conditional_quantiles_pct.csv"), index=False)

    # tail fit params
    t_fit_df = pd.DataFrame([
        {"anchor": a, "nu": nu, "loc": loc, "scale": scale}
        for a, (nu, loc, scale) in all_t_fits.items()
    ])
    t_fit_df.to_csv(os.path.join(outdir, "tail_fit_params.csv"), index=False)

    # plots
    plot_wf_predictions(all_wf, outdir, symbol)
    plot_wf_scatter(all_wf, outdir, symbol)
    plot_residual_diagnostics(all_wf, all_t_fits, outdir, symbol)
    plot_feature_importance_combined(imp_by_anchor, outdir, symbol)

    # ── console summary ──
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n{'='*60}")
    print(f"  {symbol}: model summary")
    print(f"{'='*60}")
    print(summary.to_string(index=False))

    print(f"\n  {symbol}: conditional quantiles of R[t->14:30] (median, percent)")
    pivot = cq_all[cq_all["q"].isin([0.025, 0.05, 0.25, 0.50, 0.75, 0.95, 0.975])].pivot(
        index="anchor", columns="q", values="median_ret_pct")
    pivot.columns = [f"q{c}" for c in pivot.columns]
    print(pivot.to_string())
    print(f"\n  outputs → {outdir}/")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data",          default=None)
    p.add_argument("--symbol",        default=None)
    p.add_argument("--vix",           default=None, help="India VIX daily CSV")
    p.add_argument("--expiry-dates",  default=None)
    p.add_argument("--outdir",        default=None)
    args = p.parse_args()

    if args.data:
        run_symbol(args.data, args.symbol or "NIFTY", args)
    else:
        for path, sym in DEFAULT_DATASETS:
            if not os.path.exists(path):
                print(f"[skip] {path} not found")
                continue
            run_symbol(path, sym, args)


if __name__ == "__main__":
    main()