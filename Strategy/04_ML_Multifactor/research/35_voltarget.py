"""
35_voltarget.py  -- Drawdown control via VOLATILITY TARGETING / VOL MANAGEMENT
=============================================================================
Method family: scale the OVERALL exposure of the long-only top-decile net
return stream by a time-varying, LOOK-AHEAD-FREE factor scale_t in [0, cap].

Three variants compared:
  (a) CVS   Barroso & Santa-Clara constant-vol scaling:
              scale_t = target_ann_vol / realized_ann_vol(prev 6m, .shift(1))
              target grid {12%,16%,20%}
  (b) MM    Moreira & Muir vol management:
              scale_t proportional to 1/realized_var(prev k months), k in {1,2,3}
              normalized so mean(scale)=1 over an EXPANDING window (no look-ahead)
  (c) DVS   predicted-vol scaling: EWMA (lambda grid) and GARCH(1,1) (arch, rolling
              refit / one-step forecast) -> scale_t = target/predicted_vol(.shift(1))

Exposure not invested sits in cash (rf=0 assumption, conservative -> understates
scaled return a touch). cap=1.0 (de-risk only) and cap=1.5 (mild lever) both run.

NO LOOK-AHEAD self-checks:
  - every realized-vol input to scale_t uses only returns up to t-1 (.shift(1))
  - MM normalization uses an EXPANDING mean of scale (info available at t)
  - GARCH is refit on an expanding history each month, forecasting 1 step ahead;
    never a single full-sample fit back-filled.
Baseline to beat (calmar): 0.82.
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REPO = "/Users/shenboheng/Documents/ClaudeCode/factor_new"
ANN = 12  # monthly

# ---------------------------------------------------------------- load net stream
nav = pd.read_csv(f"{REPO}/results/26_nav.csv")
nav.columns = [c.lstrip("﻿") for c in nav.columns]
nav["dt"] = pd.to_datetime(nav["dt"])
nav = nav.set_index("dt")
r = nav["V1多头"].pct_change(fill_method=None).dropna()  # 2018-02 .. 2025-11, net of 0.3% turnover
r.name = "ret"
print(f"net stream: {len(r)} months  {r.index.min().date()} -> {r.index.max().date()}")


def perf(x: pd.Series) -> dict:
    x = x.dropna()
    n = len(x)
    ann = (1 + x).prod() ** (ANN / n) - 1
    vol = x.std(ddof=1) * np.sqrt(ANN)
    sharpe = ann / vol if vol > 0 else np.nan
    curve = (1 + x).cumprod()
    mdd = (curve / curve.cummax() - 1).min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return dict(ann=ann, vol=vol, sharpe=sharpe, mdd=mdd, calmar=calmar)


def apply_scale(scale: pd.Series, cap: float) -> pd.Series:
    """scaled_ret_t = clip(scale_t,0,cap) * r_t ; cash earns 0."""
    s = scale.reindex(r.index).clip(lower=0.0, upper=cap)
    # months with no scale yet (warm-up) -> hold baseline exposure (scale=1) so we
    # don't silently discard early returns; keeps series comparable in length.
    s = s.fillna(1.0)
    return s * r, s


def turnover_of_scale(s: pd.Series) -> float:
    """mean |delta scale| per month = extra leverage rebalancing (proxy for cost)."""
    return s.diff().abs().mean()


base = perf(r)
print("\n=== BASELINE (V1 long, net) ===")
print({k: round(v, 4) for k, v in base.items()})

results = []


def record(name, variant, param, cap, scaled, s):
    p = perf(scaled)
    p["turnover_scale"] = turnover_of_scale(s)
    p["mean_expo"] = s.mean()
    p.update(dict(method=name, variant=variant, param=param, cap=cap))
    results.append(p)
    return p


# ================================================================ (a) CVS
# realized ann vol over trailing 6m, lagged one month
rv6 = r.rolling(6).std(ddof=1) * np.sqrt(ANN)
rv6_lag = rv6.shift(1)
for tv in [0.12, 0.16, 0.20]:
    scale = tv / rv6_lag
    for cap in [1.0, 1.5]:
        sc, s = apply_scale(scale, cap)
        record("CVS", f"target={tv:.0%}", tv, cap, sc, s)

# ================================================================ (b) Moreira-Muir
# scale ~ 1/realized_var(prev k m); normalize by EXPANDING mean of the raw signal
for k in [1, 2, 3]:
    var_k = (r.rolling(k).var(ddof=1) if k > 1 else r.rolling(1).apply(lambda z: z.iloc[0] ** 2))
    raw = (1.0 / var_k).shift(1)              # info up to t-1
    # expanding mean normalization -> mean exposure ~1 using only past info
    norm = raw.expanding(min_periods=6).mean()
    scale = raw / norm
    for cap in [1.0, 1.5]:
        sc, s = apply_scale(scale, cap)
        record("MM", f"k={k}", k, cap, sc, s)

# ================================================================ (c) DVS predicted vol
# --- EWMA one-step-ahead vol forecast (RiskMetrics style), lagged
def ewma_vol(returns: pd.Series, lam: float) -> pd.Series:
    """sigma2_t forecast for month t using returns up to t-1 (recursive)."""
    out = pd.Series(index=returns.index, dtype=float)
    var = np.nan
    prev = None
    for t, x in returns.items():
        # forecast for THIS month uses variance built from strictly prior months
        out[t] = np.sqrt(var * ANN) if var == var else np.nan
        # then update with realized x (becomes history for next month)
        if var != var:  # nan -> seed
            var = x ** 2
        else:
            var = lam * var + (1 - lam) * x ** 2
        prev = x
    return out  # already a forecast-for-t (no future info) -> DO NOT shift again


for lam in [0.80, 0.90, 0.94]:
    pv = ewma_vol(r, lam)          # forecast for month t from history < t
    for tv in [0.16, 0.20]:
        scale = tv / pv
        for cap in [1.0, 1.5]:
            sc, s = apply_scale(scale, cap)
            record("DVS-EWMA", f"lam={lam},tv={tv:.0%}", (lam, tv), cap, sc, s)

# --- GARCH(1,1) rolling one-step forecast (arch), expanding window refit
try:
    from arch import arch_model
    rp = (r * 100.0)               # arch prefers pct-scale
    idx = r.index
    garch_fc = pd.Series(index=idx, dtype=float)   # ann vol forecast for month t
    MINOBS = 36
    for i in range(len(idx)):
        if i < MINOBS:
            continue
        hist = rp.iloc[:i]         # strictly < t  -> no look-ahead
        try:
            am = arch_model(hist, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            f = res.forecast(horizon=1, reindex=False)
            sig_m = np.sqrt(f.variance.values[-1, 0]) / 100.0   # monthly sigma
            garch_fc.iloc[i] = sig_m * np.sqrt(ANN)
        except Exception:
            garch_fc.iloc[i] = np.nan
    for tv in [0.16, 0.20]:
        scale = tv / garch_fc
        for cap in [1.0, 1.5]:
            sc, s = apply_scale(scale, cap)
            record("DVS-GARCH", f"tv={tv:.0%}", tv, cap, sc, s)
    garch_ok = True
except Exception as e:
    print("GARCH skipped:", e)
    garch_ok = False

# ================================================================ report
res = pd.DataFrame(results)
res = res[["method", "variant", "cap", "ann", "vol", "sharpe", "mdd",
           "calmar", "turnover_scale", "mean_expo"]]
res = res.sort_values("calmar", ascending=False).reset_index(drop=True)
pd.set_option("display.width", 200, "display.max_rows", 200)

fmt = res.copy()
for c in ["ann", "vol", "mdd", "mean_expo"]:
    fmt[c] = (fmt[c] * 100).round(1)
for c in ["sharpe", "calmar", "turnover_scale"]:
    fmt[c] = fmt[c].round(3)
print("\n=== ALL VARIANTS (sorted by calmar) ===")
print(fmt.to_string(index=False))

print(f"\nBASELINE calmar={base['calmar']:.3f}  ann={base['ann']:.1%}  "
      f"mdd={base['mdd']:.1%}  sharpe={base['sharpe']:.3f}")

# picks: balanced (best calmar with ann drop <= ~1.5pp) and aggressive (lowest mdd)
cand = res.copy()
balanced = cand[cand["ann"] >= base["ann"] - 0.015].sort_values("calmar", ascending=False)
print("\n--- BALANCED candidates (ann within 1.5pp of baseline, top calmar) ---")
print(balanced.head(6)[["method", "variant", "cap", "ann", "sharpe", "mdd", "calmar", "turnover_scale"]]
      .assign(ann=lambda d: (d.ann*100).round(1), mdd=lambda d: (d.mdd*100).round(1),
              sharpe=lambda d: d.sharpe.round(3), calmar=lambda d: d.calmar.round(3),
              turnover_scale=lambda d: d.turnover_scale.round(3)).to_string(index=False))

aggressive = cand.sort_values("mdd", ascending=False)  # mdd closest to 0
print("\n--- AGGRESSIVE candidates (smallest |mdd|) ---")
print(aggressive.head(6)[["method", "variant", "cap", "ann", "sharpe", "mdd", "calmar", "turnover_scale"]]
      .assign(ann=lambda d: (d.ann*100).round(1), mdd=lambda d: (d.mdd*100).round(1),
              sharpe=lambda d: d.sharpe.round(3), calmar=lambda d: d.calmar.round(3),
              turnover_scale=lambda d: d.turnover_scale.round(3)).to_string(index=False))

res.to_csv(f"{REPO}/results/35_voltarget.csv", index=False)
print(f"\nsaved -> {REPO}/results/35_voltarget.csv")

# ---- sub-period sanity: how does the best balanced pick behave in the two crash windows
print("\n=== CRASH-WINDOW behavior of a representative CVS pick (target16%,cap1.0) ===")
sc16, s16 = apply_scale((0.16 / rv6_lag), 1.0)
for lab, a, b in [("2018 bear", "2018-01", "2018-12"),
                  ("2023-12~2024-07 microcap", "2023-12", "2024-07")]:
    seg_b = r.loc[a:b]; seg_s = sc16.loc[a:b]
    print(f"{lab:28s} base={( (1+seg_b).prod()-1):+.1%}  scaled={((1+seg_s).prod()-1):+.1%}  "
          f"mean_expo={s16.loc[a:b].mean():.2f}")
