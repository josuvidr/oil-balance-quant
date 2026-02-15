import pandas as pd
import numpy as np


def subperiod_slices(df: pd.DataFrame, breaks: list[str]) -> list[tuple[str, str]]:
    """
    breaks: list of ISO dates like ["2008-01-01", "2014-01-01", ...]
    returns consecutive intervals
    """
    b = [df.index.min().strftime("%Y-%m-%d")] + breaks + [df.index.max().strftime("%Y-%m-%d")]
    return list(zip(b[:-1], b[1:]))


def perf_from_equity(df: pd.DataFrame) -> dict:
    rets = df["strategy_return"].dropna()
    if len(rets) < 50:
        return {"sharpe": float("nan"), "ann_return": float("nan"), "max_dd": float("nan"), "multiple": float("nan")}

    sharpe = rets.mean() / rets.std() * (252 ** 0.5)

    equity = df["equity_curve"].dropna()
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    multiple = equity.iloc[-1] / equity.iloc[0]
    n_years = len(rets) / 252
    ann_return = multiple ** (1 / n_years) - 1

    return {
        "sharpe": float(sharpe),
        "ann_return": float(ann_return),
        "max_dd": float(max_dd),
        "multiple": float(multiple),
        "turnover_daily_mean": float(df["position"].diff().abs().fillna(0).mean()),
        "avg_hold_days": float(1 / max(df["position"].diff().abs().fillna(0).mean(), 1e-9)),
    }


def run_subperiod_report(df: pd.DataFrame, breaks: list[str]) -> pd.DataFrame:
    rows = []
    for start, end in subperiod_slices(df, breaks):
        sub = df.loc[start:end].copy()
        m = perf_from_equity(sub)
        rows.append({"start": start, "end": end, **m})
    return pd.DataFrame(rows)


def stress_test_costs(df: pd.DataFrame, tc_bps_list=(0.5, 1.0, 2.0, 5.0)) -> pd.DataFrame:
    """
    Recompute equity by applying alternative transaction cost bps (simple overlay).
    Assumes df has position and spread_return and vol_scaler, and original strategy_return.
    """
    rows = []
    base = df.copy()
    turnover = base["position"].diff().abs().fillna(0)

    for tc_bps in tc_bps_list:
        tc = turnover * (tc_bps / 10000.0)
        rets = (base["strategy_return"] + base.get("tc", 0)) - tc  # remove old tc if present, apply new
        equity = (1 + rets.fillna(0)).cumprod()
        tmp = base.copy()
        tmp["strategy_return_stress"] = rets
        tmp["equity_stress"] = equity

        # metrics
        r = tmp["strategy_return_stress"].dropna()
        sharpe = r.mean() / r.std() * (252 ** 0.5)
        dd = equity / equity.cummax() - 1.0

        rows.append({
            "tc_bps": float(tc_bps),
            "sharpe": float(sharpe),
            "max_dd": float(dd.min()),
            "multiple": float(equity.iloc[-1]),
        })

    return pd.DataFrame(rows)


def beta_stability(df: pd.DataFrame) -> dict:
    b = df["beta_used"].dropna()
    return {
        "beta_mean": float(b.mean()),
        "beta_std": float(b.std()),
        "beta_p05": float(b.quantile(0.05)),
        "beta_p95": float(b.quantile(0.95)),
    }
