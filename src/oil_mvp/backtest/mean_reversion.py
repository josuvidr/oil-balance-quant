import pandas as pd
import numpy as np


def generate_signal(df: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5):
    df = df.copy()

    position = 0
    positions = []

    for z in df["spread_zscore"]:
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        else:
            if abs(z) < exit_z:
                position = 0

        positions.append(position)

    df["position"] = positions

    return df



def compute_returns(df: pd.DataFrame, target_vol=0.10):
    df = df.copy()

    df["wti_ret"] = df["wti"].pct_change()
    df["brent_ret"] = df["brent"].pct_change()

    df["spread_return"] = df["brent_ret"] - df["wti_ret"]

    # Rolling realized vol (annualized)
    rolling_vol = df["spread_return"].rolling(60).std() * (252 ** 0.5)
    rolling_vol = rolling_vol.clip(lower=0.01)


    # Position scaling to target vol
    df["vol_scaler"] = target_vol / rolling_vol

    # Cap leverage
    df["vol_scaler"] = df["vol_scaler"].clip(0, 5)

    df["strategy_return"] = (
        df["position"].shift(1)
        * df["spread_return"]
        * df["vol_scaler"]
    )
    # Transaction costs (bps) applied on position changes
    tc_bps = 1.0  # 1 bp per rebalance; adjust later
    turnover = (df["position"].diff().abs()).fillna(0)
    df["tc"] = turnover * (tc_bps / 10000.0)

    df["strategy_return"] = df["strategy_return"] - df["tc"]

    df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()

    return df


def compute_performance_metrics(df: pd.DataFrame):
    df = df.copy()
    rets = df["strategy_return"].dropna()

    sharpe = rets.mean() / rets.std() * (252 ** 0.5)

    equity = df["equity_curve"].dropna()
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    total_return = equity.iloc[-1]

    n_years = len(rets) / 252
    ann_return = total_return ** (1 / n_years) - 1

    return {
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "total_return_multiple": float(total_return),
        "ann_return": float(ann_return),
    }


