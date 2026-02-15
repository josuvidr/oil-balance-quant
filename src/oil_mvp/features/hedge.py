import numpy as np
import pandas as pd


def rolling_beta_ols(
    df: pd.DataFrame,
    x_col: str = "wti",
    y_col: str = "brent",
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """
    Rolling OLS beta of y ~ alpha + beta*x using windowed covariance/variance.
    Uses beta = Cov(x,y)/Var(x).
    """
    if min_periods is None:
        min_periods = window

    x = df[x_col].astype(float)
    y = df[y_col].astype(float)

    x_mean = x.rolling(window, min_periods=min_periods).mean()
    y_mean = y.rolling(window, min_periods=min_periods).mean()

    cov_xy = ((x - x_mean) * (y - y_mean)).rolling(window, min_periods=min_periods).mean()
    var_x = ((x - x_mean) ** 2).rolling(window, min_periods=min_periods).mean()

    beta = cov_xy / var_x
    return beta.rename("beta")


def build_hedged_spread(
    df: pd.DataFrame,
    beta: pd.Series,
    x_col: str = "wti",
    y_col: str = "brent",
    lag_beta: int = 1,
) -> pd.DataFrame:
    """
    Hedged spread = y - beta_{t-lag} * x to avoid lookahead.
    """
    out = df.copy()
    out["beta"] = beta
    out["beta_used"] = out["beta"].shift(lag_beta)
    out["hedge_spread"] = out[y_col] - out["beta_used"] * out[x_col]
    return out
