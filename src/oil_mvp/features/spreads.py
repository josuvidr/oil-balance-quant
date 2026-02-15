import pandas as pd


def compute_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["brent_wti_spread"] = df["brent"] - df["wti"]

    return df


def compute_zscore(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    df = df.copy()

    spread = df["brent_wti_spread"]

    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()

    df["spread_zscore"] = (spread - rolling_mean) / rolling_std

    return df

import numpy as np


def compute_hedge_ratio(df: pd.DataFrame):
    x = df["wti"].values
    y = df["brent"].values

    beta = np.polyfit(x, y, 1)[0]

    df["hedge_spread"] = df["brent"] - beta * df["wti"]

    return df, beta
