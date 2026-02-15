from fredapi import Fred
import pandas as pd

from oil_mvp.config import load_settings
from oil_mvp.features.spreads import compute_zscore
from oil_mvp.features.hedge import rolling_beta_ols, build_hedged_spread
from oil_mvp.backtest.mean_reversion import (
    generate_signal,
    compute_returns,
    compute_performance_metrics,
)


def fetch_prices(api_key: str) -> pd.DataFrame:
    fred = Fred(api_key=api_key)

    wti = fred.get_series("DCOILWTICO")
    brent = fred.get_series("DCOILBRENTEU")

    df = pd.DataFrame({"wti": wti, "brent": brent}).dropna()
    df.index.name = "date"
    return df


def build_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    # Rolling beta (252 trading days), then lag beta by 1 day to avoid lookahead
    beta = rolling_beta_ols(df, window=252, min_periods=200)
    df = build_hedged_spread(df, beta, lag_beta=1)

    # Use hedged spread as the tradable spread for zscore + signal
    df["brent_wti_spread"] = df["hedge_spread"]

    # z-score on hedged spread
    df = compute_zscore(df, window=60)

    # Strategy + returns
    df = generate_signal(df, entry_z=2.0, exit_z=0.5)
    df = compute_returns(df, target_vol=0.10)

    return df


def main():
    s = load_settings()
    df = fetch_prices(s.fred_api_key)
    df = build_pipeline(df)

    metrics = compute_performance_metrics(df)

    print(df[["wti", "brent", "beta_used", "brent_wti_spread", "spread_zscore", "position", "equity_curve"]].tail())
    print("\nPerformance:")
    print(metrics)


if __name__ == "__main__":
    main()
