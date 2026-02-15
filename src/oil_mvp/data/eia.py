from fredapi import Fred
import pandas as pd
from oil_mvp.config import load_settings
from oil_mvp.data.prices import fetch_prices
from oil_mvp.features.spreads import compute_spread, compute_zscore


def fetch_crude_stocks(api_key):
    fred = Fred(api_key=api_key)

    # Weekly US Crude Stocks
    series = fred.get_series("WCESTUS1")

    df = series.to_frame(name="crude_stocks")
    df.index.name = "date"

    return df


def main():
    s = load_settings()
    df = fetch_prices(s.fred_api_key)

    df = compute_spread(df)
    df = compute_zscore(df)

    print(df.tail())



if __name__ == "__main__":
    main()
