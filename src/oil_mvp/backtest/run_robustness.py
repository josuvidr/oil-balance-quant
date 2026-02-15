from oil_mvp.config import load_settings
from oil_mvp.data.prices import fetch_prices, build_pipeline
from oil_mvp.backtest.robustness import run_subperiod_report, stress_test_costs, beta_stability


def main():
    s = load_settings()
    df = fetch_prices(s.fred_api_key)
    df = build_pipeline(df)

    print("Beta stability:", beta_stability(df))

    # Example breaks around common regimes
    breaks = ["2008-01-01", "2014-01-01", "2020-01-01"]
    print("\nSubperiod report:")
    print(run_subperiod_report(df, breaks))

    print("\nCost stress test:")
    print(stress_test_costs(df, tc_bps_list=(0.5, 1.0, 2.0, 5.0)))


if __name__ == "__main__":
    main()
