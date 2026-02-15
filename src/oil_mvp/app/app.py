import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from oil_mvp.config import load_settings
from oil_mvp.data.prices import fetch_prices, build_pipeline
from oil_mvp.backtest.robustness import run_subperiod_report, stress_test_costs, beta_stability


st.set_page_config(page_title="Oil Balance Quant — Brent/WTI Pair", layout="wide")

st.title("Oil Balance Quant — Brent/WTI Pair (FRED)")
st.caption("Walk-forward hedge ratio • Mean reversion on hedged spread • Vol targeting • Costs")

with st.sidebar:
    st.header("Strategy params")
    entry_z = st.slider("Entry z", 0.5, 4.0, 2.0, 0.1)
    exit_z = st.slider("Exit z", 0.0, 2.0, 0.5, 0.1)
    z_win = st.slider("Z-score window", 20, 200, 60, 5)
    beta_win = st.slider("Beta window", 60, 600, 252, 10)
    target_vol = st.slider("Target vol (ann.)", 0.02, 0.30, 0.10, 0.01)
    tc_bps = st.slider("Transaction cost (bps)", 0.0, 10.0, 1.0, 0.5)

# Build data
s = load_settings()
raw = fetch_prices(s.fred_api_key)

# Patch params into pipeline by reusing functions directly
from oil_mvp.features.hedge import rolling_beta_ols, build_hedged_spread
from oil_mvp.features.spreads import compute_zscore
from oil_mvp.backtest.mean_reversion import generate_signal, compute_returns, compute_performance_metrics

beta = rolling_beta_ols(raw, window=beta_win, min_periods=int(beta_win * 0.8))
df = build_hedged_spread(raw, beta, lag_beta=1)
df["brent_wti_spread"] = df["hedge_spread"]

df = compute_zscore(df, window=z_win)
df = generate_signal(df, entry_z=entry_z, exit_z=exit_z)
df = compute_returns(df, target_vol=target_vol)

# override costs stress-style
turnover = df["position"].diff().abs().fillna(0)
df["strategy_return"] = (df["strategy_return"] + df.get("tc", 0)) - turnover * (tc_bps / 10000.0)
df["equity_curve"] = (1 + df["strategy_return"].fillna(0)).cumprod()

metrics = compute_performance_metrics(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe", f"{metrics['sharpe']:.2f}")
c2.metric("Max DD", f"{metrics['max_drawdown_pct']*100:.1f}%")
c3.metric("Total multiple", f"{metrics['total_return_multiple']:.2f}x")
c4.metric("Ann. return", f"{metrics['ann_return']*100:.1f}%")

st.subheader("Equity curve")
fig = plt.figure()
plt.plot(df.index, df["equity_curve"])
plt.xlabel("Date")
plt.ylabel("Equity (multiple)")
st.pyplot(fig)

st.subheader("Drawdown")
dd = df["equity_curve"] / df["equity_curve"].cummax() - 1.0
fig2 = plt.figure()
plt.plot(df.index, dd)
plt.xlabel("Date")
plt.ylabel("Drawdown")
st.pyplot(fig2)

st.subheader("Hedge spread + z-score")
fig3 = plt.figure()
plt.plot(df.index, df["brent_wti_spread"])
plt.xlabel("Date")
plt.ylabel("Hedged spread")
st.pyplot(fig3)

fig4 = plt.figure()
plt.plot(df.index, df["spread_zscore"])
plt.axhline(entry_z, linestyle="--")
plt.axhline(-entry_z, linestyle="--")
plt.axhline(exit_z, linestyle=":")
plt.axhline(-exit_z, linestyle=":")
plt.xlabel("Date")
plt.ylabel("Z-score")
st.pyplot(fig4)

st.subheader("Robustness")
st.caption("Subperiod splits + transaction-cost stress + beta stability")

breaks = ["2008-01-01", "2014-01-01", "2020-01-01"]
sub = run_subperiod_report(df, breaks)
st.dataframe(sub, use_container_width=True)

stress = stress_test_costs(df, tc_bps_list=(0.5, 1.0, 2.0, 5.0))
st.dataframe(stress, use_container_width=True)

st.json(beta_stability(df))
