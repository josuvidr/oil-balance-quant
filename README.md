# oil-balance-quant
Systematic US oil fundamentals model: inventory balance reconstruction, spread fair value estimation (Brent-WTI, HO crack, time spreads), walk-forward backtesting &amp; scenario engine.

# Oil Balance Quant

Systematic US oil fundamentals model reconstructing crude and distillate supply/demand balances to estimate fair value of key spreads (Brent-WTI, HO crack, time spreads) with walk-forward backtesting and scenario analysis.

---

## Overview

This project builds a weekly US-centric oil market model using publicly available EIA data.  

It reconstructs inventory balances for:

- US crude oil
- US distillates

The balance signals are then used to estimate fair value for:

- Brent–WTI spread
- Heating Oil crack spread
- WTI time spreads

All trading signals are evaluated using strict walk-forward backtesting to avoid look-ahead bias.

---

## Methodology

### 1. Balance reconstruction

ΔInventory = Supply − Demand  

Supply and demand are reconstructed using EIA weekly data:

Crude:
- Production
- Imports
- Exports
- Refinery inputs

Distillates:
- Production
- Imports
- Exports
- Product supplied

Inventory surprise = Actual − Rolling mean (4 weeks)

---

### 2. Fair value modeling

Spreads are modeled using:

- Inventory surprises
- Refinery utilization
- Seasonal effects

Models are estimated using rolling walk-forward regression.

---

### 3. Backtesting framework

- Strict no look-ahead
- Position lagging
- Transaction costs
- Walk-forward re-estimation
- PnL attribution

---

## Project Structure

src/ → data ingestion, features, models, backtest engine  
configs/ → API & parameters  
data/ → cached parquet files (not versioned)  

---

## Disclaimer

This is a research project for educational purposes and does not constitute investment advice.

