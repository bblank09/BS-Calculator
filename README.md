# Black-Scholes Options Pricing Calculator

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Tests: 39 passed](https://img.shields.io/badge/tests-39%20passed-brightgreen)
![Dependencies: numpy scipy pandas yfinance](https://img.shields.io/badge/dependencies-numpy%20%7C%20scipy%20%7C%20pandas%20%7C%20yfinance-orange)

A from-scratch implementation of the Black-Scholes model in Python — pricing European options, computing Greeks, solving for implied volatility via Newton-Raphson, and comparing theoretical prices against live market data from Yahoo Finance.

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Motivation & Learning Objectives](#2-motivation--learning-objectives)
3. [System Architecture](#3-system-architecture)
4. [Implementation](#4-implementation)
5. [Concurrency Model](#5-concurrency-model)
6. [Async Bridge](#6-async-bridge)
7. [Comparison with Alternative Approaches](#7-comparison-with-alternative-approaches)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [Project Structure](#10-project-structure)
11. [Quickstart](#11-quickstart)
12. [Running Tests](#12-running-tests)
13. [Dependencies](#13-dependencies)
14. [References](#14-references)

---

## 1. Abstract

This project implements the **Black-Scholes (BS) model** entirely from first principles in Python, without relying on any pre-built options pricing library. The calculator prices European call and put options, computes the five primary option Greeks (Delta, Gamma, Theta, Vega), estimates historical volatility from rolling log-returns, and inverts the BS formula via Newton-Raphson iteration to solve for implied volatility (IV). A `main()` demo fetches live AAPL market data through `yfinance`, prints a multi-strike pricing table, and compares theoretical BS prices with last-traded market prices side-by-side.

**Key insight:** Implementing Black-Scholes from scratch — rather than calling a library — reveals exactly which real-world assumptions the model violates. The volatility smile visible in the market comparison output (IV varies across strikes) is direct empirical evidence of where the constant-σ assumption breaks down.

---

## 2. Motivation & Learning Objectives

Most quant libraries (`QuantLib`, `mibian`, `py_vollib`) abstract away the mechanics of the BS formula, Greeks derivation, and IV solving. This project builds the full model from scratch to answer:

1. **Mathematics:** How does the BS closed-form solution follow from the log-normal price assumption and no-arbitrage conditions?
2. **Greeks:** What do Delta, Gamma, Theta, and Vega measure, and how do they relate to the BS formula analytically?
3. **Root-finding:** How does Newton-Raphson converge on implied volatility, and when does it fail?
4. **Volatility:** What is the difference between historical volatility (backward-looking, realized) and implied volatility (forward-looking, market-derived)?
5. **Market reality:** How large is the gap between BS theoretical prices and actual market prices, and what does that gap tell us?

The implementation is intentionally minimal — no metaclass magic, no decorators, no abstractions beyond what the mathematics requires. Every design decision is visible in the source.

---

## 3. System Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    bs_calculator.py                      │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │   bs_price   │◄───│  bs_greeks   │                   │
│  │  (BS formula)│    │ (δ, γ, θ, ν) │                   │
│  └──────┬───────┘    └──────┬───────┘                   │
│         │                   │                           │
│         ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐                   │
│  │ implied_vol  │    │pricing_table │                   │
│  │(Newton-Raph) │    │ (DataFrame)  │                   │
│  └──────────────┘    └──────┬───────┘                   │
│                             │                           │
│  ┌──────────────┐           ▼                           │
│  │historical_vol│    ┌──────────────┐                   │
│  │  (yfinance)  │───►│    main()    │                   │
│  └──────────────┘    │ (AAPL demo)  │                   │
│                      └──────────────┘                   │
└─────────────────────────────────────────────────────────┘
         │
         ▼  (network I/O — blocking)
   Yahoo Finance API
```

### 3.2 Data Flow in `main()`

```
1. yf.Ticker("AAPL").history(period="5d")   → latest closing price S
2. yf.Ticker("AAPL").history(period="1y")   → 30-day rolling historical vol σ
3. pricing_table(S, strikes, T, r, σ)       → BS prices + Greeks, printed as table
4. tk.option_chain(nearest_expiry)          → market lastPrice per strike
5. bs_price + implied_vol per strike        → Market vs BS comparison table
```

### 3.3 Dependency Graph (pure functions only)

```
bs_price ──────────────────► implied_vol
bs_price ──────────────────► pricing_table
bs_greeks ─────────────────► pricing_table
historical_vol (yfinance) ──► main()
```

---

## 4. Implementation

### 4.1 Core BS Formula — `bs_price`

The Black-Scholes closed-form price for a European option:

```
d1 = [ ln(S/K) + (r + σ²/2)·T ] / (σ·√T)
d2 = d1 − σ·√T

Call = S·N(d1) − K·e^(−rT)·N(d2)
Put  = K·e^(−rT)·N(−d2) − S·N(−d1)
```

`N(·)` is the standard normal CDF (`scipy.stats.norm.cdf`).

**Parameters:**

| Symbol | Variable | Meaning |
|--------|----------|---------|
| S | `S` | Current underlying price |
| K | `K` | Strike price |
| T | `T` | Time to expiry in years (e.g. `30/365`) |
| r | `r` | Continuously compounded risk-free rate (US 3-month T-bill ≈ 0.05) |
| σ | `sigma` | Annualized volatility |

**Edge case handling:**

| Condition | Behaviour |
|-----------|-----------|
| `T == 0` | Returns intrinsic value: `max(S−K, 0)` for call, `max(K−S, 0)` for put |
| `sigma <= 0` | Raises `ValueError("sigma must be positive")` |
| `K <= 0` | Raises `ValueError("K must be positive")` |

### 4.2 Option Greeks — `bs_greeks`

Greeks measure the sensitivity of the option price to changes in each input parameter. All are derived analytically from the same `d1`, `d2` as above. `N'(·)` is the standard normal PDF (`scipy.stats.norm.pdf`).

| Greek | Formula | Meaning |
|-------|---------|---------|
| Delta (call) | `N(d1)` | ∂Price/∂S — rate of change w.r.t. underlying |
| Delta (put) | `N(d1) − 1` | Always negative for long puts |
| Gamma | `N'(d1) / (S·σ·√T)` | ∂²Price/∂S² — curvature; identical for call and put |
| Theta (call) | `−S·N'(d1)·σ/(2√T) − r·K·e^(−rT)·N(d2)` | ∂Price/∂T — time decay per year |
| Vega | `S·N'(d1)·√T` | ∂Price/∂σ — sensitivity to volatility |

> **Note:** `bs_greeks` requires `T > 0`. At expiry the Greeks are discontinuous and not defined.

### 4.3 Historical Volatility — `historical_vol`

Annualized realized volatility from log-returns over a rolling window:

```
log_returns = ln(P_t / P_{t-1})
daily_vol   = rolling_std(log_returns, window=30)
annual_vol  = daily_vol × √252            # 252 trading days / year
```

Fetches 1 year of daily closing prices via `yf.Ticker(ticker).history(period="1y")` and returns the most recent rolling window value.

### 4.4 Implied Volatility Solver — `implied_vol`

Newton-Raphson iteration on the BS pricing function:

```
σ_{n+1} = σ_n − (BS(σ_n) − market_price) / Vega(σ_n)
```

**Algorithm:**

```
σ_0 = 0.3  (initial guess)
loop up to max_iter = 100:
    compute BS price and vega at σ_n
    if |BS(σ_n) − market_price| < tol=1e-6 → converged, return σ_n
    if vega < 1e-10 → vega vanishes, return nan
    if σ_{n+1} ≤ 0  → stepped out of domain, return nan
return nan  (max iterations exceeded without convergence)
```

**Convergence guarantees:** Newton-Raphson converges quadratically near the root but diverges for deep OTM near-expiry options (vega → 0) and for market prices below intrinsic value (no real solution exists).

### 4.5 Pricing Table — `pricing_table`

Vectorises `bs_price` and `bs_greeks` across a list of strikes and returns a `pd.DataFrame`:

| Column | Description |
|--------|-------------|
| `Strike` | Input strike price |
| `Call Price` | BS call price |
| `Put Price` | BS put price |
| `Delta(C)` | Call delta |
| `Gamma` | Gamma (same for call and put) |
| `Vega` | Vega (same for call and put) |

---

## 5. Concurrency Model

### 5.1 Synchronous by Design

This project is intentionally **single-threaded and synchronous**. The computational core — `bs_price`, `bs_greeks`, `implied_vol`, `pricing_table` — consists of pure mathematical functions. They complete in microseconds, have no shared state, and are safe to call from any context.

The only blocking operation is the `yfinance` HTTP fetch inside `historical_vol` and `main()`. Since the demo fetches one ticker once at startup, the latency of spinning up an event loop would exceed the latency of the HTTP call itself.

### 5.2 When You Would Need Concurrency

| Scenario | Appropriate model |
|----------|-------------------|
| Pricing options across 1 000+ tickers simultaneously | `asyncio` + `aiohttp` for concurrent fetches |
| Real-time streaming prices via WebSocket | `asyncio` event loop |
| Monte Carlo simulation (CPU-intensive) | `multiprocessing` or vectorised NumPy |
| Live options scanner across all S&P 500 names | `asyncio` + thread pool for CPU work |

---

## 6. Async Bridge

### 6.1 The Blocking Problem

`yfinance` uses `requests` internally — it is a blocking library. Integrating it into an `asyncio` application (e.g., a FastAPI service or a streaming dashboard) requires a bridge pattern.

### 6.2 The Bridge Solution

Run the blocking call in a thread pool, leaving the event loop free:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bs_calculator import historical_vol, pricing_table

_executor = ThreadPoolExecutor(max_workers=4)

async def async_historical_vol(ticker: str, window: int = 30) -> float:
    loop = asyncio.get_event_loop()
    # Offload the blocking yfinance call to a thread pool
    return await loop.run_in_executor(_executor, historical_vol, ticker, window)

async def async_price(ticker: str, strikes: list, T: float, r: float):
    import yfinance as yf
    loop = asyncio.get_event_loop()
    sigma = await async_historical_vol(ticker)
    hist  = await loop.run_in_executor(_executor, lambda: yf.Ticker(ticker).history(period="5d"))
    S     = float(hist["Close"].iloc[-1])
    # bs_price and bs_greeks are pure math — no bridge needed
    return pricing_table(S, strikes, T, r, sigma)
```

**Key principle:** The pure math functions (`bs_price`, `bs_greeks`, `implied_vol`, `pricing_table`) require no adaptation — they are stateless and non-blocking. Only `yfinance` calls need to be wrapped.

### 6.3 Alternative: Replace yfinance Entirely

For production use, replace `yfinance` with an async-native data source (Polygon.io, Alpaca, or a broker WebSocket feed) and the bridge becomes unnecessary.

---

## 7. Comparison with Alternative Approaches

### 7.1 Black-Scholes vs Other Pricing Models

| Model | Strengths | Weaknesses | Best for |
|-------|-----------|------------|----------|
| **Black-Scholes (this project)** | Closed-form, instant, analytically tractable Greeks | Constant σ, log-normal, no dividends, European only | Quick theoretical baseline |
| **Binomial Tree (CRR)** | American exercise, discrete dividends | O(n²), slower | American options |
| **Monte Carlo** | Path-dependent payoffs (Asian, barrier) | Slow, statistical noise | Exotic options |
| **Heston stochastic vol** | Captures volatility smile endogenously | Complex calibration | When IV skew matters |

### 7.2 IV Solver: Newton-Raphson vs Alternatives

| Method | Convergence | Robustness | Notes |
|--------|------------|------------|-------|
| **Newton-Raphson (this project)** | Quadratic near root | Fails when vega → 0 | Fast for reasonable σ₀ |
| **Brent's method** | Superlinear, bracketed | Guaranteed on bracketed interval | Safer for edge cases |
| **Bisection** | Linear | Guaranteed | Very slow |
| **Jaeckel (2015)** | Direct formula, no iteration | Near-perfect | Industry gold-standard |

### 7.3 Historical Volatility: Rolling Window vs Alternatives

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Rolling std (this project)** | Equal-weight 30-day window | Transparent, intuitive | Slow to react to shocks |
| **EWMA (RiskMetrics)** | Exponentially weighted | Faster shock response | Requires λ tuning |
| **GARCH(1,1)** | Models vol clustering | Statistically rigorous | Requires calibration |
| **Parkinson estimator** | Uses high-low range | More efficient | Requires OHLC data |

---

## 8. Limitations

**Mathematical assumptions violated in practice:**

1. **Constant volatility** — BS assumes σ is constant over `[0, T]`. In reality, IV varies by strike (*volatility smile/skew*) and by expiry (*term structure*). The differing IV values in the market comparison output are direct evidence of this.
2. **Log-normal prices** — BS assumes `ln(S_T/S_0)` is normally distributed. Real returns exhibit fat tails and negative skew — OTM puts are systematically underpriced by BS.
3. **Continuous trading, no transaction costs** — Perfect delta-hedging is impossible with discrete rebalancing and bid-ask spreads.
4. **Constant risk-free rate** — `r = 0.05` is hardcoded. In practice, the appropriate rate depends on the tenor matching `T`.
5. **No dividends** — BS does not account for discrete dividend payments. For AAPL (a dividend payer), BS will systematically overprice calls.
6. **European exercise only** — AAPL options are American-style (can be exercised early). This matters most for deep ITM puts.

**Implementation limitations:**

7. **IV solver divergence near expiry** — When options expire in 1–2 days, vega → 0 and Newton-Raphson cannot converge. Near-expiry strikes return `N/A` in the market comparison.
8. **yfinance `lastPrice` is stale** — Last trade price may not reflect current mid-market. Accurate IV requires current bid/ask midpoints.
9. **Single expiry comparison** — `main()` only compares against the nearest expiry. A full volatility surface requires all available expirations.

---

## 9. Future Work

| Priority | Enhancement | Complexity |
|----------|------------|------------|
| High | Brent's method IV solver for robustness at extreme strikes | Low |
| High | Dividend-adjusted BS (Merton 1973) — add continuous yield `q` | Low |
| High | Volatility surface — loop over all expirations to build IV(K, T) | Low |
| Medium | IV smile plot with `matplotlib` | Low |
| Medium | GARCH volatility via `arch` library | Medium |
| Medium | American option pricing via Binomial Tree (CRR) | Medium |
| Medium | Greeks P&L attribution — daily theta decay and delta-hedge tracking | Medium |
| Low | FastAPI endpoint wrapping `bs_price` and `pricing_table` | Medium |
| Low | `argparse` CLI for arbitrary ticker/strike/expiry | Low |
| Low | Heston stochastic volatility model | High |

---

## 10. Project Structure

```
BS Calculator/
│
├── bs_calculator.py        # All functions + live demo
├── test_bs_calculator.py   # pytest test suite — 39 tests across 5 classes
├── requirements.txt        # Python dependencies
└── README.md
```

**`bs_calculator.py` function index:**

| Function | Signature | Purpose |
|----------|-----------|---------|
| `bs_price` | `(S, K, T, r, sigma, option_type) → float` | BS call / put price |
| `bs_greeks` | `(S, K, T, r, sigma) → dict` | Delta, Gamma, Theta, Vega |
| `historical_vol` | `(ticker, window=30) → float` | 30-day annualized vol via yfinance |
| `implied_vol` | `(market_price, S, K, T, r, option_type) → float` | Newton-Raphson IV solver |
| `pricing_table` | `(S, K_range, T, r, sigma) → DataFrame` | Multi-strike pricing grid |
| `main` | `() → None` | Live AAPL demo |

**`test_bs_calculator.py` test class index:**

| Class | Tests | What is verified |
|-------|-------|-----------------|
| `TestBsPrice` | 13 | ATM values, deep ITM/OTM, T=0 edge cases, invalid inputs, put-call parity |
| `TestBsGreeks` | 9 | Output keys, delta bounds and relationship, gamma/vega positivity, numerical vega, theta sign |
| `TestHistoricalVol` | 4 | Return type, positivity, annualisation scale, yfinance call target |
| `TestImpliedVol` | 5 | Round-trip call/put σ recovery, parity of IV, invalid price, return type |
| `TestPricingTable` | 8 | DataFrame type, columns, row count, strike values, call/put monotonicity, delta/gamma bounds |

---

## 11. Quickstart

**Prerequisites:** Python 3.10+, internet connection (for yfinance).

```bash
# Install dependencies
pip install -r requirements.txt

# Run the live AAPL demo
python3 bs_calculator.py
```

Expected output (values vary with live market data):

```
Underlying: AAPL | S=308.82 | σ=0.2210 | r=0.05 | T=30 days

  Strike    Call Price    Put Price    Delta(C)    Gamma     Vega
--------  ------------  -----------  ----------  -------  -------
300.0000       13.7253       3.6750      0.7102   0.0175  30.2972
305.0000       10.5380       5.4671      0.6152   0.0195  33.8366
310.0000        7.8485       7.7571      0.5145   0.0204  35.2974
315.0000        5.6619      10.5501      0.4144   0.0199  34.5047
320.0000        3.9519      13.8195      0.3210   0.0183  31.7044

--- Market vs BS Comparison (calls, expiry 2026-05-26) ---
  Strike    Market    BS Price    Diff  Impl Vol
--------  --------  ----------  ------  ----------
   300        9.68       13.73    4.05  N/A
   305        4.65       10.54    5.89  N/A
   310        1.18        7.85    6.67  0.0321
   315        0.20        5.66    5.46  0.0440
   320        0.05        3.95    3.90  0.0564
```

**Using as a library:**

```python
from bs_calculator import bs_price, bs_greeks, implied_vol, pricing_table

# Price a 30-day ATM call
price = bs_price(S=200, K=200, T=30/365, r=0.05, sigma=0.25, option_type='call')
print(f"Call price: ${price:.2f}")

# Greeks
g = bs_greeks(S=200, K=200, T=30/365, r=0.05, sigma=0.25)
print(f"Delta: {g['delta_call']:.4f} | Gamma: {g['gamma']:.4f} | Vega: {g['vega']:.4f}")

# Implied volatility from a market price
iv = implied_vol(market_price=7.50, S=200, K=200, T=30/365, r=0.05, option_type='call')
print(f"Implied vol: {iv:.4f}")
```

---

## 12. Running Tests

```bash
python3 -m pytest test_bs_calculator.py -v
```

Expected output:

```
platform darwin -- Python 3.14.2, pytest-8.3.3
collected 39 items

test_bs_calculator.py::TestBsPrice::test_call_atm                        PASSED
test_bs_calculator.py::TestBsPrice::test_put_atm                         PASSED
test_bs_calculator.py::TestBsPrice::test_put_call_parity                 PASSED
...
test_bs_calculator.py::TestPricingTable::test_gamma_positive             PASSED

39 passed in 1.04s
```

### Test Descriptions

| Test class | Key tests | What is verified |
|------------|-----------|-----------------|
| `TestBsPrice` | `test_call_atm`, `test_put_call_parity`, `test_t_zero_call_itm` | Known ATM value (≈10.45), C−P = S−Ke^(−rT), intrinsic at expiry |
| `TestBsGreeks` | `test_vega_numerical_check`, `test_theta_call_negative` | Analytical vega matches finite-difference within 0.01; theta is negative (time decay) |
| `TestHistoricalVol` | `test_vol_annualized_scale` | Result is in annualized range (0.05–1.0), not daily scale (~0.01) |
| `TestImpliedVol` | `test_recovers_known_sigma`, `test_impossible_price_returns_nan` | Round-trip σ recovery to 1e-5; graceful nan on invalid prices |
| `TestPricingTable` | `test_call_prices_decrease_with_strike` | Call prices are strictly decreasing as K increases |

---

## 13. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | ≥ 1.24 | Array operations, `log`, `sqrt`, `exp` |
| `scipy` | ≥ 1.10 | `norm.cdf` and `norm.pdf` for the standard normal distribution |
| `pandas` | ≥ 2.0 | `DataFrame` for pricing table; rolling std for historical vol |
| `yfinance` | ≥ 0.2 | Fetching historical prices and live option chains from Yahoo Finance |
| `tabulate` | ≥ 0.9 | Terminal-friendly table formatting in `main()` |
| `pytest` | ≥ 7.0 | Test runner (dev dependency) |

```bash
pip install -r requirements.txt
```

> **Note:** `yfinance` is a third-party scraper of Yahoo Finance data. It is not an official API. Data availability and format may change without notice.

---

## 14. References

1. Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
2. Merton, R.C. (1973). *Theory of Rational Option Pricing.* Bell Journal of Economics and Management Science, 4(1), 141–183.
3. Hull, J.C. (2022). *Options, Futures, and Other Derivatives* (11th ed.). Pearson.
4. Jaeckel, P. (2015). *Let's Be Rational.* Wilmott Magazine. — State-of-the-art direct IV computation without iteration.
5. Brenner, M. & Subrahmanyam, M.G. (1988). *A Simple Formula to Compute the Implied Standard Deviation.* Financial Analysts Journal, 44(5), 80–83.
6. Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity.* Econometrica, 50(4), 987–1007.
7. Python Software Foundation. (2024). *`scipy.stats.norm` documentation.* scipy.org.
8. Ranaroussi, R. (2024). *yfinance documentation.* github.com/ranaroussi/yfinance.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
