import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from tabulate import tabulate


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes price for European option."""
    if K <= 0:
        raise ValueError("K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    if T == 0:
        if option_type == 'call':
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Returns delta, gamma, theta, vega for call and put."""
    if K <= 0:
        raise ValueError("K must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if T <= 0:
        raise ValueError("T must be positive for Greeks")

    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    n_prime_d1 = norm.pdf(d1)
    sqrt_T = np.sqrt(T)

    delta_call = norm.cdf(d1)
    delta_put = delta_call - 1.0
    gamma = n_prime_d1 / (S * sigma * sqrt_T)
    theta_call = (
        -S * n_prime_d1 * sigma / (2 * sqrt_T)
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    vega = S * n_prime_d1 * sqrt_T

    return {
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'theta_call': theta_call,
        'vega': vega,
    }


def historical_vol(ticker: str, window: int = 30) -> float:
    """Fetch price data via yfinance, compute annualized volatility."""
    tk = yf.Ticker(ticker)
    hist = tk.history(period='1y')
    closes = hist['Close']
    log_returns = np.log(closes / closes.shift(1)).dropna()
    daily_vol = log_returns.rolling(window).std().dropna()
    return float(daily_vol.iloc[-1] * np.sqrt(252))


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = 'call', tol: float = 1e-6, max_iter: int = 100):
    """Newton-Raphson implied volatility solver. Returns float or np.nan if non-convergent."""
    if market_price <= 0:
        return np.nan

    sigma = 0.3  # initial guess
    try:
        for _ in range(max_iter):
            price = bs_price(S, K, T, r, sigma, option_type)
            d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma
            if vega < 1e-10:
                return np.nan
            sigma -= diff / vega
            if sigma <= 0:
                return np.nan
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return np.nan

    return np.nan


def pricing_table(S: float, K_range: list, T: float, r: float, sigma: float) -> pd.DataFrame:
    raise NotImplementedError
