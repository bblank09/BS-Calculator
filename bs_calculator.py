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
    raise NotImplementedError


def historical_vol(ticker: str, window: int = 30) -> float:
    raise NotImplementedError


def implied_vol(market_price: float, S: float, K: float, T: float, r: float,
                option_type: str = 'call', tol: float = 1e-6, max_iter: int = 100):
    raise NotImplementedError


def pricing_table(S: float, K_range: list, T: float, r: float, sigma: float) -> pd.DataFrame:
    raise NotImplementedError
