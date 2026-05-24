import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import pandas as pd
from bs_calculator import (
    bs_price, bs_greeks, historical_vol, implied_vol, pricing_table
)


class TestBsPrice:
    def test_call_atm(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.2, 'call')
        assert abs(price - 10.4506) < 0.01

    def test_put_atm(self):
        price = bs_price(100, 100, 1.0, 0.05, 0.2, 'put')
        assert abs(price - 5.5735) < 0.01

    def test_call_deep_itm(self):
        price = bs_price(150, 100, 1.0, 0.05, 0.2, 'call')
        assert price > 45

    def test_call_deep_otm(self):
        price = bs_price(50, 100, 1.0, 0.05, 0.2, 'call')
        assert price < 0.01

    def test_t_zero_call_itm(self):
        price = bs_price(110, 100, 0.0, 0.05, 0.2, 'call')
        assert price == pytest.approx(10.0)

    def test_t_zero_call_otm(self):
        price = bs_price(90, 100, 0.0, 0.05, 0.2, 'call')
        assert price == pytest.approx(0.0)

    def test_t_zero_put_itm(self):
        price = bs_price(90, 100, 0.0, 0.05, 0.2, 'put')
        assert price == pytest.approx(10.0)

    def test_t_zero_put_otm(self):
        price = bs_price(110, 100, 0.0, 0.05, 0.2, 'put')
        assert price == pytest.approx(0.0)

    def test_invalid_sigma_zero(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_price(100, 100, 1.0, 0.05, 0.0, 'call')

    def test_invalid_sigma_negative(self):
        with pytest.raises(ValueError, match="sigma"):
            bs_price(100, 100, 1.0, 0.05, -0.1, 'call')

    def test_invalid_strike_zero(self):
        with pytest.raises(ValueError, match="K"):
            bs_price(100, 0, 1.0, 0.05, 0.2, 'call')

    def test_invalid_strike_negative(self):
        with pytest.raises(ValueError, match="K"):
            bs_price(100, -50, 1.0, 0.05, 0.2, 'call')

    def test_put_call_parity(self):
        S, K, T, r, sigma = 100, 95, 0.5, 0.05, 0.25
        call = bs_price(S, K, T, r, sigma, 'call')
        put = bs_price(S, K, T, r, sigma, 'put')
        parity = S - K * np.exp(-r * T)
        assert abs((call - put) - parity) < 1e-6


class TestBsGreeks:
    def setup_method(self):
        self.S, self.K, self.T, self.r, self.sigma = 100, 100, 1.0, 0.05, 0.2

    def test_returns_all_greeks(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        for key in ['delta_call', 'delta_put', 'gamma', 'theta_call', 'vega']:
            assert key in g

    def test_delta_call_range(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert 0 < g['delta_call'] < 1

    def test_delta_put_range(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert -1 < g['delta_put'] < 0

    def test_delta_call_plus_put_equals_one(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert abs(g['delta_call'] - g['delta_put'] - 1.0) < 1e-10

    def test_gamma_positive(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g['gamma'] > 0

    def test_vega_positive(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g['vega'] > 0

    def test_atm_delta_call_near_half(self):
        g = bs_greeks(100, 100, 1.0, 0.0, 0.2)
        assert abs(g['delta_call'] - 0.5) < 0.1

    def test_vega_numerical_check(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        eps = 1e-4
        price_up = bs_price(self.S, self.K, self.T, self.r, self.sigma + eps, 'call')
        price_dn = bs_price(self.S, self.K, self.T, self.r, self.sigma - eps, 'call')
        numerical_vega = (price_up - price_dn) / (2 * eps)
        assert abs(g['vega'] - numerical_vega) < 0.01

    def test_theta_call_negative(self):
        g = bs_greeks(self.S, self.K, self.T, self.r, self.sigma)
        assert g['theta_call'] < 0
