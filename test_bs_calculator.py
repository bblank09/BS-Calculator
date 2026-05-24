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
