"""
gbm_simulator.py
================
Estimate the drift (μ) and volatility (σ) of a geometric Brownian motion from
historical equity prices, then simulate forward paths.

Why it matters
--------------
• Parameter estimation is done with **maximum-likelihood** on log-returns, a staple
  technique in risk analytics and time-series modelling.  
• Simulation relies on the *exact* GBM solution, giving users a fast Monte-Carlo
  engine that scales to thousands of paths in a single call.

Run it from the command line
----------------------------
$ python gbm_simulator.py AAPL --years 2 --paths 5000

Outputs μ, σ, and an interactive Matplotlib plot of 50 sample paths.
"""
from __future__ import annotations
from typing import Tuple

import argparse
import datetime as dt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf  # lightweight Yahoo Finance wrapper
except ImportError as exc:
    raise ImportError(
        "This script needs the 'yfinance' package.  Install with:\n    pip install yfinance"
    ) from exc


# Parameter estimation
def estimate_gbm_params(
    prices: pd.Series,
    dt_years: float = 1 / 252,
) -> Tuple[float, float]:
    """
    Maximum-likelihood estimates of μ and σ for GBM.

    Parameters
    ----------
    prices : pd.Series
        Chronologically ordered adjusted close prices.
    dt_years : float, optional
        Length of one time step in *years* (default ≈ one trading day).

    Returns
    -------
    mu_hat, sigma_hat
        Annualised drift and volatility.
    """
    log_ret = np.log(prices).diff().dropna()
    # μ̂ is the mean of log-returns plus half the variance, scaled to one year.
    mu_hat = log_ret.mean() / dt_years + 0.5 * log_ret.var() / dt_years
    # σ̂ is just the standard deviation of log-returns, annualised.
    sigma_hat = np.sqrt(log_ret.var() / dt_years)
    return float(mu_hat), float(sigma_hat)


# Simulation
def simulate_gbm_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_steps: int,
    dt_years: float,
    n_paths: int = 1_000,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Vectorised Monte-Carlo simulation of GBM using the analytical solution.

    Returns
    -------
    paths : ndarray
        Shape ``(n_steps + 1, n_paths)``;  first row is the initial price.
    """
    rng = np.random.default_rng(random_state)
    # Each increment is log-normally distributed; stack them cumulatively.
    increments = rng.normal(
        loc=(mu - 0.5 * sigma**2) * dt_years,
        scale=sigma * np.sqrt(dt_years),
        size=(n_steps, n_paths),
    )
    log_paths = np.vstack([np.zeros(n_paths), increments.cumsum(axis=0)])
    return s0 * np.exp(log_paths)


# Command-line entry point
def main() -> None:
    parser = argparse.ArgumentParser(
        description="GBM parameter estimation and Monte-Carlo simulation demo."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="^DJI",
        help="Yahoo Finance ticker symbol (default: Dow Jones index)",
    )
    parser.add_argument("--years", type=float, default=1.0, help="Forecast horizon in years")
    parser.add_argument("--paths", type=int, default=1000, help="Number of Monte-Carlo paths")
    args = parser.parse_args()

    # Pull five years of daily data.
    end = dt.date.today()
    start = end - dt.timedelta(days=5 * 365)
    prices = yf.download(args.ticker, start=start, end=end, progress=False)["Adj Close"].dropna()

    mu, sigma = estimate_gbm_params(prices)
    print(f"μ = {mu:.4f}   σ = {sigma:.4f}")

    n_steps = int(args.years * 252)
    paths = simulate_gbm_paths(prices.iloc[-1], mu, sigma, n_steps, 1 / 252, args.paths)

    # Plot a representative subset of paths for readability.
    plt.figure()
    plt.plot(paths[:, :50], linewidth=0.8)
    plt.title(f"{args.ticker}: {args.years:.1f}-year GBM forecast (50 sample paths)")
    plt.xlabel("Trading days ahead")
    plt.ylabel("Price")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
