"""
Practical least-squares and curve-fit helpers that operate directly on a
*Pandas* DataFrame with two required columns "x" and "y".

    1. Ordinary-least-squares polynomial regression (analytical normal equations)
    2. Non-linear curve-fitting with SciPy’s Levenberg–Marquardt optimiser
       (example model: shifted exponential)
"""



from __future__ import annotations
from typing import Tuple, Callable

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import curve_fit


#   Linear least-squares (polynomial regression)
def _design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Vandermonde matrix with columns [1, x, x², …, x^degree]."""
    return np.vander(x, N=degree + 1, increasing=True)


def poly_fit(df: pd.DataFrame, degree: int = 1) -> Tuple[np.ndarray, float]:
    """
    Fit y ≈ Σ β_k x^k via ordinary least squares.

    Parameters
    df      : DataFrame with columns ``"x"`` and ``"y"``.
    degree  : Highest polynomial power.

    Returns
    coeffs  : β coefficients, low-order first.
    rmse    : Root-mean-square error on the training data.
    """
    X = _design_matrix(df["x"].to_numpy(float), degree)
    y = df["y"].to_numpy(float)
    coeffs, *_ = lstsq(X, y, rcond=None)
    preds = X @ coeffs
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    return coeffs, rmse


#   Non-linear curve-fit (shifted exponential)
def _exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """a · exp(b x) + c."""
    return a * np.exp(b * x) + c


def exp_fit(
    df: pd.DataFrame,
    model: Callable = _exp_model,
    p0: Tuple[float, float, float] | None = None,
) -> Tuple[np.ndarray, float]:
    """
    Non-linear least-squares using SciPy’s ``curve_fit``.

    Parameters
    df    : DataFrame with ``"x"`` and ``"y"``.
    model : Callable model function f(x, *params).
    p0    : Optional initial parameter guess.

    Returns
    params : Optimised model parameters.
    rmse   : Root-mean-square error on the training data.
    """
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    params, _ = curve_fit(model, x, y, p0=p0)
    preds = model(x, *params)
    rmse = np.sqrt(np.mean((y - preds) ** 2))
    return params, rmse


#   CLI entry point 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Polynomial and exponential curve-fit demo."
    )
    parser.add_argument("csv", help="CSV file with columns x,y")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree")
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    beta, rmse_lin = poly_fit(data, degree=args.degree)
    print(f"OLS β = {beta}   RMSE = {rmse_lin:.4f}")

    params, rmse_exp = exp_fit(data)
    print(f"Exp params = {params}   RMSE = {rmse_exp:.4f}")
