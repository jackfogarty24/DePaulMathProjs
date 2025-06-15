"""

Lightweight cubic-spline interpolation that works straight from a
Pandas DataFrame containing columns x and y. Use when
you need a smooth, twice-differentiable curve for feature engineering,
data imputation, or signal smoothing without pulling in the full weight of
SciPy.

Features:
    Natural, clamped, or “not-a-knot” boundary conditions.
    Returns a small *CubicSpline* object whose ``__call__`` evaluates the
        spline at arbitrary points—handy in scikit-learn pipelines.
    Optional helper to plot the fitted curve against the raw data.
"""

from __future__ import annotations
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#   Core solver
def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for a tridiagonal system Ax = d where
    *a* is sub-diag, *b* main diag, *c* super-diag.
    """
    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[-1] = (d[-1] - a[-1] * dp[-2]) / (b[-1] - a[-1] * cp[-2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


#   Public API
class CubicSpline:
    """
    Simple container for the piece-wise coefficients.

    The polynomial on interval i is

        S_i(x) = a_i + b_i (x - x_i) + c_i (x - x_i)² + d_i (x - x_i)³
    """

    def __init__(
        self,
        x: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        d: np.ndarray,
    ):
        self.x = x
        self.a, self.b, self.c, self.d = a, b, c, d

    # Vectorised evaluation; falls back to float for scalar input.
    def __call__(self, x_new: np.ndarray | float) -> np.ndarray | float:
        xi = self.x
        idx = np.searchsorted(xi[:-1], x_new, side="right") - 1
        idx = np.clip(idx, 0, len(xi) - 2)
        dx = x_new - xi[idx]
        return self.a[idx] + dx * (
            self.b[idx] + dx * (self.c[idx] + dx * self.d[idx])
        )


def fit_spline(
    df: pd.DataFrame,
    bc_type: Literal["natural", "clamped", "not-a-knot"] = "natural",
    fp0: float = 0.0,
    fpn: float = 0.0,
) -> CubicSpline:
    """
    Fit a cubic spline to df["x"] / df["y"].

    Parameters
    df       : DataFrame with monotone ``"x"``.
    bc_type  : Boundary condition.  *natural* sets second derivative = 0
               at the ends; *clamped* fixes the first derivative to fp0/fpn;
               *not-a-knot* enforces third-derivative continuity at x₁, xₙ₋₁.
    fp0, fpn : First derivatives for clamped ends (ignored otherwise).

    Returns
    spline   : *CubicSpline* object.
    """
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)
    n = len(x) - 1
    h = np.diff(x)

    # Assemble tridiagonal system for c (second-derivative terms).
    a = h[:-1]
    b = 2 * (h[:-1] + h[1:])
    c = h[1:]
    d = 6 * (np.diff(y[1:]) / h[1:] - np.diff(y[:-1]) / h[:-1])

    if bc_type == "natural":
        # Pad with zeros for natural ends.
        a = np.hstack(([0], a))
        b = np.hstack(([1], b, [1]))
        c = np.hstack((c, [0]))
        d = np.hstack(([0], d, [0]))
    elif bc_type == "clamped":
        # Incorporate first-derivative constraints.
        a = np.hstack(([0], a))
        c = np.hstack((c, [0]))
        b = np.hstack(([2 * h[0]], b, [2 * h[-1]]))
        d = np.hstack(
            (
                [6 * ((y[1] - y[0]) / h[0] - fp0)],
                d,
                [6 * (fpn - (y[-1] - y[-2]) / h[-1])],
            )
        )
    elif bc_type == "not-a-knot":
        # Shrink system by two—imposes continuity of third derivative.
        b[0] += h[0]
        b[-1] += h[-1]
    else:
        raise ValueError("Unknown bc_type.")

    c_coeff = _tridiag_solve(a, b, c, d)

    # Recover spline coefficients.
    a_coeff = y[:-1]
    b_coeff = (y[1:] - y[:-1]) / h - (2 * c_coeff[:-1] + c_coeff[1:]) * h / 6
    d_coeff = np.diff(c_coeff) / (6 * h)
    c_coeff = c_coeff[:-1] / 2

    return CubicSpline(x, a_coeff, b_coeff, c_coeff, d_coeff)


def plot_spline(
    df: pd.DataFrame,
    spline: CubicSpline,
    num: int = 300,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Quick matplotlib visual check.

    Parameters
    ----------
    df      : Original data.
    spline  : *CubicSpline* returned by *fit_spline*.
    num     : Number of points for the smooth curve.
    ax      : Existing Axes to draw on (optional).
    """
    ax = ax or plt.gca()
    xs = np.linspace(df.x.min(), df.x.max(), num)
    ax.plot(df.x, df.y, "o", label="data", alpha=0.7)
    ax.plot(xs, spline(xs), "-", label="cubic spline")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    return ax


#   Minimal CLI demo
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Cubic-spline demo.")
    p.add_argument("csv", help="CSV with x,y columns (monotone x).")
    p.add_argument("--bc", default="natural", choices=["natural", "clamped", "not-a-knot"])
    p.add_argument("--fp0", type=float, default=0.0, help="First derivative at left for clamped.")
    p.add_argument("--fpn", type=float, default=0.0, help="First derivative at right for clamped.")
    args = p.parse_args()

    data = pd.read_csv(args.csv)
    spline = fit_spline(data, bc_type=args.bc, fp0=args.fp0, fpn=args.fpn)
    plot_spline(data, spline)
    plt.title(f"Cubic spline ({args.bc})")
    plt.show()
