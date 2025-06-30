"""
broyden_solver.py
-----------------
Self-contained implementation of Broyden’s rank-one quasi-Newton method
for solving nonlinear systems F(x) = 0.  Comes with a short demo using
the two-dimensional Rosenbrock system whose root is at (1, 1).

The algorithm starts from an identity approximation to the Jacobian,
updates that matrix on each iteration with a rank-one correction, and
terminates when both the step size and the residual norm fall below a
user-supplied tolerance.

Usage
Run the file directly to see the demo:

    python broyden_solver.py

or import the ``broyden`` function from another script.

"""
from __future__ import annotations
from typing import Callable, Tuple, Dict
import numpy as np


def broyden(
    f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 100,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Solve the system F(x) = 0 with Broyden’s rank-one update.

    Parameters
    f
        Function mapping ℝⁿ → ℝⁿ whose root is sought.
    x0
        Initial guess (length-n NumPy array).
    tol
        Stop when both the residual norm and the step norm are below this value.
    max_iter
        Hard cap on iterations to guard against non-convergence.
    verbose
        If True, print residual and step norms every iteration.

    Returns
    x
        Approximate root.
    info
        Dictionary with keys ``n_iter`` and ``converged``.
    """
    x = np.asarray(x0, dtype=float)
    n = x.size
    B = np.eye(n)               # initial Jacobian approximation
    fx = f(x)

    for k in range(1, max_iter + 1):
        # Solve B s = −F.  If B is singular the algorithm cannot proceed.
        try:
            s = np.linalg.solve(B, -fx)
        except np.linalg.LinAlgError as err:
            raise RuntimeError("Jacobian approximation became singular.") from err

        x_new = x + s
        fx_new = f(x_new)

        if verbose:
            print(f"[{k:3}] ‖F‖₂ = {np.linalg.norm(fx_new):.3e}   "
                  f"‖Δx‖₂ = {np.linalg.norm(s):.3e}")

        # Convergence test.
        if np.linalg.norm(fx_new) < tol and np.linalg.norm(s) < tol:
            return x_new, {"n_iter": k, "converged": True}

        # Rank-one Broyden update:  B ← B + (ΔF − B Δx) Δxᵀ / (Δxᵀ Δx)
        delta_f = fx_new - fx
        delta_x = s
        denom = delta_x @ delta_x
        if denom == 0.0:
            raise RuntimeError("Step collapsed to zero, cannot update Jacobian.")
        B += np.outer(delta_f - B @ delta_x, delta_x) / denom

        # Prepare for next iteration.
        x, fx = x_new, fx_new

    # Fell through loop ⇒ no convergence.
    return x, {"n_iter": max_iter, "converged": False}

# Demonstarte 
def _rosenbrock_system(x: np.ndarray) -> np.ndarray:
    """
    Two-variable Rosenbrock function written as a root-finding problem:

        F₁(x, y) = 10 (y − x²)
        F₂(x, y) = 1 − x
    """
    return np.array([10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]])


if __name__ == "__main__":
    approx_root, stats = broyden(_rosenbrock_system, np.array([-1.2, 1.0]), verbose=True)
    outcome = "converged" if stats["converged"] else "failed"
    print(f"\nBroyden {outcome} in {stats['n_iter']} iterations: root ≈ {approx_root}")
