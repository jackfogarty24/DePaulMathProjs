"""
monte_carlo_random.py
=====================
Showcases a *home-grown* linear-congruential generator (LCG) and two classic
Monte-Carlo problems:

1.  Estimate the area sandwiched between y = sin x and y = cos x on [0, π/2].
2.  Estimate the volume of the unit sphere in ℝᵈ by rejection sampling.

The point is to illustrate how numerical analysts roll their own RNG,
validate it loosely, and then feed it into statistical experiments.
"""
from __future__ import annotations
from typing import Callable

import math
import random
import numpy as np


# Linear Congruential Generator
class LinearCongruentialGenerator:
    """
    “Minimal Standard” LCG (Park & Miller, 1988).

    X_{n+1} = (a · X_n) mod m,   with  a = 48 271  and  m = 2³¹ − 1.

    The choice of parameters gives a full period for 32-bit arithmetic,
    yet is simple enough to verify analytically—handy for teaching or
    for environments where you cannot trust the system RNG.
    """

    _a = 48_271
    _m = 2**31 - 1

    def __init__(self, seed: int | None = None):
        self._state = seed or random.randrange(1, self._m)

    def random(self) -> float:
        """Return a single U(0, 1) variate."""
        self._state = (self._a * self._state) % self._m
        return self._state / self._m

    def random_vec(self, size: int) -> np.ndarray:
        """Vectorised version that returns *size* i.i.d. uniforms."""
        out = np.empty(size)
        for i in range(size):
            out[i] = self.random()
        return out


# Monte-Carlo utilities
def estimate_area_between_curves(
    f_top: Callable[[float], float],
    f_bottom: Callable[[float], float],
    x_min: float,
    x_max: float,
    n_samples: int = 200_000,
    rng: LinearCongruentialGenerator | None = None,
) -> float:
    """
    Plain-vanilla Monte-Carlo area estimator using *rejection sampling*
    between two curves on a closed interval.
    """
    rng = rng or LinearCongruentialGenerator()
    xs = rng.random_vec(n_samples) * (x_max - x_min) + x_min
    ys = rng.random_vec(n_samples)

    # Rescale Y so 0 ≤ y ≤ f_top − f_bottom.
    f_range = f_top(xs) - f_bottom(xs)
    hits = ys < f_range / f_range.max()
    return hits.mean() * (x_max - x_min) * f_range.max()


def estimate_unit_sphere_volume(
    dim: int = 3,
    n_samples: int = 1_000_000,
    rng: LinearCongruentialGenerator | None = None,
) -> float:
    """
    Volume of the unit ball via rejection sampling from the surrounding cube.
    """
    rng = rng or LinearCongruentialGenerator()
    count = 0
    for _ in range(n_samples):
        point = rng.random_vec(dim) * 2 - 1  # sample uniformly from [−1,1]^d
        if np.dot(point, point) <= 1.0:      # keep if inside the sphere
            count += 1
    cube_vol = 2**dim
    return cube_vol * count / n_samples


# Demonstration
def main() -> None:
    rng = LinearCongruentialGenerator(seed=20250615)

    area = estimate_area_between_curves(
        np.sin,
        np.cos,
        0.0,
        math.pi / 2,
        n_samples=200_000,
        rng=rng,
    )
    print(f"Estimated area between sin x and cos x on [0, π/2]: {area:.6f}")

    vol3 = estimate_unit_sphere_volume(dim=3, n_samples=500_000, rng=rng)
    print(f"Estimated volume of the 3-D unit sphere: {vol3:.6f}  "
          f"(exact: 4/3·π ≈ {4/3*math.pi:.6f})")


if __name__ == "__main__":
    main()
