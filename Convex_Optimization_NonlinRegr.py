import numpy as np
import matplotlib.pyplot as plt

def generate_data(N=200, seed=101):
    """
    Generate synthetic nonlinear data using known exponential model:
        y = exp(√3 * t) + exp(√7 * u)

    Parameters:
        N (int): Number of data points
        seed (int): Seed for pseudo-random number generator

    Returns:
        Amat (np.ndarray): Nx2 matrix of input values [t, u]
        bvect (np.ndarray): N-length array of response values
    """
    bvect = np.zeros(N)
    C1 = 16807
    M1 = 2**31 - 1
    t1 = seed * 1007
    Amat = np.zeros((N, 2))

    for j in range(N):
        for k in range(2):
            t1 = (C1 * t1) % M1
            Amat[j, k] = t1 / M1

    for j in range(N):
        t, u = Amat[j]
        bvect[j] = np.exp(np.sqrt(3) * t) + np.exp(np.sqrt(7) * u)

    return Amat, bvect

def third_order_newton(Amat, bvect, initial_guess, alpha=0.25, max_iter=10):
    """
    Apply the third-order Newton-type method to solve a nonlinear regression problem.

    Parameters:
        Amat (np.ndarray): Nx2 array of input features [t, u]
        bvect (np.ndarray): N-length array of observed outputs
        initial_guess (list or np.ndarray): Initial estimate for [x1, x2]
        alpha (float): Damping factor to control update step size
        max_iter (int): Number of iterations to perform

    Returns:
        xvect (np.ndarray): Estimated parameters [x1, x2] after iterations
    """
    xvect = np.array(initial_guess, dtype=np.float64)
    N = len(bvect)

    for iter in range(max_iter):
        x1, x2 = xvect

        # Compute F_x (gradient)
        s1, s2 = 0.0, 0.0
        for j in range(N):
            y = bvect[j]
            t, u = Amat[j]
            try:
                e1 = np.exp(x1 * t)
                e2 = np.exp(x2 * u)
                res = y - e1 - e2
                s1 += res * e1 * t
                s2 += res * e2 * u
            except OverflowError:
                print("Overflow in gradient at iteration", iter + 1)
                return xvect
        Fvect_x = np.array([-s1, -s2])

        # Compute Jacobian (Hessian approximation)
        h11, h12, h22 = 0.0, 0.0, 0.0
        for j in range(N):
            t, u = Amat[j]
            try:
                e1 = np.exp(x1 * t)
                e2 = np.exp(x2 * u)
                h11 += 2 * (e1 ** 2) * (t ** 2)
                h12 += 2 * e1 * e2 * t * u
                h22 += 2 * (e2 ** 2) * (u ** 2)
            except OverflowError:
                print("Overflow in Jacobian at iteration", iter + 1)
                return xvect
        F_prime_x = np.array([[h11, h12], [h12, h22]])

        # Compute intermediate point yvect
        try:
            yvect = xvect - np.linalg.solve(F_prime_x, Fvect_x)
        except np.linalg.LinAlgError:
            print("Singular matrix encountered at iteration", iter + 1)
            return xvect

        y1, y2 = yvect

        # Compute F_y (gradient at intermediate yvect)
        t1, t2 = 0.0, 0.0
        for j in range(N):
            y = bvect[j]
            t, u = Amat[j]
            try:
                e1 = np.exp(y1 * t)
                e2 = np.exp(y2 * u)
                r = y - e1 - e2
                t1 += r * e1 * t
                t2 += r * e2 * u
            except OverflowError:
                print("Overflow in second gradient at iteration", iter + 1)
                return xvect
        Fvect_y = np.array([-t1, -t2])

        # Final third-order update with damping
        try:
            update = alpha * np.linalg.solve(F_prime_x, Fvect_x + Fvect_y)
            xvect = xvect - update
        except np.linalg.LinAlgError:
            print("Singular matrix in update step at iteration", iter + 1)
            return xvect

        print(f"at iteration {iter + 1}, we have [x1, x2] = {np.round(xvect, 10)}")

        if np.isnan(xvect).any():
            print("NaN encountered — stopping iteration.")
            break

    return xvect

def main():
    """
    Run and visualize the third-order Newton regression with synthetic data.
    """
    # Generate data
    Amat, bvect = generate_data()

    # Visualize
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Amat[:, 0], Amat[:, 1], bvect, s=10)
    plt.title("Synthetic Data (exp(√3 t) + exp(√7 u))")
    plt.show()

    # Run solver
    final_x = third_order_newton(
        Amat,
        bvect,
        initial_guess=[1.5, 2.0],
        alpha=0.25,
        max_iter=10
    )

    print("\nFinal result after iterations:", np.round(final_x, 10))

if __name__ == "__main__":
    main()
