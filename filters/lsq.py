import math

import numpy as np

from .helpers import pretty_str


class LeastSquaresFilter:
    """Implements a Least Squares recursive filter.

    Filter may be of order 0 to 2. Order 0 assumes the value being tracked is a constant, order 1 assumes that it moves in a line, and order 2 assumes that it is tracking a second order polynomial.

    References
    ----------
    Zarchan and Musoff. "Fundamentals of Kalman Filtering: A Practical Approach." Third Edition. AIAA, 2009.
    """

    def __init__(self, dt, order, noise_sigma=0.0):
        if order < 0 or order > 2:
            raise ValueError("order must be between 0 and 2")

        self.dt = dt

        self.sigma = noise_sigma
        self._order = order

        self.reset()

    def reset(self):
        """Reset filter back to state at time of construction"""

        self.n = 0  # nth step in the recursion
        self.x = np.zeros(self._order + 1)
        self.K = np.zeros(self._order + 1)
        self.y = 0  # residual

    def update(self, z):
        """Update filter with new measurement `z`"""

        self.n += 1
        # rename for readability
        n = self.n
        dt = self.dt
        x = self.x
        K = self.K
        y = self.y

        if self._order == 0:
            K[0] = 1.0 / n
            y = z - x
            x[0] += K[0] * y

        elif self._order == 1:
            K[0] = 2.0 * (2 * n - 1) / (n * (n + 1))
            K[1] = 6.0 / (n * (n + 1) * dt)

            y = z - x[0] - (dt * x[1])

            x[0] += (K[0] * y) + (dt * x[1])
            x[1] += K[1] * y

        else:
            den = n * (n + 1) * (n + 2)
            K[0] = 3.0 * (3 * n**2 - 3 * n + 2) / den
            K[1] = 18.0 * (2 * n - 1) / (den * dt)
            K[2] = 60.0 / (den * dt**2)

            y = z - x[0] - (dt * x[1]) - (0.5 * dt**2 * x[2])

            x[0] += (K[0] * y) + (x[1] * dt) + (0.5 * dt**2 * x[2])
            x[1] += (K[1] * y) + (x[2] * dt)
            x[2] += K[2] * y
        return self.x

    def errors(self):
        """Computes and returns the error and standard deviation of the
        filter at this time step."""

        n = self.n
        dt = self.dt
        order = self._order
        sigma = self.sigma

        error = np.zeros(order + 1)
        std = np.zeros(order + 1)

        if n == 0:
            return (error, std)

        if order == 0:
            error[0] = sigma / math.sqrt(n)
            std[0] = sigma / math.sqrt(n)

        elif order == 1:
            if n > 1:
                error[0] = sigma * math.sqrt(2 * (2 * n - 1) / (n * (n + 1)))
                error[1] = sigma * math.sqrt(12.0 / (n * (n * n - 1) * dt * dt))
            std[0] = sigma * math.sqrt((2 * (2 * n - 1)) / (n * (n + 1)))
            std[1] = (sigma / dt) * math.sqrt(12.0 / (n * (n * n - 1)))

        elif order == 2:
            dt2 = dt * dt

            if n >= 3:
                error[0] = sigma * math.sqrt(
                    3 * (3 * n * n - 3 * n + 2) / (n * (n + 1) * (n + 2))
                )
                error[1] = sigma * math.sqrt(
                    12
                    * (16 * n * n - 30 * n + 11)
                    / (n * (n * n - 1) * (n * n - 4) * dt2)
                )
                error[2] = sigma * math.sqrt(
                    720 / (n * (n * n - 1) * (n * n - 4) * dt2 * dt2)
                )

            std[0] = sigma * math.sqrt(
                (3 * (3 * n * n - 3 * n + 2)) / (n * (n + 1) * (n + 2))
            )
            std[1] = (sigma / dt) * math.sqrt(
                (12 * (16 * n * n - 30 * n + 11)) / (n * (n * n - 1) * (n * n - 4))
            )
            std[2] = (sigma / dt2) * math.sqrt(720 / (n * (n * n - 1) * (n * n - 4)))

        return error, std

    def __repr__(self):
        return "\n".join(
            [
                "LeastSquaresFilter object",
                pretty_str("dt", self.dt),
                pretty_str("sigma", self.sigma),
                pretty_str("_order", self._order),
                pretty_str("x", self.x),
                pretty_str("K", self.K),
            ]
        )
