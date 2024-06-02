import numpy as np

from .helpers import pretty_str


def print_steps(estims, preds):
    for ind in range(1, len(estims)):
        print(
            f"previous x: {estims[ind-1]:0.2f}, current x: {estims[ind]:0.2f}, predicted x̂: {preds[ind-1]:0.2f}"
        )


class GFilter:
    """Implements the g filter.

    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.

    Returns
    -------
    _type_
        _description_
    """

    def __init__(self, x0, dx, dt, g: float):
        self.x = x0
        self.dx = dx
        self.dt = dt
        self.g = g
        self.dx_pred = self.dx
        self.x_pred = self.x

        if np.ndim(x0) == 0:
            self.y = 0.0  # residual
            self.z = 0.0
        else:
            self.y = np.zeros(len(x0))
            self.z = np.zeros(len(x0))

    def update(self, z, g=None, h=None):
        """Performs the g filter predict and update step on the
        measurement z.

        Parameters
        ----------
        z : _type_
            _description_
        g : _type_, optional
            _description_, by default None
        h : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        if g is None:
            g = self.g

        # prediction step
        self.dx_pred = self.dx
        self.x_pred = self.x + (self.dx * self.dt)

        # update step
        self.y = z - self.x_pred
        self.dx = self.dx_pred + h * self.y / self.dt
        self.x = self.x_pred + g * self.y

        return (self.x, self.dx)

    def batch_filter(self, data, save_preds=False, saver=None):
        """Performs g-h filter with a fixed g and h.

        Parameters
        ----------
        data : _type_
            _description_
        save_preds : bool, optional
            _description_, by default False
        saver : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """

        x = self.x
        dx = self.dx
        n = len(data)

        estimates = np.zeros((n + 1, 2))
        estimates[0, 0] = x
        estimates[0, 1] = dx

        if save_preds:
            predictions = np.zeros(n)

        for i, z in enumerate(data):
            # prediction step
            x_est = x + (dx * self.dt)

            # update step
            residual = z - x_est
            x = x_est + self.g * residual

            estimates[i + 1, 0] = x
            estimates[i + 1, 1] = dx

            if save_preds:
                predictions[i] = x_est

            if saver is not None:
                saver.save()

        return (estimates, predictions) if save_preds else estimates

    def VRF_pred(self):
        """Returns the Variance Reduction Factor of the prediction step of the filter. The VRF is the normalized variance for the filter.

        References
        ----------
        Asquith, "Weight Selection in First Order Linear Filters"
        Report No RG-TR-69-12, U.S. Army Missile Command. Redstone Arsenal, Al. November 24, 1970.
        """

        g = self.g

        return (2 * g**2) / (g * (4 - 2 * g))

    def VRF(self):
        """Returns the Variance Reduction Factor (VRF) of the state variable of the filter (x) and its derivatives (dx, ddx). The VRF is the normalized variance for the filter."""

        g = self.g
        den = g * (4 - 2 * g)
        return (2 * g**2) / den

    def __repr__(self):
        return "\n".join(
            [
                "GHFilter object",
                pretty_str("dt", self.dt),
                pretty_str("g", self.g),
                pretty_str("x", self.x),
                pretty_str("dx", self.dx),
                pretty_str("x_pred", self.x_pred),
                pretty_str("dx_pred", self.dx_pred),
                pretty_str("y", self.y),
                pretty_str("z", self.z),
            ]
        )


class GHFilter:
    """Implements the g-h filter.

    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.
    """

    def __init__(self, x0, dx, dt, g, h):
        self.x = x0
        self.dx = dx
        self.dt = dt
        self.g = g
        self.h = h
        self.dx_pred = self.dx
        self.x_pred = self.x

        if np.ndim(x0) == 0:
            self.y = 0.0  # residual
            self.z = 0.0
        else:
            self.y = np.zeros(len(x0))
            self.z = np.zeros(len(x0))

    def update(self, z, g=None, h=None):
        """Performs the g-h filter predict and update step on the
        measurement z. Modifies the member variables listed below,
        and returns the state of x and dx as a tuple as a convenience.
        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h

        # prediction step
        self.dx_pred = self.dx
        self.x_pred = self.x + (self.dx * self.dt)

        # update step
        self.y = z - self.x_pred
        self.dx = self.dx_pred + h * self.y / self.dt
        self.x = self.x_pred + g * self.y

        return (self.x, self.dx)

    def batch_filter(self, data, save_preds=False, saver=None):
        """Performs g-h filter with a fixed g and h."""

        x = self.x
        dx = self.dx
        n = len(data)

        estimates = np.zeros((n + 1, 2))
        estimates[0, 0] = x
        estimates[0, 1] = dx

        if save_preds:
            predictions = np.zeros(n)

        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i, z in enumerate(data):
            # prediction step
            x_est = x + (dx * self.dt)

            # update step
            residual = z - x_est
            dx = dx + h_dt * residual  # i.e. dx = dx + h * residual / dt
            x = x_est + self.g * residual

            estimates[i + 1, 0] = x
            estimates[i + 1, 1] = dx

            if save_preds:
                predictions[i] = x_est

            if saver is not None:
                saver.save()

        return (estimates, predictions) if save_preds else estimates

    def VRF_pred(self):
        """Returns the Variance Reduction Factor of the prediction step of the filter. The VRF is the normalized variance for the filter.

        References
        ----------
        Asquith, "Weight Selection in First Order Linear Filters"
        Report No RG-TR-69-12, U.S. Army Missile Command. Redstone Arsenal, Al. November 24, 1970.
        """

        g = self.g
        h = self.h

        return (2 * g**2 + 2 * h + g * h) / (g * (4 - 2 * g - h))

    def VRF(self):
        """Returns the Variance Reduction Factor (VRF) of the state variable of the filter (x) and its derivatives (dx, ddx). The VRF is the normalized variance for the filter."""

        g = self.g
        h = self.h

        den = g * (4 - 2 * g - h)

        vx = (2 * g**2 + 2 * h - 3 * g * h) / den
        vdx = 2 * h**2 / (self.dt**2 * den)

        return (vx, vdx)

    def __repr__(self):
        return "\n".join(
            [
                "GHFilter object",
                pretty_str("dt", self.dt),
                pretty_str("g", self.g),
                pretty_str("h", self.h),
                pretty_str("x", self.x),
                pretty_str("dx", self.dx),
                pretty_str("x_pred", self.x_pred),
                pretty_str("dx_pred", self.dx_pred),
                pretty_str("y", self.y),
                pretty_str("z", self.z),
            ]
        )


class GHKFilter:
    """Implements the g-h-k filter.

    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.
    """

    def __init__(self, x0, dx, ddx, dt, g, h, k):
        self.x = x0
        self.dx = dx
        self.ddx = ddx
        self.x_pred = self.x
        self.dx_pred = self.dx
        self.ddx_pred = self.ddx

        self.dt = dt
        self.g = g
        self.h = h
        self.k = k

        if np.ndim(x0) == 0:
            self.y = 0.0  # residual
            self.z = 0.0
        else:
            self.y = np.zeros(len(x0))
            self.z = np.zeros(len(x0))

    def update(self, z, g=None, h=None, k=None):
        """Performs the g-h filter predict and update step on the
        measurement z.

        On return, self.x, self.dx, self.y, and self.x_pred will have been updated with the estimates of the computation. For convenience, self.x and self.dx are returned in a tuple.
        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h
        if k is None:
            k = self.k

        dt = self.dt
        dt_sqr = dt**2
        # prediction step
        self.ddx_pred = self.ddx
        self.dx_pred = self.dx + self.ddx * dt
        self.x_pred = self.x + self.dx * dt + 0.5 * self.ddx * (dt_sqr)

        # update step
        self.y = z - self.x_pred

        self.ddx = self.ddx_pred + 2 * k * self.y / dt_sqr
        self.dx = self.dx_pred + h * self.y / dt
        self.x = self.x_pred + g * self.y

        return (self.x, self.dx)

    def batch_filter(self, data, save_preds=False):
        """Performs g-h filter with a fixed g and h."""

        x = self.x
        dx = self.dx
        n = len(data)

        estimates = np.zeros((n + 1, 2))
        estimates[0, 0] = x
        estimates[0, 1] = dx

        if save_preds:
            predictions = np.zeros(n)

        # optimization to avoid n computations of h / dt
        h_dt = self.h / self.dt

        for i, z in enumerate(data):
            # prediction step
            x_est = x + (dx * self.dt)

            # update step
            residual = z - x_est
            dx = dx + h_dt * residual  # i.e. dx = dx + h * residual / dt
            x = x_est + self.g * residual

            estimates[i + 1, 0] = x
            estimates[i + 1, 1] = dx

            if save_preds:
                predictions[i] = x_est

        return (estimates, predictions) if save_preds else estimates

    def VRF_pred(self):
        """Returns the Variance Reduction Factor for x of the prediction
        step of the filter.

        References
        ----------
        Asquith and Woods, "Total Error Minimization in First
        and Second Order Prediction Filters" Report No RE-TR-70-17, U.S.
        Army Missile Command. Redstone Arsenal, Al. November 24, 1970.
        """

        g = self.g
        h = self.h
        k = self.k
        gh2 = 2 * g + h
        return (g * k * (gh2 - 4) + h * (g * gh2 + 2 * h)) / (
            2 * k - (g * (h + k) * (gh2 - 4))
        )

    def bias_error(self, dddx):
        """Returns the bias error given the specified constant jerk

        References
        ----------
        Asquith and Woods, "Total Error Minimization in First
        and Second Order Prediction Filters" Report No RE-TR-70-17, U.S.
        Army Missile Command. Redstone Arsenal, Al. November 24, 1970.
        """

        return -(self.dt**3) * dddx / (2 * self.k)

    def VRF(self):
        """Returns the Variance Reduction Factor (VRF) of the state variable of the filter (x) and its derivatives (dx, ddx). The VRF is the normalized variance for the filter, as given in the equations below."""

        g = self.g
        h = self.h
        k = self.k

        # common subexpressions in the equations pulled out for efficiency,
        # they don't 'mean' anything.
        hg4 = 4 - 2 * g - h
        ghk = g * h + g * k - 2 * k

        vx = (2 * h * (2 * (g**2) + 2 * h - 3 * g * h) - 2 * g * k * hg4) / (
            2 * k - g * (h + k) * hg4
        )
        vdx = (2 * (h**3) - 4 * (h**2) * k + 4 * (k**2) * (2 - g)) / (2 * hg4 * ghk)
        vddx = 8 * h * (k**2) / ((self.dt**4) * hg4 * ghk)

        return (vx, vdx, vddx)

    def __repr__(self):
        return "\n".join(
            [
                "GHFilter object",
                pretty_str("dt", self.dt),
                pretty_str("g", self.g),
                pretty_str("h", self.h),
                pretty_str("k", self.k),
                pretty_str("x", self.x),
                pretty_str("dx", self.dx),
                pretty_str("ddx", self.ddx),
                pretty_str("x_pred", self.x_pred),
                pretty_str("dx_pred", self.dx_pred),
                pretty_str("ddx_pred", self.dx_pred),
                pretty_str("y", self.y),
                pretty_str("z", self.z),
            ]
        )


def optimal_noise_smoothing(g):
    """Provides g,h,k parameters for optimal smoothing of noise for a given value of g.

    References
    ----------
    Polge and Bhagavan. "A Study of the g-h-k Tracking Filter".
    Report No. RE-CR-76-1. University of Alabama in Huntsville.
    July, 1975
    """

    h = ((2 * g**3 - 4 * g**2) + (4 * g**6 - 64 * g**5 + 64 * g**4) ** 0.5) / (
        8 * (1 - g)
    )
    k = (h * (2 - g) - g**2) / g

    return (g, h, k)


def least_squares_params(n):
    """An order 1 least squared filter can be computed by a g-h filter
    by varying g and h over time according to the formulas below, where
    the first measurement is at n=0, the second is at n=1, and so on.
    """
    den = (n + 2) * (n + 1)

    g = (2 * (2 * n + 1)) / den
    h = 6 / den
    return (g, h)


def critical_damping_params(theta, order=2):
    """Computes values for g,h (and k for g-h-k filter) for a
    critically damped filter.

    The idea here is to create a filter that reduces the influence of
    old data as new data comes in. This allows the filter to track a
    moving target better. This goes by different names. It may be called the discounted least-squares g-h filter, a fading-memory polynomial filter of order 1, or a critically damped g-h filter.

    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.

    Polge and Bhagavan. "A Study of the g-h-k Tracking Filter".
    Report No. RE-CR-76-1. University of Alabama in Huntsville. July, 1975
    """

    if theta < 0 or theta > 1:
        raise ValueError("theta must be between 0 and 1")

    if order == 2:
        return (1.0 - theta**2, (1.0 - theta) ** 2)

    if order == 3:
        return (
            1.0 - theta**3,
            1.5 * (1.0 - theta**2) * (1.0 - theta),
            0.5 * (1 - theta) ** 3,
        )

    raise ValueError(f"bad order specified: {order}")


def benedict_bornder_constants(g, critical=False):
    """Computes the g,h constants for a Benedict-Bordner filter, which
    minimizes transient errors for a g-h filter.

    Returns the values g,h for a specified g. Strictly speaking, only h
    is computed, g is returned unchanged.

    The default formula for the Benedict-Bordner allows ringing. We can
    "nearly" critically damp it; ringing will be reduced, but not entirely eliminated at the cost of reduced performance.

    References
    ----------
    Brookner, "Tracking and Kalman Filters Made Easy". John Wiley and
    Sons, 1998.
    """

    g_sqr = g**2
    if critical:
        return (g, 0.8 * (2.0 - g_sqr - 2 * (1 - g_sqr) ** 0.5) / g_sqr)

    return (g, g_sqr / (2.0 - g))
