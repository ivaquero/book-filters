def rk4(y0, x0, dx, func):
    """Computes 4th order Runge-Kutta for dy0/dx.

    Parameters
    ----------
    y0 : _type_
        _description_
    x0 : _type_
        _description_
    dx : _type_
        _description_
    func : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    k1 = dx * func(y0, x0)
    k2 = dx * func(y0 + 0.5 * k1, x0 + 0.5 * dx)
    k3 = dx * func(y0 + 0.5 * k2, x0 + 0.5 * dx)
    k4 = dx * func(y0 + k3, x0 + dx)
    return y0 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
