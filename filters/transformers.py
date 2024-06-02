import numpy as np
from scipy import linalg


def reshape_z(z, dim_z, ndim):
    """Ensure z is a (dim_z, 1) shaped vector.

    Parameters
    ----------
    z : _type_
        _description_
    dim_z : _type_
        _description_
    ndim : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError(
            f"z (shape {z.shape}) must be convertible to shape ({dim_z}, 1)"
        )

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z


def discretize_van_loan(F, G, dt):
    """Discretizes a linear differential equation which includes white noise according to the method of C. F. van Loan.

    References
    ----------
    C. F. van Loan. "Computing Integrals Involving the Matrix Exponential." IEEE Trans. Automomatic Control, AC-23 (3): 395-404 (June 1978)

    Parameters
    ----------
    F : _type_
        _description_
    G : _type_
        _description_
    dt : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    n = F.shape[0]

    A = np.zeros((2 * n, 2 * n))
    A[0:n, 0:n] = -F @ dt
    A[0:n, n : 2 * n] = G @ G.T @ dt
    A[n : 2 * n, n : 2 * n] = F.T @ dt

    B = linalg.expm(A)

    sigma = B[n : 2 * n, n : 2 * n].T
    Q = sigma @ B[0:n, n : 2 * n]

    return (sigma, Q)


def discretize_lti(F, L=None, Q=None, dt=None):
    """
    Discretize LTI ODE with Gaussian Noise

    Syntax:
    [A,Q] = lti_disc(F,L,Qc,dt)

    In:
    F  - NxN Feedback matrix
    L  - NxL Noise effect matrix        (optional, default identity)
    Qc - LxL Diagonal Spectral Density  (optional, default zeros)
    dt - Time Step                      (optional, default 1)

    Out:
    A - Transition matrix
    Q - Discrete Process Covariance

    Description:
    Discretize LTI ODE with Gaussian Noise. The original ODE model is in form

        dx/dt = F x + L w,  w ~ N(0,Qc)

    Result of discretization is the model

        x[k] = A x[k-1] + q, q ~ N(0,Q)

    Which can be used for integrating the model exactly over time steps, which are multiples of dt.
    """

    n = F.shape[0]
    if L is None:
        L = np.eye(n)
    if Q is None:
        Q = np.zeros(shape=(n, n))
    if dt is None:
        dt = 1

    # closed form integration of transition matrix
    A = linalg.expm(F * dt)

    # closed form integration of covariance by matrix fraction decomposition
    Phi = np.block([[F, L @ Q @ L.T], [np.zeros((n, n)), -F.T]])
    AB = linalg.expm(Phi * dt) @ np.vstack([np.zeros((n, n)), np.eye(n)])
    Q = linalg.solve(AB[n : 2 * n].T, AB[:n].T)

    return A, Q
