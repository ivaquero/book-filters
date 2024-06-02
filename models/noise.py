import numpy as np
from scipy import linalg


def order_by_derivative(Q, dim, block_size):
    """Order the matrix Q by derivative order

    Parameters
    ----------
    Q : _type_
        _description_
    dim : _type_
        _description_
    block_size : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    N = dim * block_size
    D = np.zeros((N, N))
    Q = np.array(Q)
    for i, x in enumerate(Q.ravel()):
        f = np.eye(block_size) * x
        idx, idy = (i // dim) * block_size, (i % dim) * block_size
        D[idx : idx + block_size, idy : idy + block_size] = f
    return D


def white_noise_discrete(
    dim,
    dt: float = 1.0,
    var: float = 1.0,
    block_size: int = 1,
    order_by_dim: bool = True,
    seed: int = 1123,
):
    """Generate white noise with discrete time steps

    Parameters
    ----------
    dim : _type_
        _description_
    dt : float, optional
        _description_, by default 1.0
    var : float, optional
        _description_, by default 1.0
    block_size : int, optional
        _description_, by default 1
    order_by_dim : bool, optional
        _description_, by default True
    seed : int, optional
        _description_, by default 1123

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    np.random.seed(seed)

    if dim == 2:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3],
            [0.5 * dt**3, dt**2],
        ]
    elif dim == 3:
        Q = [
            [0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
            [0.5 * dt**3, dt**2, dt],
            [0.5 * dt**2, dt, 1],
        ]
    elif dim == 4:
        Q = [
            [(dt**6) / 36, (dt**5) / 12, (dt**4) / 6, (dt**3) / 6],
            [(dt**5) / 12, (dt**4) / 4, (dt**3) / 2, (dt**2) / 2],
            [(dt**4) / 6, (dt**3) / 2, dt**2, dt],
            [(dt**3) / 6, (dt**2) / 2, dt, 1.0],
        ]
    else:
        raise ValueError("dim must be 1, 2, 3, 4")

    if order_by_dim:
        return linalg.block_diag(*[Q] * block_size) * var
    return order_by_derivative(np.array(Q), dim, block_size) * var


def white_noise_continuous(
    dim,
    dt: float = 1.0,
    spectral_density: float = 1.0,
    block_size: int = 1,
    order_by_dim: bool = True,
):
    """Generate white noise with continuous time steps

    Parameters
    ----------
    dim : _type_
        _description_
    dt : float, optional
        _description_, by default 1.0
    spectral_density : float, optional
        _description_, by default 1.0
    block_size : int, optional
        _description_, by default 1
    order_by_dim : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    if dim == 2:
        Q = [[(dt**3) / 3.0, (dt**2) / 2.0], [(dt**2) / 2.0, dt]]
    elif dim == 3:
        Q = [
            [(dt**5) / 20.0, (dt**4) / 8.0, (dt**3) / 6.0],
            [(dt**4) / 8.0, (dt**3) / 3.0, (dt**2) / 2.0],
            [(dt**3) / 6.0, (dt**2) / 2.0, dt],
        ]

    elif dim == 4:
        Q = [
            [(dt**7) / 252.0, (dt**6) / 72.0, (dt**5) / 30.0, (dt**4) / 24.0],
            [(dt**6) / 72.0, (dt**5) / 20.0, (dt**4) / 8.0, (dt**3) / 6.0],
            [(dt**5) / 30.0, (dt**4) / 8.0, (dt**3) / 3.0, (dt**2) / 2.0],
            [(dt**4) / 24.0, (dt**3) / 6.0, (dt**2 / 2.0), dt],
        ]
    else:
        ValueError("dim must be 2, 3, 4")

    if order_by_dim:
        return linalg.block_diag(*[Q] * block_size) * spectral_density

    return order_by_derivative(np.array(Q), dim, block_size) * spectral_density
