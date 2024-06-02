import numpy as np
from scipy import linalg, stats


def gh_packed_pc(x, fmm_param):
    """
    GH_PACKED_PC - Pack P and C for the Gauss-Hermite transformation

    Syntax:
            pc = GH_PACKED_PC(x,fmm_param)

    In:
            x - Evaluation point
            fmm_param - Array of handles and parameters to form the functions.

    Out:
            pc - Output values

    Description:
    Packs the integrals that need to be evaluated in nice function form to ease the evaluation. Evaluates P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'.
    """

    f = fmm_param[0]
    m = fmm_param[1]
    fm = fmm_param[2]
    if len(fmm_param) >= 4:
        param = fmm_param[3]

    if type(f) == str or callable(f):
        F = f(x) if "param" not in locals() else f(x, param)
    elif type(f) == np.ndarray:
        F = f @ x
    else:
        F = f(x) if "param" not in locals() else f(x, param)
    d = x.shape[0]
    s = F.shape[0]

    # Compute P = (f-fm)(f-fm)' and C = (x-m)(f-fm)'
    # and form array of [vec(P):vec(C)]
    f_ = F.shape[1]
    pc = np.zeros((s**2 + d * s, f_))
    P = np.zeros((s, s))
    C = np.zeros((d, s))
    for k in range(f_):
        for j in range(s):
            for i in range(s):
                P[i, j] = (F[i, k] - fm[i]) * (F[j, k] - fm[j])
            for i in range(d):
                C[i, j] = (x[i, k] - m[i]) * (F[j, k] - fm[j])
        pc[:, k] = np.concatenate([P.reshape(s * s), C.reshape(s * d)])
    return pc


def hermite_polynomial(n):
    """
    HERMITE_POLYNOMIAL - Hermite polynomial

    Syntax:
        p = hermite_polynomial(n)

    In:
        n - Polynomial order

    Out:
        p - Polynomial coefficients (starting from greatest order)

    Description:
        Forms the Hermite polynomial of order n.

    The "physicists' Hermite polynomials"
    To get the differently scaled "probabilists' Hermite polynomials"
    remove the coefficient *2 in (**).
    """

    n = max(n, 0)

    # Allocate space for the polynomials and set H0(x) = -1
    H = np.zeros((n + 1, n + 1), dtype=int)
    r = np.arange(1, n + 1)
    H[0, 0] = -1

    # Calculate the polynomials starting from H2(x)
    for i in range(1, n + 1):
        H[i, 1 : n + 1] -= H[i - 1, :n] * 2  # (**)
        H[i, :n] += H[i - 1, 1 : n + 1] * r

    # Return results
    return H[n][::-1] * (-1) ** (n + 1)


def gh_nd_scaled(f, n, m, P, param=None):
    """
    NGAUSSHERMI - ND scaled Gauss-Hermite quadrature (cubature) rule

    Syntax:
            [I,x,W,F] = gh_nd_scaled(f,p,m,P,param)

    In:
            f - Function f(x,param) as inline, name or reference
            n - Polynomial order
            m - Mean of the d-dimensional Gaussian distribution
            P - Covariance of the Gaussian distribution
            param - Optional parameters for the function

    Out:
            I - The integral value
            x - Evaluation points
            W - Weights
            F - Function values

    Description:
            Approximates a Gaussian integral using the Gauss-Hermite method
            in multiple dimensions:
                    int f(x) N(x | m,P) dx
    """

    # The Gauss-Hermite cubature rule

    # The hermite polynomial of order n
    p = hermite_polynomial(n)

    # Evaluation points
    x = np.roots(p)

    # 1D coefficients
    Wc = 2 ** (n - 1) * np.math.factorial(n) * np.sqrt(np.pi) / n**2
    p2 = hermite_polynomial(n - 1)
    W = np.zeros(n)
    for i in range(n):
        W[i] = Wc * np.polyval(p2, x[i]) ** (-2)

    d = m.shape[0]
    if d == 1:
        x = x.T

    # Generate all n^d collections of indexes by
    # transforming numbers 0...n^d-1) into n-base system
    num = np.arange(n**d)
    ind = np.zeros((d, n**d), dtype=int)
    for i in range(d):
        ind[i] = num % n
        num //= n

    # Form the sigma points and weights
    L = linalg.cholesky(P)
    SX = np.sqrt(2) * L @ x[ind] + np.tile(m, ind.shape[1])
    W = np.prod(W[ind], axis=0)  # ND weights

    # Evaluate the function at the sigma points
    if type(f) == str or callable(f):
        F = f(SX) if param is None else f(SX, param)
    elif type(f) == np.ndarray:
        F = f @ SX
    else:
        F = f(SX) if param is None else f(SX, param)

    # Evaluate the integral
    I = (
        np.sum(F * np.tile(W, (F.shape[0], 1)), axis=1, keepdims=True)
        / np.sqrt(np.pi) ** d
    )

    return I, SX, x, W, F


def gh_transform(m, P, g, g_param=None, tr_param=None):
    """
    GH_TRANSFORM - Gauss-Hermite transform of random variables

    Syntax:
      [mu,S,C,SX,W] = GH_TRANSFORM(M,P,g,p,param)

    In:
      M - Random variable mean (Nx1 column vector)
      P - Random variable covariance (NxN pos.def. matrix)
      g - Transformation function of the form g(x,param) as
          matrix, inline function, function name or function reference
      g_param - Parameters of g               (optional, default empty)
      tr_param - Parameters of the integration method in form {p}:
          p - Number of points in Gauss-Hermite integration

    Out:
      mu - Estimated mean of y
       S - Estimated covariance of y
       C - Estimated cross-covariance of x and y
      SX - Sigma points of x
       W - Weights as cell array
    """

    p = tr_param if tr_param is not None else 3

    # Estimate the mean of g
    if g_param is None:
        mu, SX, _, W, _ = gh_nd_scaled(g, p, m, P)
    else:
        mu, SX, _, W, _ = gh_nd_scaled(g, p, m, P, g_param)

    # Estimate the P and C
    if tr_param is None:
        pc, SX, _, W, _ = gh_nd_scaled(gh_packed_pc, p, m, P, [g, m, mu])
    else:
        pc, SX, _, W, _ = gh_nd_scaled(gh_packed_pc, p, m, P, [g, m, mu, g_param])

    d = m.shape[0]
    s = mu.shape[0]
    S = pc[: s**2].reshape((s, s))
    C = pc[s**2 :].reshape((d, s))

    return mu, S, C, SX, W


def gh_kf_predict(M, P, f=None, Q=None, f_param=None, p=10):
    """
    GHKF_PREDICT - Gauss-Hermite Kalman filter prediction step

    Syntax:
            [M,P] = GHKF_PREDICT(M,P,[f,Q,f_param,p])

    In:
            M - Nx1 mean state estimate of previous step
            P - NxN state covariance of previous step
            f - Dynamic model function as a matrix A defining
                            linear function f(x) = A*x, inline function,
                            function handle or name of function in
                            form f(x,param)                   (optional, default eye())
            Q - Process noise of discrete model   (optional, default zero)
            f_param - Parameters of f               (optional, default empty)
            p - Degree of approximation (number of quadrature points)

    Out:
            M - Updated state mean
            P - Updated state covariance

    Description:
            Perform additive form Gauss-Hermite Kalman Filter prediction step.

            Function f(.) should be such that it can be given a
            DxN matrix of N sigma Dx1 points and it returns
            the corresponding predictions for each sigma
            point.
    """

    m_ = M.shape[0]

    # Apply defaults
    if f is None:
        f = np.eye(m_)
    if Q is None:
        Q = np.zeros(m_)

    # Do transform and add process noise
    M, P, *_ = gh_transform(M, P, f, f_param, p)
    P += Q

    return M, P


def gh_kf_update(M, P, Y, h, R=None, h_param=None, p=3):
    """
    GHKF_UPDATE - Gauss-Hermite Kalman filter update step

    Syntax:
            [M,P,K,MU,S,LH] = GHKF_UPDATE(M,P,Y,h,R,param,p)

    In:
            M  - Mean state estimate after prediction step
            P  - State covariance after prediction step
            Y  - Measurement vector.
            h  - Measurement model function as a matrix H defining
                    linear function h(x) = H*x, inline function,
                    function handle or name of function in
                    form h(x,param)
            R  - Measurement covariance
            h_param - Parameters of h
            p  - Degree of approximation (number of quadrature points)

    Out:
            M  - Updated state mean
            P  - Updated state covariance
            K  - Computed Kalman gain
            MU - Predictive mean of Y
            S  - Predictive covariance Y
            LH - Predictive probability (likelihood) of measurement.

    Description:
            Perform additive form Gauss-Hermite Kalman filter (GHKF)
            measurement update step. Assumes additive measurement
            noise.

            Function h(.) should be such that it can be given a
            DxN matrix of N sigma Dx1 points and it returns
            the corresponding measurements for each sigma
            point. This function should also make sure that
            the returned sigma points are compatible such that
            there are no 2pi jumps in angles etc.

    Example:
            h = inline('atan2(x(2,:)-s(2),x(1,:)-s(1))','x','s');
            [M2,P2] = gh_kf_update(M1,P1,Y,h,R,S);
    """

    # Do the transform and make the update
    MU, S, C, *_ = gh_transform(M, P, h, h_param, p)
    S += R
    K = linalg.solve(S.T, C.T).T
    M += K @ (Y - MU)
    P -= K @ S @ K.T

    if h_param is not None:
        LH = stats.multivariate_normal.pdf(Y, MU, S)
        return M, P, K, MU, S, LH

    return M, P, K, MU, S


def gh_rts_smooth(M, P, f, Q=None, f_param=None, p=None, same_p=True):
    """
    GHRTS_SMOOTH - Additive form Gauss-Hermite Rauch-Tung-Striebel smoother

    Syntax:
            [M,P,D] = GHRTS_SMOOTH(M,P,f,Q,[f_param,p,same_p])

    In:
            M - NxK matrix of K mean estimates from Gauss-Hermite Kalman filter
            P - NxNxK matrix of K state covariances from Gauss-Hermite filter
            f - Dynamic model function as a matrix A defining
                    linear function f(x) = A*x, inline function,
                    function handle or name of function in
                    form a(x,param)                   (optional, default eye())
            Q - NxN process noise covariance matrix or NxNxK matrix
                    of K state process noise covariance matrices for each step.
            f_param - Parameters of f(.). Parameters should be a single cell array,
                            vector or a matrix containing the same parameters for each
                            step, or if different parameters are used on each step they
                            must be a cell array of the format { param_1, param_2, ...},
                            where param_x contains the parameters for step x as a cell
                            array, a vector or a matrix.   (optional, default empty)
            p - Degree on approximation (number of quadrature points)
            same_p - If set to '1' uses the same parameters on every time step
                                    (optional, default 1)

    Out:
            M - Smoothed state mean sequence
            P - Smoothed state covariance sequence
            D - Smoother gain sequence

    Description:
            Gauss-Hermite Rauch-Tung-Striebel smoother algorithm. Calculate "smoothed" sequence from given Kalman filter output sequence by conditioning all steps to all measurements.
    """
    M = M.copy()
    P = P.copy()

    # Apply defaults
    m_1, m_2 = M.shape[:2]

    if f is None:
        f = np.eye(m_2)

    if Q is None:
        Q = np.zeros(m_2)

    if p is None:
        p = 10

    # Extend Q if NxN matrix
    if len(Q.shape) == 2:
        Q = np.tile(Q, (m_1, 1, 1))

    # Run the smoother
    D = np.zeros((m_1, m_2, m_2))
    if f_param is None:
        for k in range(m_1 - 2, -1, -1):
            m_pred, P_pred, C, *_ = gh_transform(M[k], P[k], f, f_param, p)
            P_pred += Q[k]
            D[k] = linalg.solve(P_pred.T, C.T).T
            M[k] += D[k] @ (M[k + 1] - m_pred)
            P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T
    else:
        for k in range(m_1 - 2, -1, -1):
            if f_param is None:
                params = None
            elif same_p:
                params = f_param
            else:
                params = f_param[k]

            m_pred, P_pred, C, *_ = gh_transform(M[k], P[k], f, params, p)
            P_pred += Q[k]
            D[k] = linalg.solve(P_pred.T, C.T).T
            M[k] += D[k] @ (M[k + 1] - m_pred)
            P[k] += D[k] @ (P[k + 1] - P_pred) @ D[k].T

    return M, P, D
