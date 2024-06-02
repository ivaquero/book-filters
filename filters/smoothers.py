import numpy as np
from numpy import linalg

from .helpers import pretty_str
from .kalman import KalmanFilter


def rts_smoother(Xs, Ps, Fs, Q):
    n, dim_x, _ = Xs.shape
    # smoother gain
    K = np.zeros((n, dim_x, dim_x))
    x, P, Pp, F = Xs.copy(), Ps.copy(), Ps.copy, Fs.copy
    for k in range(n - 2, -1, -1):
        Pp[k] = F @ P[k] @ F.T + Q
        K[k] = P[k] @ F.T @ linalg.inv(Pp[k])
        x[k] += K[k] @ (x[k + 1] - F @ x[k])
        P[k] += K[k] @ (P[k + 1] - Pp[k] @ K[k].T)
    return (x, P, K, Pp)


class FixedLagSmoother:
    """Fixed Lag Kalman smoother."""

    def __init__(self, dim_x, dim_z, N=None):
        """Create a fixed lag Kalman filter smoother."""

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.N = N

        self.x = np.zeros((dim_x, 1))
        self.x_s = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.F = np.eye(dim_x)
        self.H = np.eye(dim_z, dim_x)
        self.R = np.eye(dim_z)
        self.K = np.zeros((dim_x, 1))
        self.y = np.zeros((dim_z, 1))
        self.G = 0.0
        self.S = np.zeros((dim_z, dim_z))

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        self.count = 0

        if N is not None:
            self.xSmooth = []

    def smooth(self, z, u=None):
        """Smooths the measurement using a fixed lag smoother.

        On return, self.xSmooth is populated with the N previous smoothed estimates, where self.xSmooth[k] is the kth time step. self.x merely contains the current Kalman filter output of the most recent measurement, and is not smoothed at all (beyond the normal Kalman filter processing).

        self.xSmooth grows in length on each call. If you run this 1 million times. If you want to filter something else, create a new FixedLagSmoother object.
        """

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        G = self.G
        N = self.N

        k = self.count

        # predict step of normal Kalman filter
        x_pre = F @ x
        if u is not None:
            x_pre += G @ u

        P = F @ P @ F.T + Q

        # update step of normal Kalman filter
        self.y = z - H @ x_pre

        self.S = H @ P @ H.T + R
        SI = linalg.inv(self.S)

        K = P @ H.T @ SI
        x = x_pre + K @ self.y
        I_KH = self._I - K @ H
        P = I_KH @ P @ I_KH.T + K @ R @ K.T

        self.xSmooth.append(x_pre.copy())

        # compute invariants
        HTSI = H.T @ SI
        F_LH = (F - K @ H).T

        if k >= N:
            PS = P.copy()  # smoothed P for step i
            for i in range(N):
                K = PS @ HTSI  # smoothed gain
                PS = PS @ F_LH  # smoothed covariance

                si = k - i
                self.xSmooth[si] = self.xSmooth[si] + K @ self.y
        else:
            # Some sources specify starting the fix lag smoother only
            # after N steps have passed, some don't. I am getting far
            # better results by starting only at step N.
            self.xSmooth[k] = x.copy()

        self.count += 1
        self.x = x
        self.P = P

    def smooth_batch(self, zs, N, us=None):
        """Batch smooths the set of measurements using a fixed lag smoother.

        This is a batch processor, so it does not alter any of the object's data. In particular, self.x is NOT modified. All date is returned by the function.
        """

        # take advantage of the fact that np.array are assigned by reference.
        H = self.H
        R = self.R
        F = self.F
        P = self.P
        x = self.x
        Q = self.Q
        G = self.G

        if x.ndim == 1:
            xSmooth = np.zeros((len(zs), self.dim_x))
            xhat = np.zeros((len(zs), self.dim_x))
        else:
            xSmooth = np.zeros((len(zs), self.dim_x, 1))
            xhat = np.zeros((len(zs), self.dim_x, 1))
        for k, z in enumerate(zs):
            # predict step of normal Kalman filter
            x_pre = F @ x
            if us is not None:
                x_pre += G @ us[k]

            P = F @ P @ F.T + Q
            # update step of normal Kalman filter
            y = z - H @ x_pre
            S = H @ P @ H.T + R
            SI = linalg.inv(S)
            K = P @ H.T @ SI

            x = x_pre + K @ y
            I_KH = self._I - K @ H
            P = I_KH @ P @ I_KH.T + K @ R @ K.T

            xhat[k] = x.copy()
            xSmooth[k] = x_pre.copy()

            # compute invariants
            HTSI = H.T @ SI
            F_LH = (F - K @ H).T

            if k >= N:
                PS = P.copy()  # smoothed P for step i
                for i in range(N):
                    K = PS @ HTSI  # smoothed gain
                    PS = PS @ F_LH  # smoothed covariance

                    si = k - i
                    xSmooth[si] = xSmooth[si] + K @ y
            else:
                # Some sources specify starting the fix lag smoother only
                # after N steps have passed, some don't. I am getting far
                # better results by starting only at step N.
                xSmooth[k] = xhat[k]

        return xSmooth, xhat

    def __repr__(self):
        return "\n".join(
            [
                "FixedLagSmoother object",
                pretty_str("dim_x", self.x),
                pretty_str("dim_z", self.x),
                pretty_str("N", self.N),
                pretty_str("x", self.x),
                pretty_str("x_s", self.x_s),
                pretty_str("P", self.P),
                pretty_str("F", self.F),
                pretty_str("Q", self.Q),
                pretty_str("R", self.R),
                pretty_str("H", self.H),
                pretty_str("K", self.K),
                pretty_str("y", self.y),
                pretty_str("S", self.S),
                pretty_str("B", self.G),
            ]
        )


def tf_smooth(M, P, Y, A, Q, H, R, use_inf=True):
    """
    Two filter based Smoother

    Syntax:
    [M,P] = TF_SMOOTH(M,P,Y,A,Q,H,R,[use_inf])

    In:
    M - NxK matrix of K mean estimates from Kalman filter
    P - NxNxK matrix of K state covariances from Kalman Filter
    Y - Sequence of K measurement as DxK matrix
    A - NxN state transition matrix.
    Q - NxN process noise covariance matrix.
    H - DxN Measurement matrix.
    R - DxD Measurement noise covariance.
    use_inf - If information filter should be used (default 1)

    Out:
    M - Smoothed state mean sequence
    P - Smoothed state covariance sequence

    Description:
    Two filter linear smoother algorithm. Calculate "smoothed"
    sequence from given Kalman filter output sequence
    by conditioning all steps to all measurements.
    """

    M = M.copy()
    P = P.copy()

    m_ = M.shape
    p_ = P.shape

    # Run the backward filter
    if use_inf:
        zz = np.zeros(m_)
        SS = np.zeros(p_)
        IR = linalg.inv(R)
        IQ = linalg.inv(Q)
        z = np.zeros((m_[1], 1))
        S = np.zeros((m_[1], m_[1]))
        for k in range(m_[0] - 1, -1, -1):
            G = linalg.solve(S + IQ, S).T
            S = A.T @ (np.eye(m_[1]) - G) @ S @ A
            z = A.T @ (np.eye(m_[1]) - G) @ z
            zz[k] = z
            SS[k] = S
            S += H.T @ IR @ H
            z += H.T @ IR @ Y[k]
    else:
        BM = np.zeros(m_[1])
        BP = np.zeros(p_)
        IA = linalg.inv(A)
        IQ = IA @ Q @ IA.T
        fm = np.zeros((m_[1], 1))
        fP = 1e12 * np.eye(m_[1])
        BM[:] = fm
        BP[:] = fP
        for k in range(m_[0] - 2, -1, -1):
            fm, fP, *_ = KalmanFilter().update(fm, fP, Y[k + 1], H, R)
            fm, fP = KalmanFilter().predict(fm, fP, IA, IQ)
            BM[k] = fm
            BP[k] = fP

    # Combine estimates
    if use_inf:
        for k in range(m_[0] - 1):
            G = P[k] @ linalg.solve((np.eye(m_[1]) + P[k] @ SS[k]).T, SS[k].T)
            P[k] = linalg.inv(linalg.inv(P[k]) + SS[k])
            M[k] = M[k] + P[k] @ zz[k] - G @ M[k]
    else:
        for k in range(m_[0] - 1):
            tmp = linalg.inv(linalg.inv(P[k]) + linalg.inv(BP[k]))
            M[k] = tmp @ (linalg.solve(P[k], M[k]) + linalg.solve(BP[k], BM[k]))
            P[k] = tmp

    return M, P
