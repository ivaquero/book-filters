from sympy import MatMul, Matrix, init_printing, integrate, symbols

init_printing(use_latex="mathjax")


def gen_white_noise_discrete():
    dt, phi = symbols("Δt Φ_s")
    F_k = Matrix([[1, dt, dt**2 / 2], [0, 1, dt], [0, 0, 1]])
    Q_c = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 1]]) * phi
    Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

    Q = Q / phi
    print(MatMul(Q, phi))
    return dt, phi


def get_white_noise_0d():
    dt, phi = gen_white_noise_discrete()

    F_k = Matrix([[1]])
    Q_c = Matrix([[phi]])
    Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

    print("0th order discrete process noise")
    print(Q)


def gen_white_noise_1d():
    dt, phi = gen_white_noise_discrete()

    F_k = Matrix([[1, dt], [0, 1]])
    Q_c = Matrix([[0, 0], [0, 1]]) * phi
    Q = integrate(F_k * Q_c * F_k.T, (dt, 0, dt))

    print("1st order discrete process noise")
    Q = Q / phi
    print(MatMul(Q, phi))


def gen_white_noise_1d_piecewise():
    var = symbols("σ^2_v")

    dt, phi = gen_white_noise_discrete()
    v = Matrix([[dt**2 / 2], [dt]])
    Q = v * var * v.T

    Q = Q / var
    print(MatMul(Q, var))


def gen_white_noise_2d_piecewise():
    var = symbols("σ^2_v")

    dt, phi = gen_white_noise_discrete()
    v = Matrix([[dt**2 / 2], [dt], [1]])
    Q = v * var * v.T

    Q = Q / var
    print(MatMul(Q, var))
