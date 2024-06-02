from abc import ABC, abstractclassmethod


class StateSpaceModel(ABC):
    """Staet space model."""

    @abstractclassmethod
    def state_equation(self, t, x, u=0, w=0):
        """Calculate the state equation.

        x[t+1] = f(t, x[t], u[t], w[t])
        f: state equation.
        x: state
        u: control input
        w: system noise
        t: time
        """

    @abstractclassmethod
    def observation_equation(self, t, x, v=0):
        """Calculate the observation equation.

        y[t] = h(t, x[t], v[t])
        h: observation equation
        z: output
        v: observation noise
        t: time
        """

    def ndim(self, key):
        """Return dimensions."""
        return self.NDIM[key]

    def update_param(self, *params):
        """Update the time-varing parameters in the state space model."""
        raise NotImplementedError
