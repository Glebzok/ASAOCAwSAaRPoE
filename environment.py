import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from math import pi


class Environment(object):
    class RHS(nn.Module):
        def __init__(self, f, g, u):
            super().__init__()
            self.f = f
            self.g = g
            self.u = u

        def forward(self, t, x):
            return self.f(x) + self.g(x) @ self.u

    def __init__(self, integration_method, u_max, theta_func):
        self.u_max = u_max
        self.integration_method = integration_method
        self.theta_func = theta_func

    def f(self, x):
        raise NotImplementedError()

    def g(self, x):
        raise NotImplementedError()

    def q(self, x):
        raise NotImplementedError()

    def r(self, x):
        raise NotImplementedError()

    def theta(self, x):
        if self.theta_func == 'tanh':
            return self.u_max * torch.tanh(x / self.u_max)
        elif self.theta_func == 'sigmoid':
            return self.u_max * (2. / (1 + torch.exp(-x / self.u_max)) - 1)
        else:
            raise ValueError('Wrong theta function')

    def theta_inv(self, x):
        if self.theta_func == 'tanh':
            return self.u_max * torch.clip(torch.atanh(torch.clip(x / self.u_max, -1, 1)), -10, 10)
        elif self.theta_func == 'sigmoid':
            return -self.u_max * torch.clip(torch.log(torch.clip(2 / (x / self.u_max + 1) - 1, 0)), -10, 10)
        else:
            raise ValueError('Wrong theta function')

    def r_s(self, u):
        v_grid = torch.linspace(start=0, end=1, steps=100) * u
        theta_grid = self.theta_inv(v_grid) * self.r(v_grid)
        return 2 * torch.trapz(y=theta_grid, x=v_grid).sum()

    def propagate(self, x, u, t, h):
        u = u.clone()
        if t <= 1:
            u += 0.5 * (torch.sin(0.3 * pi * t) + torch.cos(0.3 * pi * t))
        x_h = \
            odeint(func=self.RHS(f=self.f, g=self.g, u=u), y0=x, t=torch.tensor([0., h]),
                   method=self.integration_method)[
                -1]
        return x_h

    def reset_state(self):
        raise NotImplementedError()


class VanDerPolOscillator(Environment):
    def __init__(self, integration_method, theta_func='tanh', u_max=0.1):
        super().__init__(integration_method=integration_method, u_max=u_max, theta_func=theta_func)

    def f(self, x):
        return torch.tensor([x[1],
                             -x[0] - 0.5 * x[1] * (1 - x[0] ** 2) - x[0] ** 2 * x[1]]).view(-1, 1)

    def g(self, x):
        return torch.tensor([0, x[0]]).view(-1, 1)

    def q(self, x):
        return torch.linalg.norm(x).item() ** 2

    def r(self, x):
        return 1

    def reset_state(self):
        return torch.tensor([1., -1.]).view(-1, 1)


class PowerPlantSystem(Environment):
    def __init__(self, integration_method, theta_func='tanh', u_max=0.02, T_g=0.08, T_t=0.1, T_p=20.,
                 R_g=2.5, K_p=120., K_t=1.):
        super().__init__(integration_method=integration_method, u_max=u_max, theta_func=theta_func)

        self.T_g = T_g
        self.T_t = T_t
        self.T_p = T_p
        self.R_g = R_g
        self.K_p = K_p
        self.K_t = K_t

    def f(self, x):
        return (torch.tensor([[-1 / self.T_g, 0, 1 / (self.R_g * self.T_g)],
                              [self.K_t / self.T_t, -1 / self.T_t, 0],
                              [0, self.K_p / self.T_p, -1 / self.T_p]]) @ x).view(-1, 1)

    def g(self, x):
        return torch.tensor([1 / self.T_g,
                             0.,
                             0.]).view(-1, 1)

    def q(self, x):
        return torch.linalg.norm(x).item() ** 2

    def r(self, x):
        return 0.5

    def reset_state(self):
        return torch.tensor([0., 0.1, 0.05]).view(-1, 1)

    def propagate(self, x, u, t, h):
        x_h = super().propagate(x=x, u=u, t=t, h=h)
        if 7 <= t <= 11:
            x_h[2] = torch.minimum(torch.tensor(1.5), x_h[2] * 1.5)
        return x_h
