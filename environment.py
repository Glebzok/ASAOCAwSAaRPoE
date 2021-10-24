import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class Environment(object):
    class RHS(nn.Module):
        def __init__(self, f, g, u):
            super().__init__()
            self.f = f
            self.g = g
            self.u = u

        def forward(self, t, x):
            return self.f(x) + self.g(x) @ self.u

    def __init__(self, u_max):
        self.u_max = u_max

    def f(self, x):
        raise NotImplementedError()

    def g(self, x):
        raise NotImplementedError()

    def q(self, x):
        raise NotImplementedError()

    def r(self, x):
        raise NotImplementedError()

    def theta(self, x):
        return self.u_max * torch.tanh(x / self.u_max)

    def theta_inv(self, x):
        return self.u_max * torch.atanh(x / self.u_max)

    def r_s(self, u):
        r_s_l = []
        for u_i in u:
            u_grid = torch.linspace(0, u_i.item(), 100)
            theta_grid = self.theta_inv(u_grid)
            r_s_l.append(torch.trapz(theta_grid, u_grid))
            # print('r_s', u_grid, theta_grid, r_s_l[-1])
        return 2 * sum(r_s_l)

    def propogate(self, x, u, h):
        x_h = odeint(self.RHS(self.f, self.g, u), x, torch.tensor([0., h]), )[-1]
        return x_h

    def reset_state(self):
        raise NotImplementedError()


class VanDerPolOscillator(Environment):
    def __init__(self, u_max=0.1):
        super().__init__(u_max)

    def f(self, x):
        return torch.tensor([x[1],
                             -x[0] - .5 * x[1] * (1 - x[0] ** 2) - x[0] ** 2 * x[1]]).view(-1, 1)

    def g(self, x):
        return torch.tensor([0, x[0]]).view(-1, 1)

    def q(self, x):
        return torch.linalg.norm(x.float()).item()

    def r(self, x):
        return 1

    def reset_state(self):
        return torch.tensor([1., -1.]).view(-1, 1)


class PowerPlantSystem(Environment):
    def __init__(self, u_max=0.02, T_g=0.08, T_t=0.1, T_p=20,
                 R_g=2.5, K_p=120, K_t=1):
        super().__init__(u_max)

        self.T_g = T_g
        self.T_t = T_t
        self.T_p = T_p
        self.R_g = R_g
        self.K_p = K_p
        self.K_t = K_t

    def f(self, x):
        return torch.tensor([[-1 / self.T_g, 0, 1 / (self.R_g * self.T_g)],
                             [self.K_t / self.T_t, -1 / self.T_t, 0],
                             [0, self.K_p / self.T_p, -1 / self.T_p]]) @ x

    def g(self, x):
        return torch.tensor([1 / self.T_g,
                             0.,
                             0.])

    def q(self, x):
        return torch.linalg.norm(x.float()).item()

    def r(self, x):
        return 0.5

    def reset_state(self):
        return torch.tensor([0.])
