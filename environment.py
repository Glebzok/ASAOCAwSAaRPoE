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
            # a = self.f(x)
            # b = self.g(x)
            # c = self.u
            # print(x.shape, a.shape, b.shape, c.shape)
            # print((self.f(x) + self.g(x) @ self.u).shape)
            return self.f(x) + self.g(x) @ self.u

    def __init__(self, integration_method, u_max):
        self.u_max = u_max
        self.integration_method = integration_method

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
        # print(x, self.u_max, x / self.u_max, torch.atanh(x / self.u_max))
        return self.u_max * torch.atanh(x / self.u_max)

    def r_s(self, u):
        r_s_l = []
        # print(u)
        for u_i in u:
            v_grid = torch.linspace(0, u_i.item(), 1000)
            # print(u, u_i, v_grid)
            # print(u_i)
            # print(u_grid)
            theta_grid = self.theta_inv(v_grid) * self.r(v_grid)
            r_s_l.append(torch.trapz(theta_grid, v_grid))
            # print('r_s', u_grid, theta_grid, r_s_l[-1])
        # print(r_s_l)
        return 2 * sum(r_s_l)

    def propagate(self, x, u, t, h):
        u = u.clone()
        if t <= 1:
            u += 0.5 * (torch.sin(0.3 * torch.pi * t) + torch.cos(0.3 * torch.pi * t))
        x_h = odeint(self.RHS(self.f, self.g, u), x, torch.tensor([0., h]), method=self.integration_method)[-1]
        return x_h

    def reset_state(self):
        raise NotImplementedError()


class VanDerPolOscillator(Environment):
    def __init__(self, integration_method, u_max=0.1):
        super().__init__(integration_method, u_max)

    def f(self, x):
        return torch.tensor([x[1],
                             -x[0] - 0.5 * x[1] * (1 - x[0]**2) - x[0]**2 * x[1]]).view(-1, 1)

    def g(self, x):
        return torch.tensor([0, x[0]]).view(-1, 1)

    def q(self, x):
        return torch.linalg.norm(x.float()).item()**2

    def r(self, x):
        return 1

    def reset_state(self):
        return torch.tensor([1., -1.]).view(-1, 1)


class PowerPlantSystem(Environment):
    def __init__(self, integration_method, u_max=0.02, T_g=0.08, T_t=0.1, T_p=20,
                 R_g=2.5, K_p=120, K_t=1):
        super().__init__(integration_method, u_max)

        self.T_g = T_g
        self.T_t = T_t
        self.T_p = T_p
        self.R_g = R_g
        self.K_p = K_p
        self.K_t = K_t

    def f(self, x):
        return (torch.tensor([[      -1 / self.T_g,                   0, 1 / (self.R_g * self.T_g)],
                             [self.K_t / self.T_t,       -1 / self.T_t,                         0],
                             [                  0, self.K_p / self.T_p,             -1 / self.T_p]]) @ x).view(-1, 1)

    def g(self, x):
        return torch.tensor([1 / self.T_g,
                             0.,
                             0.]).view(-1, 1)

    def q(self, x):
        return torch.linalg.norm(x.float()).item()

    def r(self, x):
        return 0.5

    def reset_state(self):
        return torch.tensor([0., 0., 0.]).view(-1, 1)