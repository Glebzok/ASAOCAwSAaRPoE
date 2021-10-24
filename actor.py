import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class Actor(nn.Module):
    class RHS(nn.Module):
        def __init__(self, alpha, phi_u, theta):
            super().__init__()
            self.alpha = alpha
            self.phi_u = phi_u
            self.theta = theta

        def forward(self, t, W):
            return -self.alpha * self.phi_u @ (W.T @ self.phi_u + self.theta).T

    def __init__(self, N, m, alpha, integration_method):
        super().__init__()
        self.alpha = alpha
        self.integration_method = integration_method
        self.W = torch.rand((N, m)) / 1000000

    def phi(self, x):
        raise NotImplementedError

    def act(self, x):
        # print('act', x.shape, self.W.T.shape, self.phi(x).shape)
        return self.W.T @ self.phi(x)

    def update(self, x, env, critic, h):
        theta = env.theta(0.5 * 1 / env.r(x) * env.g(x).T @ critic.phi_grad(x).T @ critic.W)
        phi_u = self.phi(x)

        self.W = odeint(self.RHS(self.alpha, phi_u, theta), self.W, torch.tensor([0., h]), method=self.integration_method)[-1]


class VanDerPolOscillatorActor(Actor):
    def __init__(self, integration_method, N=2, m=1, alpha=2):
        super().__init__(N=N, m=m, alpha=alpha, integration_method=integration_method)

    def phi(self, x):
        # return torch.tensor([2 * x[0], x[1], x[0], 2 * x[1]]).view(-1, 1)
        return torch.tensor([x[0], x[1]]).view(-1, 1)
        # return torch.tensor([x[0] ** 2, x[0] * x[1], x[1] ** 2]).view(-1, 1)

class PowerPlantSystemActor(Actor):
    def __init__(self, integration_method, N=18, m=1, alpha=2):
        super().__init__(N=N, m=m, alpha=alpha, integration_method=integration_method)

    def phi(self, x):
        return torch.tensor([[2 * x[0], x[1], x[2], 0, 0, 0],
                             [0, x[0], 0, 2 * x[1], x[2], 0],
                             [0, 0, x[0], 0, x[1], 2 * x[2]]]).view(-1, 1)
