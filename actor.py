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

    def __init__(self, N, m, alpha, integration_method, policy='adaptive'):
        super().__init__()
        self.alpha = alpha
        self.integration_method = integration_method
        self.policy = policy
        self.W = torch.rand((N, m)) / 100

    def phi(self, x):
        raise NotImplementedError

    def act(self, x):
        return self.W.T @ self.phi(x)

    def update(self, x, env, critic, h):
        theta = env.theta(0.5 * 1 / env.r(x) * env.g(x).T @ critic.phi_grad(x).T @ critic.W)
        phi_u = self.phi(x)

        self.W = odeint(func=self.RHS(alpha=self.alpha, phi_u=phi_u, theta=theta),
                        y0=self.W, t=torch.tensor([0., h]), method=self.integration_method)[-1]


class VanDerPolOscillatorActor(Actor):
    def __init__(self, integration_method, alpha=2, policy='adaptive'):
        super().__init__(N=3, m=1, alpha=alpha, integration_method=integration_method, policy=policy)

    def act(self, x):
        if self.policy == 'optimal':
            u = torch.tensor(-x[0] * x[1]).view(-1, 1)
        elif self.policy == 'zero':
            u = torch.tensor(0.).view(-1, 1)
        elif self.policy == 'adaptive':
            u = super().act(x)
        else:
            raise ValueError('Wrong policy type')
        return u

    def phi(self, x):
        return torch.tensor([x[0] ** 2, x[0] * x[1], x[1] ** 2]).view(-1, 1)


class PowerPlantSystemActor(Actor):
    def __init__(self, integration_method, alpha=2):
        super().__init__(N=6, m=1, alpha=alpha, integration_method=integration_method)

    def phi(self, x):
        return torch.tensor([x[0] ** 2, x[0] * x[1], x[0] * x[2], x[1] ** 2, x[1] * x[2], x[2] ** 2]).view(-1, 1)
