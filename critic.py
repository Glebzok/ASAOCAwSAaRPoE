import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


class Critic(nn.Module):
    class RHS(nn.Module):
        def __init__(self, alpha, omega_t, q_t, r_s_t, omega, q, r_s):
            super().__init__()
            self.alpha = alpha
            self.omega_t = omega_t
            self.q_t = q_t
            self.r_s_t = r_s_t
            self.omega = omega
            self.q = q
            self.r_s = r_s

        def forward(self, t, W):
            W_dot = - self.alpha \
                    * (self.omega_t @ (self.omega_t.T @ W + self.r_s_t + self.q_t)) \
                    / (self.omega_t.T @ self.omega_t + 1) ** 2

            # print('w', W_dot)

            for omega_i, q_i, r_s_i in zip(self.omega, self.q, self.r_s):
                W_dot -= self.alpha \
                         * (omega_i @ (omega_i.T @ W + q_i + r_s_i)) \
                         / (omega_i.T @ omega_i + 1) ** 2

            # print('w', W_dot)

            return W_dot

    def __init__(self, N, alpha, integration_method):
        super().__init__()
        self.alpha = alpha
        self.N = N
        self.integration_method = integration_method
        self.W = torch.rand((N, 1))# / 100
        self.omega = []
        self.q = []
        self.r_s = []

        self.is_full = False

    def phi(self, x):
        raise NotImplementedError

    def phi_grad(self, x):
        raise NotImplementedError

    def value(self, x):
        return self.W.T @ self.phi(x)

    def calc_omega(self, env, x, u):
        return self.phi_grad(x) @ (env.f(x) + env.g(x) @ u)

    def update(self, env, x, u, h):
        omega_t = self.calc_omega(env=env, x=x, u=u)
        q_t = env.q(x)
        r_s_t = env.r_s(u)

        # print(h)
        # print(omega_t, q_t, r_s_t)
        # print(self.W)
        self.W = odeint(func=self.RHS(alpha=self.alpha, omega_t=omega_t, q_t=q_t, r_s_t=r_s_t,
                                      omega=self.omega, q=self.q, r_s=self.r_s),
                        y0=self.W, t=torch.tensor([0., h]), method=self.integration_method)[-1]

        return omega_t, q_t, r_s_t

    def update_stack(self, omega_t, q_t, r_s_t):
        if (len(self.omega) > 0) & (not self.is_full):
            if torch.matrix_rank(torch.hstack(self.omega)).item() >= self.N:
                self.is_full = True

        if not self.is_full:
            self.omega.append(omega_t)
            self.q.append(q_t)
            self.r_s.append(r_s_t)


class VanDerPolOscillatorCritic(Critic):
    def __init__(self, integration_method, alpha=10):
        super().__init__(N=3, alpha=alpha, integration_method=integration_method)
        # self.W = torch.tensor([2.4953, 0.9991, 2.2225]).view(-1, 1)

    def phi(self, x):
        return torch.tensor([x[0]**2, x[0] * x[1], x[1]**2]).T

    def phi_grad(self, x):
        return torch.tensor([[2*x[0], x[1],         0],
                             [0,      x[0], 2 * x[1]]]).T


class PowerPlantSystemCritic(Critic):
    def __init__(self, integration_method, alpha=10):
        super().__init__(N=6, alpha=alpha, integration_method=integration_method)

    def phi(self, x):
        return torch.tensor([x[0] ** 2, x[0] * x[1], x[0] * x[2], x[1] ** 2, x[1] * x[2], x[2] ** 2]).T

    def phi_grad(self, x):
        return torch.tensor([[2 * x[0], x[1], x[2], 0, 0, 0],
                             [0, x[0], 0, 2 * x[1], x[2], 0],
                             [0, 0, x[0], 0, x[1], 2 * x[2]]]).T
