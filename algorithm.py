import numpy as np
import torch
from tqdm import tqdm


class AdaOptControl(object):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic

    def propogate(self, t_min, t_max, h):
        t_grid = torch.arange(start=t_min, end=t_max, step=h)

        x = self.env.reset_state()
        x_history, u_history, critic_w_history = [x.numpy().flatten()], [], []

        for t in tqdm(t_grid):
            # print('i')
            u = self.actor.act(x=x)
            # print(u)
            x = self.env.propagate(x=x, u=u, t=t, h=h)

            x_history.append(x.numpy().flatten())
            u_history.append(u.numpy().flatten())

            self.actor.update(x=x, env=self.env, critic=self.critic, h=h)
            # print('actor', self.actor.W)
            omega_t, q_t, r_s_t = self.critic.update(env=self.env, x=x, u=u, h=h)
            # print('critic', self.critic.W)

            critic_w_history.append(self.critic.W.numpy().flatten())

            if not self.critic.is_full:
                self.critic.update_stack(omega_t=omega_t, q_t=q_t, r_s_t=r_s_t)

        print(self.critic.W)
        print(self.actor.W)

        return t_grid, np.vstack(x_history), np.vstack(u_history), np.vstack(critic_w_history)
