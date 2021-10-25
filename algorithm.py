import numpy as np
import torch
from tqdm import tqdm


class AdaOptControl(object):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic

    def propagate(self, t_min, t_max, h):
        t_grid = torch.arange(start=t_min, end=t_max, step=h)

        x = self.env.reset_state()
        x_history, u_history, critic_w_history, actor_w_history, q_history, r_s_history = [
                                                                                              x.numpy().flatten()], [], [], [], [], []

        for t in tqdm(t_grid):
            u = self.actor.act(x=x)
            x = self.env.propagate(x=x, u=u, t=t, h=h)

            x_history.append(x.numpy().flatten())
            u_history.append(u.numpy().flatten())

            self.actor.update(x=x, env=self.env, critic=self.critic, h=h)
            omega_t, q_t, r_s_t = self.critic.update(env=self.env, x=x, u=u, h=h)

            critic_w_history.append(self.critic.W.numpy().flatten())
            actor_w_history.append(self.actor.W.numpy().flatten())
            q_history.append(q_t)
            r_s_history.append(r_s_t)

            if not self.critic.is_full:
                self.critic.update_stack(omega_t=omega_t, q_t=q_t, r_s_t=r_s_t)

        print('Final critic weights:\n', self.critic.W)
        print('Final actor weights:\n', self.actor.W)
        print('Cumulative reward:\n', sum(q_history))

        return t_grid, \
               np.vstack(x_history), np.vstack(u_history), \
               np.vstack(critic_w_history), np.vstack(actor_w_history), \
               q_history, r_s_history
