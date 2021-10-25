import numpy as np
import torch
from tqdm import tqdm


class AdaOptControl(object):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic

    def propagate(self, t_min, t_max, h, policy):
        t_grid = torch.arange(start=t_min, end=t_max, step=h)

        for _ in range(1):

            x = self.env.reset_state()
            x_history, u_history, critic_w_history, actor_w_history, q_history, r_s_history = [x.numpy().flatten()], [], [], [], [], []

            for t in tqdm(t_grid):
                # print('i')
                if policy == 'Optimal':
                    u = torch.tensor(-x[0] * x[1]).view(-1, 1)
                elif policy == 'Zero':
                    u = torch.tensor(0.).view(-1, 1)
                else:
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
                actor_w_history.append(self.actor.W.numpy().flatten())
                q_history.append(q_t)
                r_s_history.append(r_s_t)

                if not self.critic.is_full:
                    self.critic.update_stack(omega_t=omega_t, q_t=q_t, r_s_t=r_s_t)

            print(self.critic.W)
            print(self.actor.W)
            print(sum(q_history))
            # print(sum(r_s_history))

        return t_grid, np.vstack(x_history), np.vstack(u_history), np.vstack(critic_w_history), np.vstack(actor_w_history), q_history, r_s_history
