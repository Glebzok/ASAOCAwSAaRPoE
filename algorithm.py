import torch


class AdaOptControl(object):
    def __init__(self, env, actor, critic):
        self.env = env
        self.actor = actor
        self.critic = critic

    def propogate(self, t_min, t_max, h):
        t_grid = torch.arange(t_min, t_max, h)

        x = self.env.reset_state()
        x_history, u_history = [x.clone()], []

        for t in t_grid:
            # print('i')
            u = self.actor.act(x)
            x = self.env.propogate(x, u, h)

            self.actor.update(x, self.env, self.critic, h)
            omega_t, q_t, r_s_t = self.critic.update(self.env, x, u, h)

            if not self.critic.is_full:
                self.critic.update_stack(omega_t, q_t, r_s_t)

            x_history.append(x.clone())
            u_history.append(u.clone())

        return x_history, u_history
