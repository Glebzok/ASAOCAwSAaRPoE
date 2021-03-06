import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from environment import VanDerPolOscillator, PowerPlantSystem
from actor import VanDerPolOscillatorActor, PowerPlantSystemActor
from critic import VanDerPolOscillatorCritic, PowerPlantSystemCritic
from algorithm import AdaOptControl

RANDOM_SEED = 13

if __name__ == '__main__':

    experiments = [{'env': 'VanDerPolOscillator', 'policy': 'zero'},
                   {'env': 'VanDerPolOscillator', 'policy': 'optimal'},
                   {'env': 'VanDerPolOscillator', 'policy': 'adaptive'},
                   {'env': 'PowerPlantSystem', 'policy': 'adaptive'}]

    for experiment in experiments:
        torch.manual_seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        env_name = experiment['env']
        policy = experiment['policy']

        if env_name == 'VanDerPolOscillator':
            integration_method = 'rk4'

            env = VanDerPolOscillator(integration_method, u_max=0.1)
            actor = VanDerPolOscillatorActor(integration_method, alpha=20, policy=policy)
            critic = VanDerPolOscillatorCritic(integration_method, alpha=20)

            T = 20
            h = 0.01

        elif env_name == 'PowerPlantSystem':
            integration_method = 'dopri5'

            env = PowerPlantSystem(integration_method, u_max=1)
            actor = PowerPlantSystemActor(integration_method, alpha=20)
            critic = PowerPlantSystemCritic(integration_method, alpha=20)

            T = 30
            h = 0.1

        t, x_history, u_history, critic_w_history, actor_w_history, q_history, r_s_story \
            = AdaOptControl(env, actor, critic).propagate(t_min=0, t_max=T, h=h)

        plt.rcParams.update({'font.size': 22})

        if env_name == 'VanDerPolOscillator':
            fig, ax = plt.subplots(6, 1, figsize=(10, 25))
            ax[0].scatter(x_history[:, 0], x_history[:, 1])
            ax[0].set_xlim(-0.5, 1)
            ax[0].set_ylim(-1, 1)
            ax[0].set_xticks([-0.5, 0, 0.5, 1])
            ax[0].set_yticks([-1, -0.5, 0, 0.5, 1])
            ax[0].grid()
            ax[0].set_title("%s policy\n"
                            "Cumulative reward: %.0f" % (policy, sum(q_history)))
            ax[0].set_xlabel('$x_1(t)$')
            ax[0].set_ylabel('$x_2(t)$')

            ax[1].plot(t, u_history)
            ax[1].plot(t, env.u_max * torch.ones_like(t), ls='--', c='r')
            ax[1].plot(t, -env.u_max * torch.ones_like(t), ls='--', c='r')
            ax[1].grid()
            ax[1].set_xlabel('Time(s)')
            ax[1].set_ylabel('u')

            ax[2].plot(t, critic_w_history)
            ax[2].set_xlabel('Time(s)')
            ax[2].set_ylabel('NN critic Parameters')

            ax[3].plot(t, actor_w_history)
            ax[3].set_xlabel('Time(s)')
            ax[3].set_ylabel('NN actor Parameters')

            ax[4].plot(t, q_history)
            ax[4].set_xlabel('Time(s)')
            ax[4].set_ylabel('Q')

            ax[5].plot(t, r_s_story)
            ax[5].set_xlabel('Time(s)')
            ax[5].set_ylabel('$R_s$')

        elif env_name == 'PowerPlantSystem':
            fig, ax = plt.subplots(8, 1, figsize=(15, 40))
            ax[0].plot(t, x_history[1:, 0])
            ax[0].grid()
            ax[0].set_ylabel('$?? ??$')

            ax[1].plot(t, x_history[1:, 1])
            ax[1].grid()
            ax[1].set_ylabel('$?? P_m$')

            ax[2].plot(t, x_history[1:, 2])
            ax[2].grid()
            ax[2].set_ylabel('$?? f_G$')
            ax[2].set_xlabel('Time(s)')

            ax[3].plot(t, u_history)
            ax[3].plot(t, env.u_max * torch.ones_like(t), ls='--', c='r')
            ax[3].plot(t, -env.u_max * torch.ones_like(t), ls='--', c='r')
            ax[3].set_ylabel('u')
            ax[3].set_xlabel('Time(s)')

            ax[4].plot(t, critic_w_history)
            ax[4].set_xlabel('Time(s)')
            ax[4].set_ylabel('NN critic Parameters')

            ax[5].plot(t, actor_w_history)
            ax[5].set_xlabel('Time(s)')
            ax[5].set_ylabel('NN actor Parameters')

            ax[6].plot(t, q_history)
            ax[6].set_xlabel('Time(s)')
            ax[6].set_ylabel('Q')

            ax[7].plot(t, r_s_story)
            ax[7].set_xlabel('Time(s)')
            ax[7].set_ylabel('$R_s$')

        plt.tight_layout()
        plt.savefig('{}_{}.png'.format(env_name, policy))
