import matplotlib.pyplot as plt
import torch
import random
import numpy as np

from environment import VanDerPolOscillator, PowerPlantSystem
from actor import VanDerPolOscillatorActor, PowerPlantSystemActor
from critic import VanDerPolOscillatorCritic, PowerPlantSystemCritic
from algorithm import AdaOptControl

RANDOM_SEED = 13
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


if __name__ == '__main__':
    integration_method = 'dopri5'
    # integration_method = 'rk4'
    POLICY = 'Adaptive'
    # POLICY = 'Zero'
    # POLICY = 'Optimal'

    ENV = 'VanDerPolOscillator'
    # ENV = 'PowerPlantSystem'

    if ENV == 'VanDerPolOscillator':

        env = VanDerPolOscillator(integration_method, u_max=0.1)
        actor = VanDerPolOscillatorActor(integration_method, alpha=20)
        critic = VanDerPolOscillatorCritic(integration_method, alpha=20)

        T = 20

    else:

        env = PowerPlantSystem(integration_method, u_max=1)
        actor = PowerPlantSystemActor(integration_method, alpha=20)
        critic = PowerPlantSystemCritic(integration_method, alpha=20)

        T = 30

    t, x_history, u_history, critic_w_history, actor_w_history, q_history, r_s_story \
        = AdaOptControl(env, actor, critic).propagate(t_min=0, t_max=T, h=0.1, policy=POLICY)

    plt.rcParams.update({'font.size': 22})

    if ENV == 'VanDerPolOscillator':
        fig, ax = plt.subplots(6, 1, figsize=(10, 25))
        ax[0].scatter(x_history[:, 0], x_history[:, 1])
        ax[0].set_xlim(-0.5, 1)
        ax[0].set_ylim(-1, 1)
        ax[0].set_xticks([-0.5, 0, 0.5, 1])
        ax[0].set_yticks([-1, -0.5, 0, 0.5, 1])
        ax[0].grid()
        ax[0].set_title("%s policy\n"
                        "Cumulative reward: %.0f"%(POLICY, sum(q_history)))
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

    else:
        fig, ax = plt.subplots(7, 1, figsize=(15, 35))
        ax[0].plot(t, x_history[1:, 0])
        ax[1].plot(t, x_history[1:, 1])
        ax[2].plot(t, x_history[1:, 2])
        ax[3].plot(t, u_history)
        ax[3].plot(t, env.u_max * torch.ones_like(t), ls='--', c='r')
        ax[3].plot(t, -env.u_max * torch.ones_like(t), ls='--', c='r')
        ax[4].plot(t, critic_w_history)
        ax[5].plot(t, q_history)
        ax[6].plot(t, r_s_story)

    plt.tight_layout()
    fig.show()
