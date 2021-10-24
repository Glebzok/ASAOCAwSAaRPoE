import matplotlib.pyplot as plt
import torch

from environment import VanDerPolOscillator, PowerPlantSystem
from actor import VanDerPolOscillatorActor, PowerPlantSystemActor
from critic import VanDerPolOscillatorCritic, PowerPlantSystemCritic
from algorithm import AdaOptControl


if __name__ == '__main__':
    integration_method = 'rk4'

    ENV = 'VanDerPolOscillator'
    # ENV = 'PowerPlantSystem'

    if ENV == 'VanDerPolOscillator':

        env = VanDerPolOscillator(integration_method, u_max=0.1)
        actor = VanDerPolOscillatorActor(integration_method, alpha=2)
        critic = VanDerPolOscillatorCritic(integration_method, alpha=10)

        T = 20

    else:

        env = PowerPlantSystem(integration_method, u_max=1000)
        actor = PowerPlantSystemActor(integration_method, alpha=2)
        critic = PowerPlantSystemCritic(integration_method, alpha=10)

        T = 4

    t, x_history, u_history, critic_w_history, q_history, r_s_story \
        = AdaOptControl(env, actor, critic).propogate(t_min=0, t_max=T, h=0.01)

    plt.rcParams.update({'font.size': 22})

    if ENV == 'VanDerPolOscillator':
        fig, ax = plt.subplots(5, 1, figsize=(15, 25))
        ax[0].scatter(x_history[:, 0], x_history[:, 1])
        ax[1].plot(t, u_history)
        ax[1].plot(t, env.u_max * torch.ones_like(t), ls='--', c='r')
        ax[1].plot(t, -env.u_max * torch.ones_like(t), ls='--', c='r')
        ax[2].plot(t, critic_w_history)
        ax[3].plot(t, q_history)
        ax[4].plot(t, r_s_story)

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

    fig.show()
