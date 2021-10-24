import matplotlib.pyplot as plt

from environment import VanDerPolOscillator, PowerPlantSystem
from actor import VanDerPolOscillatorActor, PowerPlantSystemActor
from critic import VanDerPolOscillatorCritic, PowerPlantSystemCritic
from algorithm import AdaOptControl


if __name__ == '__main__':
    integration_method = 'rk4'
    env = VanDerPolOscillator(integration_method, 1000)
    actor = VanDerPolOscillatorActor(integration_method, 10)
    critic = VanDerPolOscillatorCritic(integration_method)

    # env = PowerPlantSystem(integration_method)
    # actor = PowerPlantSystemActor(integration_method)
    # critic = PowerPlantSystemCritic(integration_method)

    t, x_history, u_history, critic_w_history = AdaOptControl(env, actor, critic).propogate(t_min=0, t_max=20, h=0.01)

    plt.rcParams.update({'font.size': 22})

    fig, ax = plt.subplots(3, 1, figsize=(15, 15))
    ax[0].scatter(x_history[:, 0], x_history[:, 1])
    ax[1].plot(t, u_history)
    ax[2].plot(t, critic_w_history)

    # fig, ax = plt.subplots(5, 1, figsize=(25, 15))
    # ax[0].plot(t, x_history[1:, 0])
    # ax[1].plot(t, x_history[1:, 1])
    # ax[2].plot(t, x_history[1:, 2])
    # ax[3].plot(t, u_history)
    # ax[4].plot(t, critic_w_history)

    fig.show()
