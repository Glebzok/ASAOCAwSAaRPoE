import matplotlib.pyplot as plt

from environment import VanDerPolOscillator, PowerPlantSystem
from actor import VanDerPolOscillatorActor, PowerPlantSystemActor
from critic import VanDerPolOscillatorCritic, PowerPlantSystemCritic
from algorithm import AdaOptControl


if __name__ == '__main__':
    env = VanDerPolOscillator(0.1)
    actor = VanDerPolOscillatorActor()
    critic = VanDerPolOscillatorCritic()

    # env = VanDerPolOscillator(0.1)
    # actor = VanDerPolOscillatorActor()
    # critic = VanDerPolOscillatorCritic()

    x_history, u_history = AdaOptControl(env, actor, critic).propogate(t_min=0, t_max=20, h=0.1)

    plt.figure()
    plt.scatter([i[0] for i in x_history], [i[1] for i in x_history])
    plt.show()

    plt.figure()
    plt.plot(u_history)
    plt.show()

    print(critic.W)
