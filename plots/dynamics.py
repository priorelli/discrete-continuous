import numpy as np
import matplotlib.pyplot as plt


def plot_dynamics(log, width):
    n = 900

    # Load variables
    probs = log['probs'][-1]

    grasp_pos = log['grasp_pos'][-1]
    target_pos = log['target_pos'][-1]

    trajectories = [(probs[:n, 0] + probs[:n, 1]), probs[:n, 2], probs[:n, 3]]

    # Plots
    e_pos = np.linalg.norm(grasp_pos - target_pos, axis=1)[:n]

    fig, axs = plt.subplots(2, 1, figsize=(18, 15))

    axs[0].plot(e_pos, lw=width)
    axs[0].set_ylabel('L2 Norm (px)')

    for traj, label, marker in zip(trajectories, ['reach', 'open', 'grasp'],
                                   ['P', 'v', '*']):
        axs[1].plot(traj, label=label, alpha=0.7, lw=width, markersize=10,
                    marker=marker, markevery=50)

    axs[1].legend(loc=0)
    axs[1].set_ylabel('Action probs')
    axs[1].set_xlabel('Time step')

    for ax in axs:
        ax.axvline(300, lw=width - 2, ls='--', color='grey')
        ax.axvline(500, lw=width - 2, ls='--', color='grey')
        ax.axvline(600, lw=width - 2, ls='--', color='grey')
        ax.axvline(800, lw=width - 2, ls='--', color='grey')

    plt.tight_layout()
    fig.savefig('plots/plot_dynamics', bbox_inches='tight')
    # plt.show()
