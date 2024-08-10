import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import config as c


def display_model(logs_cont, logs_hybrid):
    models = ['continuous', 'hybrid']
    vels = [0, 10, 20, 30, 40, 50, 60, 70, 80]

    # Plot accuracy
    score = {'Model': [models[0]] * 9 + [models[1]] * 9,
             'Vel': vels * 2, 'Acc': []}

    for log_model in logs_cont + logs_hybrid:
        log = np.load(log_model)

        success = np.count_nonzero(log['success'])
        score['Acc'].append(success * 100 / c.n_trials)

    fig, axs = plt.subplots(1, 2, num='Model', figsize=(30, 12))

    axs[0].set_xlabel('Target velocity')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_ylim(0, 100)
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))

    sns.lineplot(x='Vel', y='Acc', ax=axs[0], hue='Model',
                 data=score, palette=('b', 'r'), lw=6)
    sns.scatterplot(x='Vel', y='Acc', hue='Model', s=800,
                    data=score, ax=axs[0], legend=False)

    score = {'Model': [], 'Vel': [], 'Time': []}

    for log_model, model in zip([logs_cont, logs_hybrid], models):
        for log_vel, vel in zip(log_model, vels):
            log = np.load(log_vel)

            for succ in log['success']:
                reached = np.where(succ == 1)
                if reached[0].size > 0:
                    score['Model'].append(model)
                    score['Vel'].append(vel)
                    score['Time'].append(reached[0][0])

    axs[1].set_xlabel('Target velocity')
    axs[1].set_ylabel('Time (t)')
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))

    sns.lineplot(x='Vel', y='Time', ax=axs[1], hue='Model',
                 data=score, palette=('b', 'r'), lw=6)

    plt.tight_layout()
    plt.savefig('plots/plot_model', bbox_inches='tight')
    plt.close()


def display_sampling(logs_hybrid):
    samplings = [15, 30, 50, 80, 100, 120, 140, 160, 180, 200]

    score = {'Sampling': samplings, 'Acc': []}

    # Plot accuracy
    for log_sampling in logs_hybrid:
        log = np.load(log_sampling)

        success = np.count_nonzero(log['success'])
        score['Acc'].append(success * 100 / c.n_trials)

    fig, axs = plt.subplots(1, 2, num='Sampling', figsize=(30, 12))

    axs[0].set_xlabel('Sampling time')
    axs[0].set_ylabel('Accuracy (%)')
    axs[0].set_ylim(0, 100)
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(4))

    sns.barplot(x='Sampling', y='Acc', ax=axs[0], data=score,
                order=score['Sampling'], hue='Sampling', legend=False)

    # Plot time
    score = {'Sampling': [], 'Time': []}

    for log_sampling, sampling in zip(logs_hybrid, samplings):
        log = np.load(log_sampling)

        for succ, sim_time in zip(log['success'], log['sim_time']):
            reached = np.where(succ == 1)
            if reached[0].size == 0:
                score['Sampling'].append(sampling)
                score['Time'].append(sim_time)

    axs[1].set_xlabel('Sampling time')
    axs[1].set_ylabel('Simulation time (s)')
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(4))

    sns.barplot(x='Sampling', y='Time', ax=axs[1], data=score,
                order=score['Sampling'], hue='Sampling', legend=False)

    plt.tight_layout()
    plt.savefig('plots/plot_sampling', bbox_inches='tight')
    plt.close()
