import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from numpy.linalg import norm
from matplotlib.gridspec import GridSpec
from pylab import tight_layout
import time
import sys
import config as c
import matplotlib.animation as animation


def record_video(log, width):
    plot_type = 0
    frame = c.n_steps - 1
    trial = 58
    dynamics = False

    # Initialize arm
    idxs = {}
    ids = {joint: j for j, joint in enumerate(c.joints)}
    size = np.zeros((c.n_joints, 2))
    for joint in c.joints:
        size[ids[joint]] = c.joints[joint]['size']
        if c.joints[joint]['link']:
            idxs[ids[joint]] = ids[c.joints[joint]['link']]
        else:
            idxs[ids[joint]] = -1

    # Load variables
    n_t = log['angles'].shape[0] * log['angles'].shape[1]

    angles = log['angles'].reshape(n_t, c.n_joints)
    est_angles = log['est_angles'].reshape(n_t, c.n_joints)

    target_pos = log['target_pos'].reshape(n_t, 2)
    est_target_pos = log['est_target_pos'].reshape(n_t, 2)

    home_pos = log['home_pos'].reshape(n_t, 2)
    est_home_pos = log['est_home_pos'].reshape(n_t, 2)

    pos = log['pos'].reshape(n_t, c.n_joints + 1, 2)
    est_pos = log['est_pos'].reshape(n_t, c.n_joints + 1, 2)

    grasp_pos = log['grasp_pos'].reshape(n_t, 2)
    est_grasp_pos = log['est_grasp_pos'].reshape(n_t, 2)

    # a = log['a'].reshape(n_t, c.n_joints)

    picked = log['picked'].reshape(n_t)
    target_size = log['target_size'].reshape(n_t)

    # error_h = norm(hand_pos - c.home_pos, axis=1)

    # Create plot
    x_range = (-c.width / 2 - c.off_x, c.width / 2 - c.off_x)
    y_range = (-c.height / 2 - c.off_y, c.height / 2 - c.off_y)

    if dynamics:
        fig = plt.figure(figsize=(40, (y_range[1] - y_range[0]) * 20 /
                                  (x_range[1] - x_range[0])))
        gs = GridSpec(2, 3, figure=fig)

        axs = [fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[1, 1])]
    else:
        fig = plt.figure(figsize=(20, (y_range[1] - y_range[0]) * 20 /
                                  (x_range[1] - x_range[0])))
        gs = GridSpec(1, 1, figure=fig)

        axs = [fig.add_subplot(gs[:, 0])]

    xlims = [x_range, (0, c.n_steps), (0, c.n_steps), (0, c.n_steps)]
    ylims = [y_range, (-0.1, 20), (-0.1, 3), (-0.1, 5)]
    titles = ['', '', '']

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rTrial: {:d} \tStep: {:d}'
                             .format(int(n / c.n_steps) + 1,
                                     (n % c.n_steps) + 1))
            sys.stdout.flush()

        # Clear plot
        n_axs = len(axs)
        for w, xlim, ylim, title in zip(range(n_axs), xlims, ylims, titles):
            axs[w].clear()
            axs[w].set_xlim(xlim)
            axs[w].set_ylim(ylim)
            # axs[w].title.set_text(title)
        axs[0].get_xaxis().set_visible(False)
        axs[0].get_yaxis().set_visible(False)
        tight_layout()

        # Draw arm
        for j in range(c.n_joints):
            axs[0].plot(*np.array([est_pos[n, idxs[j] + 1],
                                   est_pos[n, j + 1]]).T,
                        linewidth=size[j, 1] * 2.5, color='lightblue',
                        zorder=1)
            axs[0].plot(*np.array([pos[n, idxs[j] + 1], pos[n, j + 1]]).T,
                        linewidth=size[j, 1] * 2.5, color='b',
                        zorder=1)

        # Draw target
        t_size = target_size[n] * 450
        axs[0].scatter(*est_target_pos[n], color='m', s=t_size, zorder=0)
        target_cl = 'brown' if picked[n] else 'r'
        axs[0].scatter(*target_pos[n], color=target_cl, s=t_size, zorder=0)

        # Draw home position
        rect = patches.Rectangle(
            home_pos[n] - [c.home_size / 2, c.home_size / 2],
            c.home_size, c.home_size,
            color='grey', alpha=0.6, zorder=0)
        axs[0].add_patch(rect)
        rect = patches.Rectangle(
            est_home_pos[n] - [c.home_size / 2, c.home_size / 2],
            c.home_size, c.home_size,
            color='darkgrey', alpha=0.6, zorder=0)
        axs[0].add_patch(rect)

        # Draw trajectories
        axs[0].scatter(*est_target_pos[n - (n % c.n_steps): n + 1].T,
                       color='darkred', linewidth=width, zorder=2)
        axs[0].scatter(*est_home_pos[n - (n % c.n_steps): n + 1].T,
                       color='darkgrey', linewidth=width, zorder=2)
        axs[0].scatter(*est_grasp_pos[n - (n % c.n_steps): n + 1].T,
                       color='cyan', linewidth=width, zorder=2)
        axs[0].scatter(*grasp_pos[n - (n % c.n_steps): n + 1].T,
                       color='darkblue', linewidth=width, zorder=2)

    # Plot video
    if plot_type == 0:
        start = time.time()
        ani = animation.FuncAnimation(fig, animate, n_t)
        writer = animation.writers['ffmpeg'](fps=80)
        ani.save('plots/video.mp4', writer=writer)
        print('\nTime elapsed:', time.time() - start)

    # Plot frame sequence
    elif plot_type == 1:
        for i in range(0, n_t, c.n_tau):
            animate(i)
            plt.savefig('plots/frame_%d' % i)

    # Plot single frame
    elif plot_type == 2:
        animate(frame)
        plt.savefig('plots/frame_' + c.log_name, bbox_inches='tight')
