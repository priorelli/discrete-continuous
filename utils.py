import numpy as np
from numpy.linalg import norm
import argparse
import sys
import config as c


# Normalize categorical distribution
def norm_dist(dist):
    if dist.ndim == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c],
                                          dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))


# Sample from probability
def sample(probs):
    sample_onehot = np.random.multinomial(1, probs.squeeze())
    return np.where(sample_onehot == 1)[0][0]


# Compute stable logarithm
def log_stable(arr):
    return np.log(arr + 1e-16)


# Compute softmax function
def softmax(dist):
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output


# Compute sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Normalize data
def normalize(x, limits):
    x = np.array(x)
    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    x_norm = x_norm * 2 - 1
    return x_norm


# Denormalize data
def denormalize(x, limits):
    x = np.array(x)
    x_denorm = (x + 1) / 2
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]
    return x_denorm


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-c', '--continuous',
                        action='store_true', help='Start continuous model')
    parser.add_argument('-y', '--hybrid',
                        action='store_true', help='Start hybrid model')
    parser.add_argument('-a', '--ask-params',
                        action='store_true', help='Ask parameters')

    args = parser.parse_args()
    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')
    parser.add_argument('-d', '--dynamics',
                        action='store_true', help='Plot dynamics')
    parser.add_argument('-s', '--scores',
                        action='store_true', help='Plot scores')

    args = parser.parse_args()
    return args


# Compute score
def get_score(ground, est, mean=True):
    error = norm(ground - est, axis=2)
    f_error = error[:, -1]
    acc = (f_error < c.reach_dist) * 100

    time = []
    for ep in error:
        reached = np.where(ep < c.reach_dist)
        if reached[0].size > 0:
            time.append(reached[0][0])

    if mean:
        return np.mean(acc), np.mean(f_error), np.mean(time)
    else:
        return acc, f_error, time


# Print score
def print_score(log, time):
    # score = np.array((get_score(log.target_pos, log.grasp_pos),
    #                   get_score(log.grasp_pos, log.est_grasp_pos),
    #                   get_score(log.target_pos, log.est_target_pos)))

    print('\n' + '=' * 30)
    # print('\t\tReach\t\tGrasp\t\tTarget')
    # for m, measure in enumerate(('Acc', 'Error', 'Time')):
    #     print('{:s}\t\t{:.2f}\t\t{:.2f}\t\t{:.2f}'.format(
    #         measure, *score.T[m]))
    success = np.count_nonzero(log.success)
    print('Successful trials: {:.2f}%'.format(success * 100 / c.n_trials))
    print('Time elapsed: {:.2f}'.format(time))


# Print simulation info
def print_info(trial, log_success, step):
    success = np.count_nonzero(log_success)

    sys.stdout.write('\rTrial: {:4d}({:4d})/{:d} \t '
                     'Step: {:4d}/{:d} \t Accuracy: {:6.2f}%'
                     .format(trial + 1, int(success), c.n_trials,
                             step + 1, c.n_steps,
                             success * 100 / (trial + 1)))
    sys.stdout.flush()


# Print inference info
def print_inference(trial, step, log):
    e_p = norm(log.pos[trial, step] - log.est_pos[trial, step], axis=1)
    e_t = norm(log.target_pos[trial, step] -
               log.est_target_pos[trial, step])
    e_h = norm(log.home_pos[trial, step] -
               log.est_home_pos[trial, step])

    sys.stdout.write('\rTime: {:4d}/{:3d}'
                     '  |  Arm: {:+6.1f} {:+6.1f} {:+6.1f} {:+6.1f}'
                     '  |  Target: {:+6.1f}'
                     '  |  Home: {:+6.1f}'
                     .format(step + 1, trial + 1, *e_p[1:5], e_t, e_h))
    sys.stdout.flush()


# Print picked info
def print_picked(step, agent, touch):
    txt = '\rStep: {:4d} \t Touch: ({:.1f}, {:.1f}) '.format(step + 1, *touch)

    if agent:
        grasp_pos = denormalize(agent.mu_ext[0, 0], c.norm_cart)
        target_pos = denormalize(agent.mu_ext[0, 1], c.norm_cart)
        home_pos = denormalize(agent.mu_ext[0, 2], c.norm_cart)

        grasp_target = np.linalg.norm(grasp_pos - target_pos)
        home_target = np.linalg.norm(home_pos - target_pos)

        txt += '\t Int precisions: ({:.1f} {:.1f} {:.1f} {:.1f}) ' \
               '\t Ext precisions: ({:.1f} {:.1f}) ' \
               '\t Grasp-target: {:6.2f} \t Home-target: {:6.2f}'.format(
                   *agent.gamma_int, *agent.gamma_ext,
                   grasp_target, home_target)

    sys.stdout.write(txt)
    sys.stdout.flush()
