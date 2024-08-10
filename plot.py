import seaborn as sns
import numpy as np
import utils
import config as c
from plots.video import record_video
from plots.scores import display_model, display_sampling
from plots.dynamics import plot_dynamics

sns.set_theme(style='darkgrid', font_scale=3)


def main():
    width = 5

    # Parse arguments
    options = utils.get_plot_options()

    # Load log
    log = np.load('simulation/log_' + c.log_name + '.npz')

    # Choose plot to display
    if options.video:
        record_video(log, width)
    elif options.scores:
        logs_cont = ('simulation/log_c00.npz', 'simulation/log_c10.npz',
                     'simulation/log_c20.npz', 'simulation/log_c30.npz',
                     'simulation/log_c40.npz', 'simulation/log_c50.npz',
                     'simulation/log_c60.npz', 'simulation/log_c70.npz',
                     'simulation/log_c80.npz')
        logs_hybrid = ('simulation/log_h00.npz', 'simulation/log_h10.npz',
                       'simulation/log_h20.npz', 'simulation/log_h30.npz',
                       'simulation/log_h40.npz', 'simulation/log_h50.npz',
                       'simulation/log_h60.npz', 'simulation/log_h70.npz',
                       'simulation/log_h80.npz')
        display_model(logs_cont, logs_hybrid)

        # logs_hybrid = ('simulation/log_w15.npz', 'simulation/log_w30.npz',
        #                'simulation/log_w50.npz', 'simulation/log_w80.npz',
        #                'simulation/log_w100.npz', 'simulation/log_w120.npz',
        #                'simulation/log_w140.npz', 'simulation/log_w160.npz',
        #                'simulation/log_w180.npz', 'simulation/log_w200.npz',)
        # display_sampling(logs_hybrid)
    else:
        plot_dynamics(log, width)


if __name__ == '__main__':
    main()
