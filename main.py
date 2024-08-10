import utils
import config as c
from simulation.manual_control import ManualControl
from simulation.inference_cont import InferenceContinuous
from simulation.inference_hybrid import InferenceHybrid


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        sim = ManualControl()

    elif options.hybrid:
        c.target_vel = 10
        c.n_tau = 50
        c.log_name = ''
        c.k_int = 0.25
        c.k_ext = 0.15
        c.pi_eta_int = 0.01
        c.pi_eta_ext = 0.01
        c.pi_x_int = 0.06
        c.pi_x_ext = 0.06

        sim = InferenceHybrid()

    elif options.ask_params:
        print('Choose model:')
        print('0 --> continuous')
        print('1 --> mixed')
        model = input('Model: ')

        print('\nChoose target velocity:')
        print('0 --> static')
        print('1 --> slow')
        print('2 --> medium')
        print('3 --> high')
        velocity = input('Velocity: ')

        c.target_vel = 0 if velocity == '0' else 10 if velocity == '1' \
            else 25 if velocity == '2' else 50

        if model == '0':
            c.k_int = 0.1
            c.k_ext = 0.06
            c.pi_eta_int = 1.0
            c.pi_eta_ext = 1.0
            c.pi_x_int = 1.0
            c.pi_x_ext = 0.2

            sim = InferenceContinuous()
        else:
            c.k_int = 0.25
            c.k_ext = 0.15
            c.pi_eta_int = 0.01
            c.pi_eta_ext = 0.01
            c.pi_x_int = 0.06
            c.pi_x_ext = 0.06

            sim = InferenceHybrid()

    else:
        c.target_vel = 10
        c.log_name = ''
        c.k_int = 0.1
        c.k_ext = 0.06
        c.pi_eta_int = 1.0
        c.pi_eta_ext = 1.0
        c.pi_x_int = 1.0
        c.pi_x_ext = 0.2

        sim = InferenceContinuous()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
