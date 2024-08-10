import time
import numpy as np
import utils
import config as c
from environment.window import Window
from simulation.hybrid_agent_cont import ContinuousAgent
from simulation.hybrid_agent_disc import DiscreteAgent
from environment.log import Log


# Define inference class
class InferenceHybrid(Window):
    def __init__(self):
        super().__init__()
        # Initialize objects
        pos = self.objects.sample()

        # Initialize agents
        self.cont_agent = ContinuousAgent(self.arm)
        self.cont_agent.init_belief(self.arm.get_angles(),
                                    self.get_dim_obs(),
                                    self.arm.grasp.get_pos(), *pos)

        self.disc_agent = DiscreteAgent(self.cont_agent.evidence)

        # Initialize error tracking
        self.log = Log()
        self.time = time.time()

    def update(self, dt):
        dt = 1 / c.fps

        # Get observations
        Y_cont = [self.get_prop_obs(), self.get_visual_obs(),
                  self.get_dim_obs()]
        o_tact = self.get_tactile_obs(disc=True)

        # Perform discrete step
        if self.step % c.n_tau == 0:
            eta = self.disc_agent.inference_step(
                self.cont_agent.evidence, o_tact)
            self.cont_agent.eta_int, self.cont_agent.eta_ext = eta

        # Perform continuous step
        action = self.cont_agent.inference_step(Y_cont, self.step)

        # Update arm
        self.arm.update(utils.add_gaussian_noise(action, c.w_a))

        # Update physics
        for i in range(c.phys_steps):
            self.space.step(c.speed / (c.fps * c.phys_steps))

        # Move sprites
        self.update_sprites()

        # Check collision type if grasping
        collide = (np.all(self.touch) and self.disc_agent.P_u[2] < 0.7) or \
                  (self.target_reached() and self.disc_agent.P_u[3] > 0.5)
        self.objects.target.set_collision(collide)

        # Check if target is picked
        self.target_picked()

        # Track log
        self.log.track(self.step, self.trial, self.cont_agent, self.arm,
                       self.objects, np.all(self.touch), self.task_done(),
                       self.disc_agent.P_u)

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, self.log.success, self.step)
            # utils.print_picked(self.step, self.agent, self.touch)
            # utils.print_inference(self.trial, self.step, self.log)

        # Reset trial
        self.step += 1
        if self.step == c.n_steps or self.task_done():
            self.log.sim_time[self.trial] = time.time() - self.time
            self.reset_trial()

    def reset_trial(self):
        # Simulation done
        if self.trial == c.n_trials - 1:
            self.log.save_log()
            self.stop()
        else:
            # Initialize simulation
            self.picked = False

            # Sample objects
            self.objects.sample()

            # Reset agents
            self.disc_agent.reset()
            self.cont_agent.a = np.zeros(c.n_joints)

            self.step = 0
            self.trial += 1
            self.time = time.time()
