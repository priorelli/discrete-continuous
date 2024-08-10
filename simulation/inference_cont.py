import time
import numpy as np
import utils
import config as c
from environment.window import Window
from simulation.cont_only_agent import Agent
from environment.log import Log


# Define inference class
class InferenceContinuous(Window):
    def __init__(self):
        super().__init__()
        # Initialize objects
        pos = self.objects.sample()

        # Initialize agent
        self.agent = Agent(self.arm)
        self.agent.init_belief(self.arm.get_angles(),
                               self.get_dim_obs(),
                               self.arm.grasp.get_pos(), *pos)

        # Initialize error tracking
        self.log = Log()
        self.time = time.time()

    def update(self, dt):
        dt = 1 / c.fps

        # Get observations
        Y = [self.get_prop_obs(), self.get_visual_obs(),
             self.get_dim_obs(), self.get_tactile_obs()]

        # Perform free energy step
        action = self.agent.inference_step(Y)

        # Update arm
        self.arm.update(utils.add_gaussian_noise(action, c.w_a))

        # Update physics
        for i in range(c.phys_steps):
            self.space.step(c.speed / (c.fps * c.phys_steps))

        # Move sprites
        self.update_sprites()

        # Check collision type if grasping
        self.objects.target.set_collision(not self.agent.gamma_int[2])

        # Check if target is picked
        self.target_picked()

        # Track log
        self.log.track(self.step, self.trial, self.agent, self.arm,
                       self.objects, np.all(self.touch), self.task_done())

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

            self.agent.a = 0
            self.step = 0
            self.trial += 1
