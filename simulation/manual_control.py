from pyglet.window import key
import utils
import config as c
from environment.window import Window


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()
        self.objects.target.set_collision(1)

        # Initialize objects
        self.objects.sample()

    def update(self, dt):
        dt = 1 / c.fps

        # Get action from user
        action = self.get_pressed()

        # Update arm
        self.arm.update(action)

        # Update physics
        for i in range(c.phys_steps):
            self.space.step(c.speed / (c.fps * c.phys_steps))

        # Move sprites
        self.update_sprites()

        # Check if target is picked
        self.target_picked()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, 0, self.step)
            # utils.print_picked(self.step, None, self.touch)

        # Reset trial
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        self.success += self.task_done()

        # Simulation done
        if self.trial == c.n_trials - 1:
            self.stop()
        else:
            self.picked = False

            # Sample objects
            self.objects.sample()

            self.step = 0
            self.trial += 1

    # Get action from user input
    def get_pressed(self):
        return [(key.Z in self.keys) - (key.X in self.keys),
                (key.LEFT in self.keys) - (key.RIGHT in self.keys),
                (key.UP in self.keys) - (key.DOWN in self.keys),
                (key.A in self.keys) - (key.S in self.keys),
                (key.Q in self.keys) - (key.W in self.keys),
                (key.W in self.keys) - (key.Q in self.keys),
                (key.Q in self.keys) - (key.W in self.keys),
                (key.W in self.keys) - (key.Q in self.keys)]
