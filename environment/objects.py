import numpy as np
import utils
import config as c
from environment.sprites import Circle, Wall, Home


# Define objects class
class Objects:
    def __init__(self, batch, space, kinematics, limits):
        self.kinematics = kinematics
        self.limits = limits.T
        self.target, self.home, self.walls = self.setup(batch, space)

        # Set collision type
        self.target.shape.collision_type = 3

    def setup(self, batch, space):
        target = Circle(batch, space, c.target_size, (200, 100, 0))
        home = Home(batch, space)

        walls = []
        corners = [(0, 0), (0, c.height), (c.width, c.height),
                   (c.width, 0), (0, 0)]
        for i in range(len(corners) - 1):
            walls.append(Wall(space, corners[i], corners[i + 1]))

        return target, home, walls

    def sample(self):
        pos = []

        # Reset target and obstacle
        for o, obj in enumerate((self.target, self.home)):
            joint = utils.denormalize(np.random.rand(c.n_joints) * 2 - 1,
                                      self.limits)

            obj.set_pos(self.kinematics(joint)[3])
            pos.append(self.kinematics(joint)[3])

        c.target_size = np.random.uniform(*c.target_min_max)
        self.target.set_radius(c.target_size)

        # Set velocities
        target_dir = np.random.rand() * 2 * np.pi
        self.target.set_vel(np.cos(target_dir), np.sin(target_dir),
                            c.target_vel)

        return pos
