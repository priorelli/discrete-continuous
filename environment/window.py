import numpy as np
import pyglet
import pymunk
import utils
import config as c
from environment.arm import Arm
from environment.objects import Objects


class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Pick and Place', vsync=False)
        # Start physics engine
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)

        # Initialize arm
        self.arm = Arm(self.batch, self.space)

        # Initialize objects
        self.objects = Objects(self.batch, self.space,
                               self.arm.kinematics, self.arm.limits)

        # Initialize agent
        self.agent = None

        # Initialize simulation variables
        self.step, self.trial = 0, 0
        self.picked, self.touch = False, np.zeros(2)

        # Start collision handlers
        self.handlers = []
        for i in range(2):
            handler = self.space.add_collision_handler(i + 1, 3)
            handler.begin = self.begin(i)
            handler.separate = self.separate(i)
            self.handlers.append(handler)

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    def begin(self, i):
        def f(arbiter, space, data):
            self.touch[i] = 1
            return True
        return f

    def separate(self, i):
        def f(arbiter, space, data):
            self.touch[i] = 0
        return f

    # Update sprites rotation and position
    def update_sprites(self):
        sprites = [self.arm.grasp, self.objects.target,
                   self.objects.home]
        for sprite in self.arm.links + sprites:
            sprite.position = sprite.body.position
            sprite.rotation = -np.degrees(sprite.body.angle)

    # Get visual observation
    def get_visual_obs(self):
        grasp_norm = utils.normalize(self.arm.grasp.get_pos(), c.norm_cart)
        # wrist_norm = utils.normalize(self.arm.links[3].get_pos(), c.norm_cart)
        target_norm = utils.normalize(self.objects.target.get_pos(),
                                      c.norm_cart)
        home_norm = utils.normalize(self.objects.home.get_pos(), c.norm_cart)

        return np.array((grasp_norm, target_norm, home_norm))

    # Get proprioceptive observation
    def get_prop_obs(self):
        angles_noise = utils.add_gaussian_noise(
            self.arm.get_angles(), c.w_p)

        return utils.normalize(angles_noise, c.norm_polar)

    # Get dimension observation
    def get_dim_obs(self):
        return np.array([self.objects.target.radius, self.objects.home.width])

    # Get tactile observation
    def get_tactile_obs(self, disc=False):
        if disc:
            return np.array([not self.touch.all(),
                             self.touch.all()]).astype(int)
        return self.touch

    # Check if target is picked
    def target_picked(self):
        if np.all(self.touch):
            self.picked = True

        if np.all(self.touch):
            self.objects.target.color = (100, 100, 100)
        else:
            self.objects.target.color = (200, 100, 0)

        if (self.picked and self.home_reached() and self.target_reached()
                and self.objects.target.get_vel() < 70.0):
            self.objects.target.set_vel(0, 0)

    # Check if target is reached
    def target_reached(self):
        dist = np.linalg.norm(self.arm.grasp.get_pos() -
                              self.objects.target.get_pos())

        return dist < c.reach_dist

    # Check if home is reached
    def home_reached(self):
        dist = np.linalg.norm(self.objects.home.get_pos() -
                              self.objects.target.get_pos())

        return dist < c.home_size / 2 + 15

    # Check if task is successful
    def task_done(self):
        return (self.picked and not np.all(self.touch) and
                self.home_reached() and self.target_reached())
