import numpy as np
import pyglet
import pymunk
import config as c

offset = pymunk.Vec2d(c.width / 2 + c.off_x, c.height / 2 + c.off_y)


class Circle(pyglet.shapes.Circle):
    def __init__(self, batch, space, radius, color, pos=(0, 0)):
        super().__init__(*pos, radius, color=color, batch=batch,
                         group=pyglet.graphics.Group(1))
        self.body = pymunk.Body()
        self.body.position = pos

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 2
        self.shape.friction = 1
        self.shape.elasticity = 1

        self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                               categories=0b00001)

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def set_pos(self, pos):
        self.body.position = pymunk.Vec2d(*pos) + offset

    def set_vel(self, x, y, w=0):
        self.body.velocity = pymunk.Vec2d(x, y) * w

    def get_vel(self):
        return np.linalg.norm(self.body.velocity)

    def set_radius(self, radius):
        self.radius = radius
        self.body.radius = radius
        self.shape.unsafe_set_radius(radius)

    def set_collision(self, mask):
        if mask:
            self.shape.filter = pymunk.ShapeFilter(mask=0b01110,
                                                   categories=0b00010)
        else:
            self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                                   categories=0b00001)


class Joint(pyglet.shapes.Circle):
    def __init__(self, batch, space, radius, pin):
        super().__init__(*pin.position, radius, color=(0, 100, 200),
                         batch=batch, group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = pin.position

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 2
        self.shape.friction = 1
        self.shape.elasticity = 1

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b01110,
                                               categories=0b00010)

        space.add(pymunk.PinJoint(pin, self.body, (0, 0)))

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)


class Grasp(pyglet.shapes.Circle):
    def __init__(self, batch, space, pin, v):
        super().__init__(*(pin.position + v), c.reach_dist / 2,
                         color=(0, 100, 200), batch=batch,
                         group=pyglet.graphics.Group(0))
        self.opacity = 50
        self.body = pymunk.Body()
        self.body.position = pin.position + v

        self.shape = pymunk.Circle(self.body, c.reach_dist / 2)
        self.shape.density = 1

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b11000,
                                               categories=0b00001)

        space.add(pymunk.PinJoint(pin, self.body, v))

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)


class Link(pyglet.shapes.Rectangle):
    def __init__(self, batch, space, size, pin, v, limit):
        super().__init__(*(pin.position + v), *size, (0, 100, 200), batch,
                         group=pyglet.graphics.Group(2))
        self.body = pymunk.Body()
        self.body.position = pin.position + v

        w, h = size
        self.shape = pymunk.Segment(self.body, (0, 0), (size[0], 0),
                                    size[1] / 2)
        self.shape.density = 2
        self.shape.friction = 1
        self.shape.elasticity = 0

        self.shape.filter = pymunk.ShapeFilter(group=1, mask=0b01110,
                                               categories=0b00010)

        self.motor = pymunk.SimpleMotor(pin, self.body, 0)
        self.motor.max_force = 2e10
        space.add(self.motor)

        space.add(pymunk.PinJoint(pin, self.body, v))

        min_, max_ = np.radians(limit[0]), np.radians(limit[1])
        space.add(pymunk.RotaryLimitJoint(pin, self.body, min_, max_))

        self.anchor_x = -h / 6
        self.anchor_y = h / 2

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def get_end(self):
        v = pymunk.Vec2d(self.width, 0)
        return self.body.local_to_world(v) - offset

    def get_local(self, other):
        return other.body.world_to_local(self.get_end() + offset)


class Home(pyglet.shapes.Rectangle):
    def __init__(self, batch, space):
        super().__init__(*offset, c.home_size, c.home_size, (200, 200, 200),
                         batch, group=pyglet.graphics.Group(0))
        self.body = pymunk.Body()
        self.body.position = offset

        self.shape = pymunk.Poly.create_box(self.body,
                                            (c.home_size, c.home_size))
        self.shape.density = 1

        self.shape.filter = pymunk.ShapeFilter(mask=0b11000,
                                               categories=0b00001)

        self.anchor_x = self.width / 2
        self.anchor_y = self.height / 2

        space.add(self.body, self.shape)

    def get_pos(self):
        return np.array(self.position - offset)

    def set_pos(self, pos):
        self.body.position = pymunk.Vec2d(*pos) + offset


class Wall:
    def __init__(self, space, a, b):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)

        self.shape = pymunk.Segment(self.body, a, b, 1)
        self.shape.elasticity = 1

        space.add(self.body, self.shape)


class Origin:
    def __init__(self, space):
        self.body = space.static_body
        self.body.position = offset
