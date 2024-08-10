import numpy as np
import config as c
from environment.sprites import Origin, Joint, Link, Grasp


# Define arm class
class Arm:
    def __init__(self, batch, space):
        # Initialize arm parameters
        ids = {joint: j for j, joint in enumerate(c.joints)}

        self.start = np.zeros(c.n_joints)
        self.limits = np.zeros((c.n_joints, 2))
        self.size = np.zeros((c.n_joints, 2))

        self.idxs = {}

        for joint in c.joints:
            self.start[ids[joint]] = c.joints[joint]['angle']
            self.limits[ids[joint]] = c.joints[joint]['limit']
            self.size[ids[joint]] = c.joints[joint]['size']

            if c.joints[joint]['link']:
                self.idxs[ids[joint]] = ids[c.joints[joint]['link']]
            else:
                self.idxs[ids[joint]] = -1

        self.joints, self.links, self.grasp = self.setup(batch, space)

        # Set collision types
        for j in range(4):
            self.links[-4 + j].shape.collision_type = j % 2 + 1

    def setup(self, batch, space):
        # Initialize origin
        origin = Origin(space)
        joint = Joint(batch, space, 30, origin.body)

        links = []
        for j in range(c.n_joints):
            # Add link
            idx = self.idxs[j]
            pin = origin.body if j == 0 else links[idx].body
            v = (0, 0) if j == 0 else (links[idx].width, 0)
            link = Link(batch, space, self.size[j], pin, v, self.limits[j])
            links.append(link)

        # Initialize grasp sprite
        v = (links[3].width + c.grasp_dist, 0)
        grasp = Grasp(batch, space, links[3].body, v)

        return [joint], links, grasp

    def update(self, action):
        for j in range(c.n_joints):
            self.links[j].motor.rate = -action[j]

    def get_angles(self):
        angles = np.zeros(c.n_joints)

        for l, link in enumerate(self.links):
            angles[l] = -link.rotation - np.sum(angles[:min(l, -4)])

        angles[-2] -= angles[-4]
        angles[-1] -= angles[-3]

        return angles

    # Compute pose of every link
    def kinematics(self, angles, lengths=None, grasp=c.grasp_dist):
        if lengths is None:
            lengths = self.size[:, 0].copy()

        poses = np.zeros((c.n_joints + 1, 3))

        for j in range(c.n_joints):
            old_pose = poses[self.idxs[j] + 1]
            length = lengths[j] + grasp if j == 3 else lengths[j]
            poses[j + 1] = self.forward(angles[j], length, old_pose)

        return poses[1:, :2]

    # Kinematic forward pass
    def forward(self, theta, length, pose):
        position, phi = pose[:2], pose[2]
        new_phi = theta + phi

        direction = np.array([np.cos(np.radians(new_phi)),
                              np.sin(np.radians(new_phi))])
        new_position = position + length * direction

        return *new_position, new_phi

    # Kinematic inverse pass
    def inverse(self, phi, length):
        return np.array([-length * np.sin(np.radians(phi)),
                         length * np.cos(np.radians(phi)), 1])
