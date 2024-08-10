import numpy as np
import utils
import config as c


# Define agent class
class Agent:
    def __init__(self, arm):
        self.kinematics = arm.kinematics
        self.inverse = arm.inverse
        self.limits = utils.normalize(arm.limits, c.norm_polar)

        # Initialize belief and action
        n_int = (c.n_objects + 1) * c.n_joints
        n_ext = (c.n_objects + 1) * 2

        self.mu_int_x = np.zeros((c.n_orders, n_int))
        self.mu_int_v = [np.eye(n_int), np.zeros(n_int)]

        self.mu_ext_x = np.zeros((c.n_orders, n_ext))
        self.mu_ext_v = [np.eye(n_ext), np.zeros(n_ext)]

        self.mu_dim = np.zeros((c.n_orders, c.n_objects))
        self.mu_tact = np.zeros((c.n_orders, 2))
        self.mu_len = np.zeros((c.n_objects + 1, c.n_joints))

        self.a = np.zeros(c.n_joints)

        # Initialize intention matrices
        self.W_int = np.array([np.eye(n_int) for _ in range(5)])
        self.b_int = np.array([np.zeros(n_int) for _ in range(5)])
        self.W_ext = np.array([np.eye(n_ext) for _ in range(2)])
        self.b_ext = np.array([np.zeros(n_ext) for _ in range(2)])

        # Reach target
        z = np.zeros((int(c.n_joints / 2), c.n_joints))
        i = np.eye(int(c.n_joints / 2), c.n_joints)
        self.W_int[0, :int(c.n_joints / 2)] = np.block([[z, i, z]])

        # Close hand
        self.W_int[1, int(c.n_joints / 2): c.n_joints] = 0
        self.b_int[1, int(c.n_joints / 2): c.n_joints] = utils.normalize(
            [0, 0, -30, 30], c.norm_polar)

        # Open hand
        self.W_int[2, int(c.n_joints / 2): c.n_joints] = 0
        self.b_int[2, int(c.n_joints / 2): c.n_joints] = utils.normalize(
            [c.open_angle, -c.open_angle, 0, 0], c.norm_polar)

        # Reach home (int)
        self.W_int[3, :int(c.n_joints / 2)] = np.block([[z, z, i]])
        self.W_int[3, c.n_joints: c.n_joints + int(c.n_joints / 2)] = \
            np.block([[z, z, i]])

        # Reach home (ext)
        z, i = np.zeros((2, 2)), np.eye(2)
        self.W_ext[0, 2: 4] = np.block([[z, z, i]])
        self.W_ext[0, :2] = np.block([[z, z, i]])

        # Initialize intention precisions
        self.gamma_int = np.zeros(5)
        self.gamma_ext = np.zeros(2)

    # Initialize belief
    def init_belief(self, init_mu_int, init_mu_dim,
                    grasp_pos, target_pos, home_pos):
        self.mu_int_x[0, :] = np.tile(utils.normalize(
            init_mu_int, c.norm_polar), c.n_objects + 1)
        self.mu_ext_x[0] = self.g_ext()

        self.mu_dim[0] = np.array(init_mu_dim)

        self.mu_len = np.tile(utils.normalize(
            np.array(c.lengths), c.norm_cart), (c.n_objects + 1, 1))
        self.mu_len[:, 3] += utils.normalize(c.grasp_dist, c.norm_cart)
        # self.mu_len[1, 3] += utils.normalize(c.grasp_dist, c.norm_cart)

    # Get extrinsic belief
    def g_ext(self):
        mu_int_denorm = utils.denormalize(self.mu_int_x[0], c.norm_polar)
        mu_int_denorm = mu_int_denorm.reshape(c.n_objects + 1, c.n_joints)

        return np.ravel([self.kinematics(mu_int_denorm[o], self.mu_len[o], 0)
                         [3] for o in range(c.n_objects + 1)])

    # Get extrinsic gradient
    def grad_ext(self, E_ext):
        mu_int_denorm = utils.denormalize(self.mu_int_x[0], c.norm_polar)
        mu_int_denorm = mu_int_denorm.reshape(c.n_objects + 1, c.n_joints)
        E_ext_reshape = E_ext.reshape(c.n_objects + 1, 2)

        lkh_int = np.zeros((c.n_objects + 1, c.n_joints))

        for o in range(c.n_objects + 1):
            grad_int = np.zeros((c.n_joints, 2))

            for j in range(c.n_joints - 4):
                inv = self.inverse(np.sum(mu_int_denorm[o, :j + 1]),
                                   self.mu_len[o, j])[:2]
                for n in range(j + 1):
                    grad_int[n] += inv

            # Gradient of normalization
            grad_int *= (c.norm_polar[1] - c.norm_polar[0])
            grad_int *= np.pi / 180

            lkh_int[o] = E_ext_reshape[o].dot(grad_int.T)

        return lkh_int.flatten()

    # Get predictions
    def get_p(self):
        p_ext = self.g_ext()
        p_prop = self.mu_int_x[0, :c.n_joints].copy()
        p_vis = self.mu_ext_x[0].copy()
        p_dim = self.mu_dim[0].copy()
        p_tact = self.mu_tact[0].copy()

        return p_ext, p_prop, p_vis, p_dim, p_tact

    # Get sensory prediction errors
    def get_e_y(self, Y, P):
        return [y - p for y, p in zip(Y, P)]

    # Get priors
    def get_eta(self):
        eta_int_W = np.sum([gamma * W for gamma, W in zip(
            self.gamma_int, self.W_int)], axis=0)
        eta_int_b = self.gamma_int.dot(self.b_int)

        eta_ext_W = np.sum([gamma * W for gamma, W in zip(
            self.gamma_ext, self.W_ext)], axis=0)
        eta_ext_b = self.gamma_ext.dot(self.b_ext)

        return [eta_int_W, eta_int_b], [eta_ext_W, eta_ext_b]

    # Get prior errors
    def get_e_eta(self, eta_int, eta_ext):
        e_eta_int_W = self.mu_int_v[0] - eta_int[0]
        e_eta_int_b = self.mu_int_v[1] - eta_int[1]

        e_eta_ext_W = self.mu_ext_v[0] - eta_ext[0]
        e_eta_ext_b = self.mu_ext_v[1] - eta_ext[1]

        return [e_eta_int_W, e_eta_int_b], [e_eta_ext_W, e_eta_ext_b]

    # Get intentions
    def get_i(self):
        i_int = self.mu_int_x[0].dot(self.mu_int_v[0].T) + self.mu_int_v[1]
        i_ext = self.mu_ext_x[0].dot(self.mu_ext_v[0].T) + self.mu_ext_v[1]

        return i_int, i_ext

    # Get dynamics prediction errors
    def get_e_x(self, i_int, i_ext):
        e_i_int = (i_int - self.mu_int_x[0]) * c.k_int
        e_i_ext = (i_ext - self.mu_ext_x[0]) * c.k_ext

        e_x_int = self.mu_int_x[1] - e_i_int
        e_x_ext = self.mu_ext_x[1] - e_i_ext

        e_x_back_int_x = e_x_int.dot(self.mu_int_v[0].T)
        e_x_back_ext_x = e_x_ext.dot(self.mu_ext_v[0].T)

        e_x_back_int_W = e_x_int[np.newaxis, :].T.dot(
            self.mu_int_x[0][np.newaxis, :])
        e_x_back_ext_W = e_x_ext[np.newaxis, :].T.dot(
            self.mu_ext_x[0][np.newaxis, :])

        e_x_back_int_b = e_x_int.copy()
        e_x_back_ext_b = e_x_ext.copy()

        return (e_x_int, e_x_ext), (e_x_back_int_x, e_x_back_ext_x), \
            (e_x_back_int_W, e_x_back_ext_W), \
            (e_x_back_int_b, e_x_back_ext_b)

    # Get likelihood components
    def get_likelihood(self, E_g):
        lkh = {}

        lkh['int'] = c.pi_ext * self.grad_ext(E_g[0])

        lkh['prop'] = np.zeros_like(self.mu_int_x[0])
        lkh['prop'][:c.n_joints] = E_g[1] * c.pi_prop

        lkh['vis'] = E_g[2] * c.pi_vis
        lkh['dim'] = E_g[3] * c.pi_dim
        lkh['tact'] = E_g[4] * c.pi_tact

        lkh['forward_ext'] = -E_g[0] * c.pi_ext

        return lkh

    # Get belief update
    def get_mu_dot(self, lkh, e_eta_int, e_eta_ext, E_x, E_x_back_x,
                   E_x_back_W, E_x_back_b):
        mu_int_x_dot = np.zeros_like(self.mu_int_x)
        mu_int_v_dot = [np.zeros_like(self.mu_int_v[0]),
                        np.zeros_like(self.mu_int_v[1])]

        mu_ext_x_dot = np.zeros_like(self.mu_ext_x)
        mu_ext_v_dot = [np.zeros_like(self.mu_ext_v[0]),
                        np.zeros_like(self.mu_ext_v[1])]

        mu_dim_dot = np.zeros_like(self.mu_dim)
        mu_tact_dot = np.zeros_like(self.mu_tact)

        mu_int_v_dot[0] = -e_eta_int[0] * c.pi_eta_int
        mu_int_v_dot[1] = -e_eta_int[1] * c.pi_eta_int

        mu_ext_v_dot[0] = -e_eta_ext[0] * c.pi_eta_ext
        mu_ext_v_dot[1] = -e_eta_ext[1] * c.pi_eta_ext

        # Update likelihoods
        mu_int_x_dot[0] = self.mu_int_x[1] + lkh['int'] + lkh['prop']
        mu_ext_x_dot[0] = self.mu_ext_x[1] + lkh['vis'] + lkh['forward_ext']
        mu_dim_dot[0] = self.mu_dim[1] + lkh['dim']
        mu_tact_dot[0] = self.mu_tact[1] + lkh['tact']

        # Update dynamics
        mu_int_x_dot[1] = -E_x[0] * c.pi_x_int
        mu_ext_x_dot[1] = -E_x[1] * c.pi_x_ext

        # Backward errors
        mu_int_x_dot[0] += E_x_back_x[0] * c.pi_x_int
        mu_ext_x_dot[0] += E_x_back_x[1] * c.pi_x_ext

        mu_int_v_dot[0] += E_x_back_W[0] * c.pi_eta_int
        mu_ext_v_dot[0] += E_x_back_W[1] * c.pi_eta_ext

        mu_int_v_dot[1] += E_x_back_b[0] * c.pi_eta_int
        mu_ext_v_dot[1] += E_x_back_b[1] * c.pi_eta_ext

        return (mu_int_v_dot, mu_ext_v_dot, mu_int_x_dot,
                mu_ext_x_dot, mu_dim_dot, mu_tact_dot)

    # Get action update
    def get_a_dot(self, e_prop):
        return -c.dt * e_prop

    # Integrate with gradient descent
    def integrate(self, mu_dot, a_dot):
        # Update belief
        mus = (self.mu_int_v, self.mu_ext_v, self.mu_int_x,
               self.mu_ext_x, self.mu_dim, self.mu_tact)

        for mu, dot in zip(mus, mu_dot):
            mu[0] += c.dt * dot[0]
            mu[1] += c.dt * dot[1]

        self.mu_int_x[0] = np.clip(
            self.mu_int_x[0], *np.tile(self.limits, (c.n_objects + 1, 1)).T)
        self.mu_tact[0] = np.clip(self.mu_tact[0], 0, 1)

        # Update action
        self.a += c.dt * a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    # Set intention sequence
    def set_intentions(self):
        mu_ext_reshape = self.mu_ext_x[0].reshape(c.n_objects + 1, 2)

        grasp_pos = utils.denormalize(mu_ext_reshape[0], c.norm_cart)
        target_pos = utils.denormalize(mu_ext_reshape[1], c.norm_cart)
        home_pos = utils.denormalize(mu_ext_reshape[2], c.norm_cart)

        grasp_target = np.linalg.norm(grasp_pos - target_pos)
        home_target = np.linalg.norm(home_pos - target_pos)

        # Intention conditions
        target_reached = grasp_target < c.reach_dist - 10
        home_reached = home_target < c.home_size / 2
        picked = np.all(self.mu_tact[0] > 0.5)

        # Reach target until picked
        self.gamma_int[0] = not picked  # not home_reached

        # Pick target when is on reach
        self.gamma_int[1] = not home_reached and not picked and target_reached

        # Open grip when target is on home or out of reach
        self.gamma_int[2] = home_reached or (not target_reached and not picked)

        # Reach home when target picked
        self.gamma_int[3] = picked
        # self.gamma_ext[0] = picked

        # Stay
        self.gamma_int[4] = not self.gamma_int[:3].any()
        self.gamma_ext[1] = not self.gamma_ext[0]

        # Normalize precisions
        self.gamma_int = self.gamma_int / self.gamma_int.sum()
        self.gamma_ext = self.gamma_ext / self.gamma_ext.sum()

    # Run an inference step
    def inference_step(self, Y):
        # Set intention sequence
        self.set_intentions()

        # Get predictions
        P = self.get_p()

        # Get sensory prediction errors
        Y[1] = Y[1].flatten()
        E_y = self.get_e_y((self.mu_ext_x[0], *Y), P)

        # Get priors
        eta_int, eta_ext = self.get_eta()

        # Get prior errors
        e_eta_int, e_eta_ext = self.get_e_eta(eta_int, eta_ext)

        # Get intentions
        i_int, i_ext = self.get_i()

        # Get dynamics prediction errors
        E_x, E_x_back_x, E_x_back_W, E_x_back_b = \
            self.get_e_x(i_int, i_ext)

        # Get likelihood components
        likelihood = self.get_likelihood(E_y)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, e_eta_int, e_eta_ext,
                                 E_x, E_x_back_x, E_x_back_W, E_x_back_b)

        # Get action update
        a_dot = self.get_a_dot(likelihood['prop'][:c.n_joints])

        # Update
        self.integrate(mu_dot, a_dot)

        return utils.denormalize(self.a, c.norm_polar) * c.gain_a
