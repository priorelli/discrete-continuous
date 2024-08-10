import numpy as np
import utils
import config as c


# Define continuous agent class
class ContinuousAgent:
    def __init__(self, arm):
        # Initialize generative models
        self.kinematics = arm.kinematics
        self.inverse = arm.inverse
        self.limits = utils.normalize(arm.limits, c.norm_polar)

        # Initialize belief and action
        self.mu_int_x = np.zeros((c.n_orders, c.n_objects + 1, c.n_joints))
        self.mu_ext_x = np.zeros((c.n_orders, c.n_objects + 1, 2))

        self.mu_dim = np.zeros((c.n_orders, c.n_objects))
        self.mu_len = np.zeros((c.n_objects + 1, c.n_joints))

        self.a = np.zeros(c.n_joints)

        # Initialize discrete variables
        self.evidence = {}
        self.evidence['mu_int'] = np.tile(self.mu_int_x[0], (c.n_tau, 1, 1))
        self.evidence['mu_ext'] = np.tile(self.mu_ext_x[0], (c.n_tau, 1, 1))

        self.eta_int = np.zeros_like(self.mu_int_x[0])
        self.eta_ext = np.zeros_like(self.mu_ext_x[0])

    # Initialize belief
    def init_belief(self, init_mu_int, init_mu_dim,
                    grasp_pos, target_pos, home_pos):
        self.mu_int_x[0, :] = utils.normalize(init_mu_int, c.norm_polar)
        self.mu_ext_x[0] = self.g_ext()

        self.mu_dim[0] = np.array(init_mu_dim)

        self.mu_len = np.tile(utils.normalize(
            np.array(c.lengths), c.norm_cart), (c.n_objects + 1, 1))
        self.mu_len[:, 3] += utils.normalize(c.grasp_dist, c.norm_cart)
        # self.mu_len[1, 3] += utils.normalize(c.grasp_dist, c.norm_cart)

    # Get extrinsic belief
    def g_ext(self):
        mu_int_denorm = utils.denormalize(self.mu_int_x[0], c.norm_polar)

        return np.array([self.kinematics(mu_int_denorm[o], self.mu_len[o], 0)
                         [3] for o in range(c.n_objects + 1)])

    # Get extrinsic gradient
    def grad_ext(self, E_ext):
        mu_int_denorm = utils.denormalize(self.mu_int_x[0], c.norm_polar)

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

            lkh_int[o] = E_ext[o].dot(grad_int.T)

        return lkh_int

    # Get predictions
    def get_p(self):
        p_ext = self.g_ext()
        p_prop = self.mu_int_x[0, 0].copy()
        p_vis = self.mu_ext_x[0].copy()
        p_dim = self.mu_dim[0].copy()

        return p_ext, p_prop, p_vis, p_dim

    # Get sensory prediction errors
    def get_e_y(self, Y, P):
        return [y - p for y, p in zip(Y, P)]

    # Get prior prediction errors
    def get_e_x(self):
        eta_int = self.mu_int_x[0].copy()
        eta_ext = self.mu_ext_x[0].copy()

        eta_int[0] = self.eta_int[0].copy()
        eta_ext[0] = self.eta_ext[0].copy()

        e_x_int = (self.mu_int_x[0] - eta_int) * c.pi_x_int
        e_x_ext = (self.mu_ext_x[0] - eta_ext) * c.pi_x_ext

        return e_x_int, e_x_ext

    # Get likelihood components
    def get_likelihood(self, E_y):
        lkh = {}

        lkh['int'] = c.pi_ext * self.grad_ext(E_y[0])

        lkh['prop'] = np.zeros_like(self.mu_int_x[0])
        lkh['prop'][0] = E_y[1] * c.pi_prop

        lkh['vis'] = E_y[2] * c.pi_vis
        lkh['dim'] = E_y[3] * c.pi_dim

        lkh['forward_ext'] = -E_y[0] * c.pi_ext

        return lkh

    # Get belief update
    def get_mu_dot(self, lkh, e_x_int, e_x_ext):
        mu_int_dot = np.zeros_like(self.mu_int_x)
        mu_ext_dot = np.zeros_like(self.mu_ext_x)
        mu_dim_dot = np.zeros_like(self.mu_dim)

        # Update likelihoods
        mu_int_dot[0] = self.mu_int_x[1] + lkh['int'] + lkh['prop']
        mu_ext_dot[0] = self.mu_ext_x[1] + lkh['vis'] + lkh['forward_ext']
        mu_dim_dot[0] = self.mu_dim[1] + lkh['dim']

        # Update priors
        mu_int_dot[0] -= e_x_int
        mu_ext_dot[0] -= e_x_ext

        return mu_int_dot, mu_ext_dot, mu_dim_dot

    # Get action update
    def get_a_dot(self, e_prop):
        return -c.dt * e_prop

    # Integrate with gradient descent
    def integrate(self, mu_dot, a_dot):
        # Update belief
        mus = (self.mu_int_x, self.mu_ext_x, self.mu_dim)

        for mu, dot in zip(mus, mu_dot):
            mu[0] += c.dt * dot[0]
            mu[1] += c.dt * dot[1]

        self.mu_int_x[0] = np.clip(self.mu_int_x[0], *self.limits.T)
        self.mu_ext_x[0] = np.clip(self.mu_ext_x[0], -1, 1)

        # Update action
        self.a += c.dt * a_dot
        self.a = np.clip(self.a, -c.a_max, c.a_max)

    # Accumulate evidence
    def acc_evidence(self, step):
        t = step % c.n_tau
        self.evidence['mu_int'][t] = self.mu_int_x[0]
        self.evidence['mu_ext'][t] = self.mu_ext_x[0]

    # Run an inference step
    def inference_step(self, Y, step):
        # Get predictions
        P = self.get_p()

        # Get sensory prediction errors
        E_y = self.get_e_y((self.mu_ext_x[0], *Y), P)

        # Get prior errors
        e_x_int, e_x_ext = self.get_e_x()

        # Get likelihood components
        likelihood = self.get_likelihood(E_y)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, e_x_int, e_x_ext)

        # Get action update
        a_dot = self.get_a_dot(likelihood['prop'][0])

        # Update
        self.integrate(mu_dot, a_dot)

        # Accumulate evidence
        self.acc_evidence(step)

        return utils.denormalize(self.a, c.norm_polar) * c.gain_a
