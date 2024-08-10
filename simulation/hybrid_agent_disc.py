import numpy as np
import itertools
import utils
import config as c


# Define discrete agent class
class DiscreteAgent:
    def __init__(self, evidence):
        # Likelihood matrix
        pos_states = ['FREE', 'AT_TARGET', 'AT_HOME']
        hand_states = ['OPEN', 'PICKED', 'CLOSED']

        self.states = [pos_states, hand_states]
        self.n_states = len(pos_states) * len(hand_states)
        self.A, self.state_to_idx, self.idx_to_state = self.get_A()
        self.A_tact = self.get_A_tact()

        # Transition matrix
        self.actions = ['TO_TARGET', 'TO_HOME', '  OPEN', '  CLOSE', '  STAY']
        self.n_actions = len(self.actions)
        self.B = self.get_B()

        # Preference matrix
        self.reward_state = ('AT_HOME', 'OPEN')
        reward_idx = self.state_to_idx[self.reward_state]

        self.C = np.zeros(self.n_states)
        self.C[reward_idx] = 1.0

        # Prior matrix
        self.start_state = ('FREE', 'CLOSED')
        start_idx = self.state_to_idx[self.start_state]

        self.D = np.zeros(self.n_states)
        self.D[start_idx] = 1.0

        # Initialize policies
        self.policies = self.construct_policies()
        self.H = np.zeros(len(self.policies))
        # self.H[1875: 2500] = -10.0

        # Compute entropy
        self.H_A = self.entropy()

        # Initialize variables
        self.prior = self.D.copy()
        self.P_u = np.zeros(self.n_actions)
        self.eta_m_int, self.eta_m_ext = self.get_eta_m(evidence)
        self.eta_int = self.eta_m_int[0].copy()
        self.eta_ext = self.eta_m_ext[0].copy()

    # Reset initial state
    def reset(self):
        # Marginalize over hand status
        open_prob = np.sum(self.prior[[0, 3, 6]])
        picked_prob = np.sum(self.prior[[1, 4, 7]])
        closed_prob = np.sum(self.prior[[2, 5, 8]])

        self.prior = np.zeros(self.n_states)
        self.prior[0] = open_prob
        self.prior[1] = picked_prob
        self.prior[2] = closed_prob

    # Get likelihood matrix
    def get_A(self):
        A = np.eye(self.n_states)

        state_to_idx = {}
        idx_to_state = {}
        count = 0
        for i in self.states[0]:
            for j in self.states[1]:
                state_to_idx[(i, j)] = count
                idx_to_state[count] = (i, j)
                count += 1

        return A, state_to_idx, idx_to_state

    # Get tactile likelihood matrix
    def get_A_tact(self):
        A_tact = np.zeros((self.n_states, 2))

        for state, idx in self.state_to_idx.items():
            if state[1] == 'PICKED':
                A_tact[idx, 1] = 1.0
            else:
                A_tact[idx, 0] = 1.0

        return A_tact

    # Get transition matrix
    def get_B(self):
        B = np.zeros((self.n_states, self.n_states, self.n_actions))

        for state, idx in self.state_to_idx.items():
            for action_id, action_label in enumerate(self.actions):

                if action_label == '  OPEN':
                    if state[0] == 'FREE':
                        B[0, idx, action_id] = 1.0
                    elif state[0] == 'AT_TARGET':
                        if state[1] == 'PICKED':
                            B[3, idx, action_id] = 1.0
                        else:
                            B[3, idx, action_id] = 1.0
                            B[0, idx, action_id] = 0.0
                    else:
                        B[6, idx, action_id] = 1.0

                elif action_label == '  CLOSE':
                    if idx == 0:
                        B[2, idx, action_id] = 1.0
                    elif idx == 3:
                        B[2, idx, action_id] = 0.2
                        B[5, idx, action_id] = 0.0
                        B[4, idx, action_id] = 0.8
                    else:
                        B[idx, idx, action_id] = 1.0

                elif action_label == 'TO_TARGET':
                    if state[0] == 'FREE':
                        next_label = ('AT_TARGET', state[1])
                        B[self.state_to_idx[next_label], idx, action_id] = 0.7
                        B[idx, idx, action_id] = 0.3
                    else:
                        B[idx, idx, action_id] = 1.0

                elif action_label == 'TO_HOME':
                    if idx == 4:
                        B[7, idx, action_id] = 0.7
                        B[idx, idx, action_id] = 0.3
                    else:
                        B[idx, idx, action_id] = 1.0

                else:
                    B[idx, idx, action_id] = 1.0

        return B

    # Get all policies
    def construct_policies(self):
        x = [self.n_actions] * c.n_policy

        policies = list(itertools.product(*[list(range(i)) for i in x]))
        for pol_i in range(len(policies)):
            policies[pol_i] = np.array(policies[pol_i]).reshape(c.n_policy, 1)

        return policies

    # Compute entropy of P(o|s)
    def entropy(self):
        H_A = - (self.A * utils.log_stable(self.A)).sum(axis=0)

        return H_A

    # Infer new states q(s_t)
    def infer_states(self, qr_t, o_tact):
        qo_t = self.get_expected_observations(qr_t)

        log_prior = utils.log_stable(self.prior) * c.gain_prior
        log_posterior = utils.log_stable(qo_t) * c.gain_posterior
        log_tact = utils.log_stable(self.A_tact.dot(o_tact))

        qs = utils.softmax(log_posterior + log_prior + log_tact)

        return qs

    # Compute expected states Q(s_t+1 | u_t)
    def get_expected_states(self, qs_current, action):
        qs_u = self.B[:, :, action].dot(qs_current)

        return qs_u

    # Compute expected observations Q(o_t+1 | u_t)
    def get_expected_observations(self, qs_u):
        qo_u = self.A.dot(qs_u)

        return qo_u

    # Compute KL divergence
    def kl_divergence(self, qo_u):
        return (utils.log_stable(qo_u) - utils.log_stable(self.C)).dot(qo_u)

    # Compute action posterior
    def compute_prob_actions(self, Q_pi):
        P_u = np.zeros(self.n_actions)

        for policy_id, policy in enumerate(self.policies):
            P_u[int(policy[0, 0])] += Q_pi[policy_id]

        P_u = utils.norm_dist(P_u)

        return P_u

    # Get prior for next step
    def get_next_prior(self, qs_current):
        qs_next = np.zeros(self.n_states)

        for action_idx, prob in enumerate(self.P_u):
            qs_next += prob * self.B[:, :, action_idx].dot(qs_current)

        return qs_next

    # Compute expected free energy
    def compute_G(self, qs_current):
        G = np.zeros(len(self.policies))

        for policy_id, policy in enumerate(self.policies):
            qs_pi_t = 0

            for t in range(policy.shape[0]):
                action = policy[t, 0]
                qs_prev = qs_current if t == 0 else qs_pi_t

                qs_pi_t = self.get_expected_states(qs_prev, action)
                qo_pi_t = self.get_expected_observations(qs_pi_t)

                kld = self.kl_divergence(qo_pi_t)

                G[policy_id] += self.H_A.dot(qs_pi_t) + kld

        return G

    # Perform Bayesian model comparison
    def do_bmc(self, evidence):
        # Update model outcomes
        self.eta_m_int, self.eta_m_ext = self.get_eta_m(evidence)

        E = -utils.log_stable(self.prior) * c.gain_E

        for t in range(c.n_tau - 10, c.n_tau, 1):
            for m in range(len(E)):
                # Intrinsic
                mu_m_int = evidence['mu_int'][t] + \
                       (c.pi_eta_int / c.pi_x_int) * \
                       (self.eta_m_int[m] - self.eta_int)

                L_m_int = np.sum(c.pi_x_int * mu_m_int[:4] ** 2 -
                                 c.pi_eta_int * self.eta_m_int[m][:4] ** 2 -
                                 c.pi_x_int * evidence['mu_int'][t][:4] ** 2 +
                                 c.pi_eta_int * self.eta_int[:4] ** 2)

                E[m] -= L_m_int * c.delta_int

                # Extrinsic
                mu_m_ext = evidence['mu_ext'][t] + \
                    (c.pi_eta_ext / c.pi_x_ext) * \
                    (self.eta_m_ext[m] - self.eta_ext)

                L_m_ext = np.sum(c.pi_x_ext * mu_m_ext ** 2 -
                                 c.pi_eta_ext * self.eta_m_ext[m] ** 2 -
                                 c.pi_x_ext * evidence['mu_ext'][t] ** 2 +
                                 c.pi_eta_ext * self.eta_ext ** 2)

                E[m] -= L_m_ext * c.delta_ext

        E[:3] = c.bias_E

        qr_t = utils.softmax(-E)

        return qr_t, E

    # Perform Bayesian model average
    def do_bma(self):
        eta_int = np.zeros_like(self.eta_m_int[0])
        eta_ext = np.zeros_like(self.eta_m_ext[0])

        for m, p_o in zip(self.eta_m_int, self.prior):
            eta_int += m * p_o

        for m, p_o in zip(self.eta_m_ext, self.prior):
            eta_ext += m * p_o

        return eta_int, eta_ext

    # Get continuous causes from posterior
    def get_eta_m(self, evidence):
        eta_m_int = np.tile(evidence['mu_int'][-1], (self.n_states, 1, 1))
        eta_m_ext = np.tile(evidence['mu_ext'][-1], (self.n_states, 1, 1))

        for state, idx in self.state_to_idx.items():
            if state[1] == 'OPEN':
                open_grasp = utils.normalize([c.open_angle, -c.open_angle,
                                              0, 0], c.norm_polar)
                eta_m_int[idx, 0, 4:] = open_grasp

            else:
                closed_grasp = utils.normalize([0, 0, -30, 30], c.norm_polar)
                eta_m_int[idx, 0, 4:] = closed_grasp

            if state[0] == 'AT_TARGET':
                eta_m_int[idx, 0, :4] = eta_m_int[idx, 1, :4].copy()
                eta_m_ext[idx, 0] = eta_m_ext[idx, 1].copy()

            elif state[0] == 'AT_HOME':
                eta_m_int[idx, :2, :4] = eta_m_int[idx, 2, :4].copy()
                eta_m_ext[idx, :2] = eta_m_ext[idx, 2].copy()

        return eta_m_int, eta_m_ext

    # Run discrete step based on accumulated evidence
    def inference_step(self, evidence, o_tact):
        # Get posterior from continuous model
        posterior, E = self.do_bmc(evidence)

        # Infer current state
        qs_current = self.infer_states(posterior, o_tact)

        # Compute expected free energy
        G = self.compute_G(qs_current)

        # Marginalize P(u|pi) with probs of each policy Q(pi)
        Q_pi = utils.softmax(self.H - G)

        # Compute action posterior
        self.P_u = self.compute_prob_actions(Q_pi)

        # Compute prior for next step
        self.prior = self.get_next_prior(qs_current)

        # Get prior for continuous model
        self.eta_int, self.eta_ext = self.do_bma()

        if c.debug:
            np.set_printoptions(precision=2, suppress=True)
            print()
            print('Last evidence mu_int:\t\t', evidence['mu_int'][-1, 0])
            print('Last evidence mu_ext:\t\t', evidence['mu_ext'][-1].flatten())
            print('Free energy E:\t\t\t', np.array([np.sum(E[:3]),
                  np.sum(E[3:6]), np.sum(E[6:])]))
            print('Posterior r_t:\t\t\t', posterior)
            print('Current state s_t:\t\t', qs_current)
            print('Prior for next step B*s_t:\t', self.prior)
            print('Continuous prior:\t\t', self.eta_ext[0], self.eta_int[0])
            print('-' * 20)
            print('Posterior pos:\t\t\t', np.array([np.sum(posterior[:3]),
                  np.sum(posterior[3:6]), np.sum(posterior[6:])]))
            print('Current pos:\t\t\t', np.array([np.sum(qs_current[:3]),
                  np.sum(qs_current[3:6]), np.sum(qs_current[6:])]))
            print('Action probs P_u:\t\t', self.P_u)
            input()

        return self.eta_int, self.eta_ext
