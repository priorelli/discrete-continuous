import numpy as np
import utils
import config as c


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.mu_int = np.zeros((c.n_trials, c.n_steps, c.n_orders,
                                c.n_objects + 1, c.n_joints))
        self.mu_ext = np.zeros((c.n_trials, c.n_steps, c.n_orders,
                                c.n_objects + 1, 2))

        self.a = np.zeros((c.n_trials, c.n_steps, c.n_joints))

        self.angles = np.zeros((c.n_trials, c.n_steps, c.n_joints))
        self.est_angles = np.zeros((c.n_trials, c.n_steps, c.n_joints))

        self.pos = np.zeros((c.n_trials, c.n_steps, c.n_joints + 1, 2))
        self.est_pos = np.zeros((c.n_trials, c.n_steps, c.n_joints + 1, 2))

        self.grasp_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_grasp_pos = np.zeros((c.n_trials, c.n_steps, 2))

        self.target_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_target_pos = np.zeros((c.n_trials, c.n_steps, 2))

        self.home_pos = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_home_pos = np.zeros((c.n_trials, c.n_steps, 2))

        self.target_size = np.zeros((c.n_trials, c.n_steps))
        self.success = np.zeros((c.n_trials, c.n_steps))
        self.picked = np.zeros((c.n_trials, c.n_steps))

        self.probs = np.zeros((c.n_trials, c.n_steps, 5))
        self.sim_time = np.zeros(c.n_trials)

    # Track logs for each iteration
    def track(self, step, trial, agent, arm, objects, picked,
              task_done, P_u=None):
        if P_u is None:
            self.mu_int[trial, step] = agent.mu_int_x.reshape(
                c.n_orders, c.n_objects + 1, c.n_joints)
            self.mu_ext[trial, step] = agent.mu_ext_x.reshape(
                c.n_orders, c.n_objects + 1, 2)
        else:
            self.mu_int[trial, step] = agent.mu_int_x
            self.mu_ext[trial, step] = agent.mu_ext_x

        self.a[trial, step] = agent.a

        self.angles[trial, step] = arm.get_angles()
        est_angles = utils.denormalize(self.mu_int[trial, step, 0, 0],
                                       c.norm_polar)
        self.est_angles[trial, step] = est_angles

        pos = arm.kinematics(arm.get_angles(), grasp=0)
        self.pos[trial, step] = np.r_[[np.zeros(2)], pos]
        est_pos = arm.kinematics(est_angles, grasp=0)
        self.est_pos[trial, step] = np.r_[[np.zeros(2)], est_pos]

        est_grasp_pos = arm.kinematics(est_angles)[3]
        self.grasp_pos[trial, step] = arm.grasp.get_pos()
        self.est_grasp_pos[trial, step] = est_grasp_pos

        target_joint = utils.denormalize(self.mu_int[trial, step, 0, 1],
                                         c.norm_polar)
        est_target_pos = arm.kinematics(target_joint)[3]
        self.target_pos[trial, step] = objects.target.get_pos()
        self.est_target_pos[trial, step] = est_target_pos

        home_joint = utils.denormalize(self.mu_int[trial, step, 0, 2],
                                       c.norm_polar)
        est_home_pos = arm.kinematics(home_joint)[3]
        self.home_pos[trial, step] = objects.home.get_pos()
        self.est_home_pos[trial, step] = est_home_pos

        self.target_size[trial, step] = objects.target.radius
        self.picked[trial, step] = picked
        self.success[trial, step] = task_done

        if P_u is not None:
            self.probs[trial, step] = P_u
        else:
            self.probs[trial, step, :] = agent.gamma_int

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_' + c.log_name,
                            mu_int=self.mu_int, mu_ext=self.mu_ext,
                            a=self.a, angles=self.angles,
                            est_angles=self.est_angles,
                            pos=self.pos, est_pos=self.est_pos,
                            grasp_pos=self.grasp_pos,
                            est_grasp_pos=self.est_grasp_pos,
                            target_pos=self.target_pos,
                            est_target_pos=self.est_target_pos,
                            home_pos=self.home_pos,
                            est_home_pos=self.est_home_pos,
                            target_size=self.target_size,
                            success=self.success, picked=self.picked,
                            probs=self.probs, sim_time=self.sim_time)
