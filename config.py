# Window
width = 500
height = 400
off_x = -70
off_y = -120

debug = 0
fps = 600
speed = 6
phys_steps = 20

# Agent
dt = 0.4
a_max = 10.0
gain_a = 2.0

k_int = 0.25
k_ext = 0.15

w_p = 0  # 2e-3
w_a = 0  # 5e-5

pi_prop = 1.0
pi_ext = 0.1
pi_vis = 1.0
pi_dim = 1.0
pi_tact = 1.0

pi_eta_int = 0.01
pi_eta_ext = 0.01
pi_x_int = 0.06
pi_x_ext = 0.06

gain_E = 0.01
bias_E = -0.2
delta_int = 0
delta_ext = 20
gain_prior = 0.55
gain_posterior = 2.0

# Inference
target_size = 14
target_vel = 0
target_min_max = 10, 20

reach_dist = 32
grasp_dist = 40
open_angle = 70

n_trials = 500
n_steps = 6000
n_orders = 2
n_objects = 2
n_policy = 5
n_tau = 50
log_name = ''

home_pos = [-100, 100]
home_size = 50

start = [0, 30, 100, 0, open_angle, -open_angle, 0, 0]
lengths = [50, 70, 90, 20, 34, 34, 30, 30]

# Arm
joints = {}
joints['trunk'] = {'link': None, 'angle': start[0],
                   'limit': (-5, 10), 'size': (lengths[0], 50)}
joints['shoulder'] = {'link': 'trunk', 'angle': start[1],
                      'limit': (-10, 130), 'size': (lengths[1], 40)}
joints['elbow'] = {'link': 'shoulder', 'angle': start[2],
                   'limit': (-5, 130), 'size': (lengths[2], 36)}
joints['wrist'] = {'link': 'elbow', 'angle': start[3],
                   'limit': (-90, 90), 'size': (lengths[3], 36)}
joints['thumb1'] = {'link': 'wrist', 'angle': start[4],
                    'limit': (0, 70), 'size': (lengths[4], 10)}
joints['index1'] = {'link': 'wrist', 'angle': start[5],
                    'limit': (-70, 0), 'size': (lengths[5], 10)}
joints['thumb2'] = {'link': 'thumb1', 'angle': start[6],
                    'limit': (-30, 0), 'size': (lengths[6], 10)}
joints['index2'] = {'link': 'index1', 'angle': start[7],
                    'limit': (0, 30), 'size': (lengths[7], 10)}
n_joints = len(joints)

norm_polar = [-180.0, 180.0]
norm_cart = [-sum(lengths[:6]), sum(lengths[:6])]
