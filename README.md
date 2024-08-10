# Slow but flexible or fast but rigid?

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the paper [Slow but flexible or fast but rigid? Discrete and continuous processes compared](https://www.biorxiv.org/content/10.1101/2023.08.20.554008v2.abstract). It investigates the tradeoff, in active inference, between high-level cognitive planning and low-level motor control regarding multi-step tasks. It compares two models (a hybrid discrete-continuous model and a continuous-only model) on a dynamic pick-and-place operation. The study further explores how discrete actions could lead to continuous attractors, and draws parallelisms with different motor learning phases, aiming to inform future research on bio-inspired task specialization.

Video simulations are found [here](https://priorelli.github.io/projects/4_discrete_or_continuous/).

Check [this](https://priorelli.github.io/projects/) and [this](https://priorelli.github.io/blog/) for additional projects and guides.

This study has received funding from the [MAIA project](https://www.istc.cnr.it/it/content/maia-multifunctional-adaptive-and-interactive-ai-system-acting-multiple-contexts), under the European Union's Horizon 2020 Research and Innovation programme.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-c` for the continuous-only model, `-y` for the hybrid model, or `-a` for choosing the parameters from the console. If no option is specified, the continuous-only model will be launched. For the manual control simulation, the arm can be moved with the keys `Z`, `X`, `LEFT`, `RIGHT`, `UP` and `DOWN`. The hand can be rotated with `A` and `S`. The fingers can be open or closed for grasping with `Q` and `W`.

Plots can be generated through *plot.py*, either with the option `-d` for the hand-target distance and dynamics of action probabilities, `-s` for the scores, or `-v` for generating a video of the simulation.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials and steps can be set with the variables `n_trials` and `n_steps`, respectively.

The parameter `grasp_dist` controls the distance from the wrist at which the object is grasped.

The parameter `open_angle` controls the maximum opening of the hand.

The parameter `n_tau` specifies the sampling time window used for evidence accumulation.

The parameter `target_vel` specifies the velocity of the object, which in the simulations have been varied from 0 to 80 pixels per time step.

Different precisions and attractor gains have been used for the hybrid and continuous-only model: these are specified in *main.py*.

The agent configuration is defined through the dictionary `joints`. The value `link` specifies the joint to which the new one is attached; `angle` encodes the starting value of the joint; `limit` defines the min and max angle limits.

### Agent

The scripts *simulation/inference_cont.py* and *simulation/inference_hybrid.py* contain subclasses of `Window` in *environment/window.py*, which is in turn a subclass of `pyglet.window.Window`. The only overriden function is `update`, which defines the instructions to run in a single cycle. Specifically, the subclasses `InferenceContinuous` and `InferenceHybrid` initialize the agent, the object and the goal; during each update, they retrieve proprioceptive, visual, and tactile observations through functions defined in *environment/window.py*, call the function `inference_step` of the respective agents (defined in *cont_only_agent.py* and *hybrid_agent_cont.py*), move the arm and the objects, and finally check for possible collisions. Every `n_tau` steps, `InferenceHybrid` calls an inference step of an additional discrete agent, defined in *hybrid_agent_disc.py*.

The continuous agents have a similar function `inference_step`, but with some differences. In particular, the continuous-only agent generates visual, proprioceptive, and tactile predictions via the function `get_p`, while the continuous agent of the hybrid method only generates visual and proprioceptive predictions. The continuous-only agent specifies transitions between intentions via the function `set_intentions`. This function generates variables `gamma_int` and `gamma_ext`, which affect the beliefs over hidden causes. Then, the function `get_i` computes the intentions from the intrinsic and extrinsic beliefs over hidden states and hidden causes. These are used to generate dynamics prediction errors via the function `get_e_x`. Instead, the continuous agent of the hybrid method generates prior prediction errors via the function `get_e_x`, depending on the full priors computed by the discrete agent. For both models, the function `get_likelihood` backpropagates the sensory prediction errors toward the beliefs, calling the function `grad_ext` to compute the extrinsic gradient. Finally, the function `mu_dot` computes the total belief updates. At each continuous time step, the continuous agent of the hybrid method accumulate evidence in both (intrinsic and extrinsic) modalities.

Bayesian model average and Bayesian model comparison between reduced priors is computed via the functions `do_bmc` and `do_bma` in the discrete agent of the hybrid method. The reduced priors are instead updated through the function `get_eta_m`.

Note that the function `init_belief` changes the length of the last joint to account for the grasping distance.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between -1 and 1 to ensure that every contribution to the belief updates has the same magnitude.
