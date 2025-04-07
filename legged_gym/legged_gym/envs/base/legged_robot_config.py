# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base_config import BaseConfig
import numpy as np

num_steps = 24#48 # original 24, changed to 48 for 2048 envs

class LeggedRobotCfg(BaseConfig):
    class env:
        num_envs = 4096
        # num_observations = 253
        # num_observations = 258
        n_history = 10
        n_scan = 187
        n_priv = 3+3 +3
        n_priv_latent = 4 + 1 + 18 +18
        n_proprio = 3+2+1+3+1+18+18+1+1+18+6#3 + 2 + 3 + 4 + 36 + 5
        history_len = 10
        #num_observations = 207
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent + n_priv #n_scan + n_proprio + n_priv #187 + 47 + 5 + 12 
        # 72+

        num_privileged_obs = None # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 12
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 25 # episode length in seconds

        history_encoding = True
        # action_delay_range = [0, 5]
        contact_buf_len = 100


    class depth:
        use_camera = True
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 5  # 5 works without retraining, 8 worse

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2
        
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        
        scale = 1
        invert = True

    class terrain:
        mesh_type = 'trimesh' # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # measured_points_x = [-0.45, -0.3, -0.15, 0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2] # 1mx1.6m rectangle (without center line)
        # measured_points_y = [-0.75, -0.6, -0.45, -0.3, -0.15, 0., 0.15, 0.3, 0.45, 0.6, 0.75]
        measure_horizontal_noise = 0
        selected = False # select a unique terrain type and pass all arguments
        terrain_kwargs = None # Dict of arguments for selected terrain
        max_init_terrain_level = 5 # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows= 10 # number of terrain rows (levels)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, flat]
        terrain_max_multiplier = 0.3# 0.1 for easy

        # terrain_proportions = [0.1*terrain_max_multiplier, 0.1*terrain_max_multiplier, 
        #                        0.3*terrain_max_multiplier, 0.2*terrain_max_multiplier, 
        #                        0.2*terrain_max_multiplier, 0.1*terrain_max_multiplier]

        terrain_proportions = [ 0.0, 0.0,
                               0.35*terrain_max_multiplier, 0.25*terrain_max_multiplier, 
                               0.25*terrain_max_multiplier, 0.15*terrain_max_multiplier]

        num_cols = 4*len(terrain_proportions) # number of terrain cols (types)

        # trimesh only:
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        rough_flat = False
        flat_terrain_flag = True



    class commands:
        curriculum = True
        max_curriculum = 1.
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = True # if true: compute ang vel command from heading error
        lin_vel_clip = 0.1
        rand_heading = True
        class ranges:
            lin_vel_x = [0., 0.5] # min max [m/s]
            lin_vel_y = [0., 0.]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-np.pi/3, np.pi/3]

    class init_state:
        pos = [0.0, 0.0, 1.] # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = { # target angles when action = 0.0
            "joint_a": 0., 
            "joint_b": 0.}

    class control:
        control_type = 'P' # P: position, V: velocity, T: torques
        # PD Drive parameters:
        stiffness = {'joint_a': 10.0, 'joint_b': 15.}  # [N*m/rad]
        damping = {'joint_a': 1.0, 'joint_b': 1.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset:
        file = ""
        name = "legged_robot"  # actor name
        foot_name = "None" # name of the feet bodies, used to index body state and contact force tensors
        penalize_contacts_on = []
        terminate_after_contacts_on = []
        disable_gravity = False
        collapse_fixed_joints = False # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up
        
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = False
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.
        action_delay = False
        motor_strength_range = [0.8, 1.2]
        action_buf_len = 8

        
        randomize_base_com = True
        added_com_range = [-0.2, 0.2]
        

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        delay_update_global_steps = 24 * 8000
        action_delay = False
        action_curr_step = [1, 1]
        action_curr_step_scratch = [0, 1]
        action_delay_view = 1

    class rewards:
        class scales:
            termination = -1.0
            # tracking_lin_vel = 2.0

            tracking_goal_vel = 4 # for flat
            delta_yaw = 1.2#0.75 # for flat

            # tracking_goal_vel = 3.0 # for terrain
            # delta_yaw = 1.2#0.75

            # tracking_ang_vel = 0.5
            # lin_vel_z = -2.5#-1 for flat
            lin_vel_z = -1 # for terrain

            lin_vel_x = 0
            lin_vel_y = 0
            # lin_vel_heading= 2.
            
            # velocity_magnitude=0.5 # why is this here??
            #lin_vel_y= -2.0
            # ang_vel_xy = -2.0#-0.05 for flat
            ang_vel_xy = -0.05 # for terrain

            
            orientation = -1.0 # for terrain
            # orientation = -1.0 # for flat

            torques = -0.00001
            dof_vel = 0#2.5e-5
            dof_acc = -2.5e-7
            # base_height = -0.25
            # feet_air_time = 0.5#1.
            # feet_air_time = 0.75#1.
            feet_air_time = 0.7#1.


            # three_feet_on_ground=0.3
            #no_leg_dragging=-0.3
            #no_leg_hovering=-0.3
            collision = -1.
            #balanced_leg_contact=0.2
            stumble = -1.0#-0.5
            action_rate = -0.01
            stand_still = -0.5
            rule_1 = 0.35
            # rule_2 = 0.25
            rule_3 = 0.1

        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.5
        soft_torque_limit = 1.
        base_height_target = 0.2
        max_contact_force = 100. # forces above this value are penalized
        vel_tracking_reward = "tracking_goal_vel"
        torque_limit_hard = 4
        contact_tresh = 0.5
        exp_coeff_rule3 = -10
        stumble_tresh = 2.5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.01
            orientation = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [11., 5, 3.]  # [m]
        show_heading=False

    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**24#2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

class LeggedRobotCfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'
    class policy:
        init_noise_std = 1.0
        #Scan rncoder
        scan_encoder_dims = [128, 64, 32]
        priv_encoder_dims = [64, 20]

        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = [64, 20]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1
        tanh_encoder_output = False
    class distil:
        num_episodes = 10000
        num_epochs = 10000
        num_teacher_obs = 235 - 12 - 24 - 3
        logging_interval = 5
        save_interval = 1000  
        epoch_save_interval = 10
        batch_size = 1024
        num_steps = 100
        num_training_iters = 10
        lr = 1e-3
        training_device = "cuda:0"
        max_buffer_length = 1000000
        num_warmup_steps = 100
    class teacher_policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        use_depth_backbone = False
        backbone_type = "deepgait_coordconv"
        num_input_vis_obs = 10 * 32 * 32
        num_output_vis_obs = 1280
    
    class student_policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        use_depth_backbone = False
        backbone_type = "deepgait_coordconv"
        num_input_vis_obs = 10 * 32 * 32
        num_output_vis_obs = 1280
        
    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed        run_name = 'Trained_on_curriculum_terrain_feet_air_time_0.75_with_no_y_component'
        dagger_update_freq = 20
        priv_reg_coef_schedual = [0, 0.1, 2000, 3000]
        priv_reg_coef_schedual_resume = [0, 0.1, 0, 1]
        gamma = 0.99

        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.
    class teacher_runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1500 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'rough_a1'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class depth_encoder:
        if_depth = LeggedRobotCfg.depth.use_camera
        depth_shape = LeggedRobotCfg.depth.resized
        buffer_len = LeggedRobotCfg.depth.buffer_len
        hidden_dims = 512
        learning_rate = 1.e-3
        num_steps_per_env = LeggedRobotCfg.depth.update_interval * num_steps#LeggedRobotCfg.depth.update_interval * 24


    class estimator:
        train_with_estimated_states = True
        learning_rate = 1.e-4
        hidden_dims = [128, 64]
        priv_states_dim = LeggedRobotCfg.env.n_priv
        num_prop = LeggedRobotCfg.env.n_proprio
        num_scan = LeggedRobotCfg.env.n_scan

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = num_steps#24 # per iteration, changed to 48 because 
        max_iterations = 20000# number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations
        experiment_name = 'test'
        # run_name = 'Trained_on_flat_terrain_fresh-airtime_0_5-parkour_obs_no_yawvel-heading_reward_1_2'
        run_name = ''

        # load and resume
        resume = False
        load_run = -1# -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt