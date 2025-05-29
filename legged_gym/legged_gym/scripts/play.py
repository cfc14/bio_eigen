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

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import glob 

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.env.num_envs = args.num_envs

    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_max_multiplier = 0.3
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False

    env_cfg.commands.ranges.lin_vel_x = [0.2,0.4]
    env_cfg.viewer.show_heading=args.show_heading
    # import ipdb;ipdb.set_trace()
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    # import ipdb;ipdb.set_trace()
    policy = ppo_runner.get_inference_policy(device=env.device)

    # expt_path = os.path.join(log_root, train_cfg.runner.expt_id + "_*")
    # matching_runs = sorted(glob.glob(expt_path))
    # if len(matching_runs) == 0:
    #     raise ValueError(f"No runs found for experiment ID: {train_cfg.runner.expt_id} in {log_root}")  
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:

        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 18, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None

    for i in range(10*int(env.max_episode_length)):
        actions = policy(obs.detach())
        if args.use_jit:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    depth_latent = torch.ones((env_cfg.env.num_envs, 32), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), True, infos["depth"], depth_latent)
                else:
                    depth_buffer = torch.ones((env_cfg.env.num_envs, 58, 87), device=env.device)
                    actions, depth_latent = policy_jit(obs.detach(), False, depth_buffer, depth_latent)
            else:
                obs_jit = torch.cat((obs.detach()[:, :env_cfg.env.n_proprio+env_cfg.env.n_priv], obs.detach()[:, -env_cfg.env.history_len*env_cfg.env.n_proprio:]), dim=1)
                actions = policy(obs_jit)
        else:
            if env.cfg.depth.use_camera:
                if infos["depth"] is not None:
                    # scandots_latent_buffer.append(scandots_latent)
                    # obs_prop_depth = obs[:, :self.env.cfg.env.n_proprio].clone()
                    
                    # depth_latent = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)
                    obs_student = obs[:, :env.cfg.env.n_proprio].clone()
                    # import ipdb;ipdb.set_trace()
                    # obs_student[:, 6:8] = 0
                    depth_latent = depth_encoder(infos["depth"].clone(), obs_student)
                    # depth_latent = depth_latent_and_yaw[:, :-2]
                    # yaw = depth_latent_and_yaw[:, -2:]
                # obs[:, 6:8] = 1.5*yaw
                    
            else:
                depth_latent = None
            
            if hasattr(ppo_runner.alg, "depth_actor"):
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            else:
                actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        obs, _, rews, dones, infos = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', str(img_idx) + ".png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    # 'base_height': (env.root_states[robot_index,2]-env.measured_heights[robot_index]) .item(),   
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        # if  0 < i < stop_rew_log:
        #     if infos["episode"]:
        #         num_episodes = torch.sum(env.reset_buf).item()
        #         if num_episodes>0:
        #             logger.log_rewards(infos["episode"], num_episodes)
        # elif i==stop_rew_log:
        #     logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
