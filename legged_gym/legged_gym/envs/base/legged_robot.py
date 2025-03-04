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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg


def euler_from_quaternion(quat_angle):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def heading_vector_from_yaw(yaw_tensor):
    """
    Given a yaw angle tensor (Nx1), return the 2D unit heading vectors as an Nx2 tensor.
    """
    x = torch.cos(yaw_tensor)
    y = torch.sin(yaw_tensor)
    return torch.stack((x, y), dim=1)  # Stack along the second dimension to get an Nx2 tensor
class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        

        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        # Given mode (mx) value for Rayleigh distribution
        mx = 0.01
        sigma = mx / np.sqrt(2 * np.log(2))

        # Initialize per-joint delays with first Rayleigh sample
        self.joint_delays = np.clip(np.random.rayleigh(sigma, 18),0,0.2)  # Initial delays
        self.timestep=0


        # Initialize joint counters as the delays themselves
        self.joint_counters = self.joint_delays.copy()
        for i in range(0,18,3):
            self.joint_counters[i+1]+=self.joint_counters[i]
            self.joint_counters[i+2]+=self.joint_counters[i+1]
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        num_joints = 18
        num_legs = 6
        joints_per_leg = 3

        dt = 0.005  # Simulation timestep

        # Given mode (mx) value for Rayleigh distribution
        mx = 0.01
        sigma = mx / np.sqrt(2 * np.log(2))
        self.nn_actions = actions
        

        # # **Fill the buffer for the first 5 timesteps**
        # if self.timestep < 1:
        #     self.action_delay_buffer[:, :, self.timestep] = actions
        # else:
        #     # Loop through each joint asynchronously
        #     for joint in range(num_joints):
        #         self.joint_counters[joint] -= dt  # Reduce counter by dt each step

        #         if self.joint_counters[joint] <= 0:  # If counter reaches zero, execute action
        #             # Apply the oldest action from the buffer
        #             #actions = self.action_delay_buffer[:, joint, 0]

        #             # Shift the buffer **only for this joint** (FIFO behavior)
        #             #self.action_delay_buffer[:, joint, :-1] = self.action_delay_buffer[:, joint, 1:]
        #             self.action_delay_buffer[:, joint, -1] = actions[:, joint]  # Insert new action

        #             # Generate a new delay for this joint
        #             new_delay = np.clip(np.random.rayleigh(sigma),0,0.2)
        #             self.joint_delays[joint] = new_delay  # Store latest delay

        #             # Identify the leg this joint belongs to
        #             leg_index = joint // joints_per_leg  # Each leg has 3 joints
        #             base_index = leg_index * joints_per_leg  # First joint of this leg

        #             # Accumulate previous joint delays in the same leg
        #             accumulated_delay = new_delay  # Start with own new delay
        #             if joint == base_index + 1:  # Second joint in leg
        #                 accumulated_delay += self.joint_delays[base_index]  # Add J1's delay
        #             elif joint == base_index + 2:  # Third joint in leg
        #                 accumulated_delay += self.joint_delays[base_index] + self.joint_delays[base_index + 1]  # Add J1 + J2 delays

        #             # Reset the counter with the accumulated delay
        #             self.joint_counters[joint] = accumulated_delay

        # # Increment timestep counter
        # self.timestep += 1
        #actions=self.action_delay_buffer[:,:,0]

        # Clip actions based on config constraints
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # clip_nn_actions = self.cfg.normalization.clip_actions
        # self.nn_actions = torch.clip(self.nn_actions, -clip_nn_actions, clip_nn_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)

        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """
        imu_obs = torch.stack((self.roll, self.pitch), dim=1)

        current_heading = self.compute_yaw()
        commanded_heading = self.commands[ :, 3]
        delta_yaw = (commanded_heading - current_heading)[:,None]
        self.flat=1

        flat_tensor = (torch.ones(self.num_envs, device=self.device, requires_grad=False) * self.flat)[:,None]
        not_flat_tensor = (torch.ones(self.num_envs, device=self.device, requires_grad=False)  * self.flat)[:,None]
        #import ipdb; ipdb.set_trace()


        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             self.projected_gravity,
        #                             self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             self.actions,
        #                             ),dim=-1)
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    imu_obs,
                                    delta_yaw,
                                    self.projected_gravity,
                                    self.commands[:, 0:1] * self.commands_scale,
                                    # self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    flat_tensor,                        # [batch_size, 1]
                                    not_flat_tensor,
                                    self.actions,
                                    self.contact_filt.float()-0.5
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        device = self.obs_buf.device  # Get the device of obs_buf
        self.noise_scale_vec = self.noise_scale_vec.to(device)
        # add noise if needed
        if self.add_noise:
        #     print(f"obs_buf shape: {self.obs_buf.shape}")
        #     print(f"noise_scale_vec shape: {self.noise_scale_vec.shape}")
            #import ipdb; ipdb.set_trace()
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        #self._draw_commanded_velocities(env_ids)
        if self.cfg.viewer.show_heading:
            self.gym.clear_lines(self.viewer)
            self._draw_dir()

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            #heading=0 
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    def _draw_commanded_velocities(self, env_ids):
        """ 
        Draws clearly visible arrows for:
        - Commanded velocity direction (blue)
        - Robot heading direction (black)

        Args:
            env_ids (tensor or list): Indices of the environments to visualize.
        """
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Exit early if no valid environments
        if len(env_ids) == 0:
            return

        # Increase scale factor for longer arrows
        scale_factor = 15  

        # Get base positions and commanded velocities
        base_positions = self.root_states[env_ids, :3].cpu().numpy()
        commanded_vels = self.commands[env_ids, :2].cpu().numpy()

        # Compute commanded velocity end positions (scaled)
        end_positions_cmd = base_positions.copy()
        end_positions_cmd[:, 0] += commanded_vels[:, 0] * scale_factor
        end_positions_cmd[:, 1] += commanded_vels[:, 1] * scale_factor

        # **Fix Shape Issue: Expand forward_vec properly**
        forward_vecs = quat_apply(self.base_quat[env_ids], self.forward_vec.expand(len(env_ids), -1)).cpu().numpy()

        # Compute heading end positions (scaled)
        end_positions_heading = base_positions.copy()
        end_positions_heading[:, 0] += forward_vecs[:, 0] * scale_factor
        end_positions_heading[:, 1] += forward_vecs[:, 1] * scale_factor

        # **Prepare line vertices and colors for visualization**
        line_vertices = []
        line_colors = []

        for i in range(len(env_ids)):
            # Commanded velocity (blue)
            line_vertices.append(gymapi.Vec3(*base_positions[i]))
            line_vertices.append(gymapi.Vec3(*end_positions_cmd[i]))
            line_colors.append(gymapi.Vec3(0, 0, 1))  # Blue
            line_colors.append(gymapi.Vec3(0, 0, 1))  # Blue

            # Robot heading direction (black)
            line_vertices.append(gymapi.Vec3(*base_positions[i]))
            line_vertices.append(gymapi.Vec3(*end_positions_heading[i]))
            line_colors.append(gymapi.Vec3(0, 0, 0))  # Black
            line_colors.append(gymapi.Vec3(0, 0, 0))  # Black

        # **Clear previous lines before drawing new ones**
        self.gym.clear_lines(self.viewer)

        # **Draw the arrows as proper lines**
        self.gym.add_lines(self.viewer, self.envs[env_ids[0]], len(line_vertices) // 2, line_vertices, line_colors)


    def _draw_dir(self):
        """ 
        Draws two arrows for each robot:
        - Commanded velocity direction (red-orange)
        - Current heading direction (blue)
        """

        # Define sphere geometries for arrows
        sphere_geom_cmd_vel = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(1, 0.35, 0.25))  # Red-orange
        sphere_geom_heading = gymutil.WireframeSphereGeometry(0.02, 16, 16, None, color=(0, 0, 1))  # Blue

        # Get base positions of robots
        pose_robot = self.root_states[:, :3].cpu().numpy()
        base_positions = self.root_states[:, :3].cpu().numpy()

        # Get commanded velocities and normalize them
        target_vec = heading_vector_from_yaw(self.commands[:, 3])
        norm = torch.norm(target_vec, dim=-1, keepdim=True)
        target_vec_norm_cmd = target_vec / (norm + 1e-5)

        # Get the current heading direction using `forward`
        forward_vecs = quat_apply(self.base_quat, self.forward_vec)[:, :2]  # Extract (x, y) components
        norm_heading = torch.norm(forward_vecs, dim=-1, keepdim=True)
        target_vec_norm_heading = forward_vecs / (norm_heading + 1e-5)  # Normalize

        # Draw arrows for each environment
        for k in range(self.num_envs):
            for i in range(5):
                step_size = 0.2 * (i + 3)  # Scale for arrow visibility

                # Commanded velocity arrow
                pose_arrow_cmd = pose_robot[k, :2] + step_size * target_vec_norm_cmd[k, :2].cpu().numpy()
                pose_cmd = gymapi.Transform(gymapi.Vec3(pose_arrow_cmd[0], pose_arrow_cmd[1], pose_robot[k, 2]), r=None)
                gymutil.draw_lines(sphere_geom_cmd_vel, self.gym, self.viewer, self.envs[k], pose_cmd)

                # Current heading direction arrow
                pose_arrow_heading = pose_robot[k, :2] + step_size * target_vec_norm_heading[k, :2].cpu().numpy()
                pose_heading = gymapi.Transform(gymapi.Vec3(pose_arrow_heading[0], pose_arrow_heading[1], pose_robot[k, 2]), r=None)
                gymutil.draw_lines(sphere_geom_heading, self.gym, self.viewer, self.envs[k], pose_heading)


    # def _resample_commands(self, env_ids):
    #     """ 
    #     Randomly selects new commands for some environments.

    #     Args:
    #         env_ids (List[int]): Environment IDs for which new commands are needed.
    #     """
    #     # Compute the current forward direction (robot's heading in world space)
    #     forward = quat_apply(self.base_quat[env_ids], self.forward_vec)  # Shape: (N, 3)

    #     # Extract the current heading (yaw angle) from forward vector
    #     current_heading = torch.atan2(forward[:, 1], forward[:, 0])  # Compute yaw from (x, y)

    #     # Generate a random heading offset in range [-π/3, π/3]
    #     heading_offset = torch_rand_float(-np.pi / 3, np.pi / 3, (len(env_ids),1), device=self.device).squeeze(1)
    #     #import ipdb;ipdb.set_trace()

    #     # Apply the offset and wrap the heading using wrap_to_pi()
    #     self.commands[env_ids, 3] = wrap_to_pi(current_heading + heading_offset)  # Column 3 stores heading

    #     # Generate linear velocity magnitude
    #     lin_vel_mag = torch_rand_float(
    #         self.command_ranges["lin_vel_x"][0], 
    #         self.command_ranges["lin_vel_x"][1], 
    #         (len(env_ids),1), device=self.device
    #     ).squeeze(1)

    #     # Compute velocity components based on the new heading
    #     self.commands[env_ids, 0] = lin_vel_mag * torch.cos(self.commands[env_ids, 3])  # X component
    #     self.commands[env_ids, 1] = lin_vel_mag * torch.sin(self.commands[env_ids, 3])  # Y component

    #     # Set small velocity commands to zero for stability
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def compute_yaw(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        return heading

    def _resample_commands_parkour(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids, 2] *= torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.ang_vel_clip

        # set small commands to zero
        self.commands[env_ids, :2] *= torch.abs(self.commands[env_ids, 0:1]) > self.cfg.commands.lin_vel_clip
    def _resample_commands(self, env_ids):
        """ Randomly select commands of some environments

        Args:
            env_ids (List[int]): Environment ids for which new commands are needed
        """
        # Generate new heading within [-pi/3, pi/3] of the current heading
        
        current_heading = self.compute_yaw()
        #delta_heading = torch_rand_float(-np.pi / 3, np.pi / 3, (len(env_ids), 1), device=self.device).squeeze(1)
        # Compute safe delta_heading limits
        max_delta = torch.minimum(torch.full_like(current_heading[env_ids], np.pi) - current_heading[env_ids], torch.tensor(np.pi / 3, device=self.device))
        min_delta = torch.maximum(torch.full_like(current_heading[env_ids], -np.pi) - current_heading[env_ids], torch.tensor(-np.pi / 3, device=self.device))

        delta_heading = min_delta + (max_delta - min_delta) * torch.rand((len(env_ids),1), device=self.device).squeeze(1)
        self.commands[env_ids, 3] = wrap_to_pi(current_heading[env_ids]+delta_heading)  # Update heading



        # Generate linear velocity magnitude
        lin_vel_mag = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], 
                                    (len(env_ids), 1), device=self.device).squeeze(1)

        # Compute x and y components of velocity
        # self.commands[env_ids, 0] = lin_vel_mag * torch.cos(self.commands[env_ids, 3])  # x component
        # self.commands[env_ids, 1] = lin_vel_mag * torch.sin(self.commands[env_ids, 3])  # y component
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)


        # Set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
    #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     lin_vel_mag = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     self.commands[env_ids, 0] = lin_vel_mag * torch.cos(self.commands[env_ids,3])  # x component
    #     self.commands[env_ids, 1] = lin_vel_mag * torch.sin(self.commands[env_ids,3])  # y component
    #     # self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # #self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # self.commands[env_ids, 1] = torch_rand_float(0, 0, (len(env_ids), 1), device=self.device).squeeze(1)
    #     # if self.cfg.commands.heading_command:
    #     #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # else:
    #     #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     # set small commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)
    # def _resample_commands(self, env_ids):
    # # Preserve linear velocities along the current heading
    #     forward = quat_apply(self.base_quat, self.forward_vec)
    #     heading = torch.atan2(forward[:, 1], forward[:, 0])

    # # Resample linear velocities (x and y) based on the current heading direction
    #     lin_vel_mag = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    # # Apply the heading direction to velocity components
    #     self.commands[env_ids, 0] = lin_vel_mag * torch.cos(heading[env_ids])  # x component
    #     self.commands[env_ids, 1] = lin_vel_mag * torch.sin(heading[env_ids])  # y component
    
    # # Resample or adjust heading if required, ensuring the bot keeps going in the same direction
    #     if self.cfg.commands.heading_command:
    #         self.commands[env_ids, 3] = heading[env_ids]  # Keep heading as current heading
    #     else:
    #         self.commands[env_ids, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[env_ids, 3] - heading[env_ids]), -1., 1.)

    # # Set small velocity commands to zero
    #     self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)


    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        # import ipdb;ipdb.set_trace()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        # move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        move_down = (distance < torch.norm(self.commands[env_ids, :1], dim=1)*self.max_episode_length_s*0.5) * ~move_up

        # threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        # move_up =dis_to_origin > 0.8*threshold
        # move_down = dis_to_origin < 0.4*threshold

        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums[self.cfg.rewards.vel_tracking_reward][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales[self.cfg.rewards.vel_tracking_reward]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # obs_dim = 253  # This should match the size of each observation in obs_buf
        obs_dim = 256
        noise_vec = torch.zeros(obs_dim)  # Initialize with the correct size
        # self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
        #                             self.base_ang_vel  * self.obs_scales.ang_vel,
        #                             imu_obs,
        #                             delta_yaw,
        #                             self.projected_gravity,
        #                             self.commands[:, 0:1] * self.commands_scale,
        #                             # self.commands[:, :3] * self.commands_scale,
        #                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
        #                             self.dof_vel * self.obs_scales.dof_vel,
        #                             flat_tensor,                        # [batch_size, 1]
        #                             not_flat_tensor,
        #                             self.actions,
        #                             self.contact_filt.float()-0.5
        #                             ),dim=-1)

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level

        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:5] = noise_scales.orientation * noise_level
        noise_vec[5:6] = noise_scales.orientation * noise_level
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:10] = 0.  # commands
        noise_vec[10:22] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[22:34] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[34:46] = 0.  # previous actions

        if self.cfg.terrain.measure_heights:
            # noise_vec[48:253] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
            noise_vec[46:256] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        else:
            # If terrain height measurements are not used, fill the rest with zeros
            #noise_vec[48:253] = 0.
            noise_vec[46:256] = 0.

            # Ensure that noise_vec has exactly 253 elements
        assert noise_vec.shape[0] == obs_dim, f"noise_vec has incorrect size: {noise_vec.shape[0]} != {obs_dim}"

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel], device=self.device, requires_grad=False,) # TODO change this
        # self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.action_delay_buffer=torch.zeros(self.num_envs,18,1,device=self.device)
        # Add this to the constructor of the LeggedRobot class
        self.leg_contact_time = torch.zeros(self.num_envs, len(self.feet_indices), device=self.device)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    #import ipdb;ipdb.set_trace()
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    def _reward_lin_vel_x(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 0])
    def _reward_lin_vel_y(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 1])
    
    # def _reward_lin_vel_y(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.base_lin_vel[:, 1])
    # def _reward_lin_vel_heading(self):
    #     # Calculate forward direction
    #     forward = quat_apply(self.base_quat, self.forward_vec)
        
    #     # Calculate heading angle
    #     heading = torch.atan2(forward[:, 1], forward[:, 0])
        
    #     # Calculate the velocity in the direction of the heading
    #     vel_direction = torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0])
        
    #     # Reward for velocity alignment with heading
    #     vel_alignment = torch.cos(heading - vel_direction)
        
    #     # Reward scaled by the magnitude of velocity in the direction of the heading
    #     vel_magnitude = torch.norm(self.base_lin_vel[:, :2], dim=1)
    #     reward = vel_alignment * vel_magnitude

    #     return reward

    def _reward_lin_vel_heading(self):
    # Calculate forward direction
        forward = quat_apply(self.base_quat, self.forward_vec)
        
        # Calculate heading angle
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        
        # Calculate the velocity in the direction of the heading
        vel_direction = torch.atan2(self.base_lin_vel[:, 1], self.base_lin_vel[:, 0])
        
        # Reward for velocity alignment with heading
        vel_alignment = torch.cos(heading - vel_direction)
        
        # Penalize misalignment (negative cos gives low reward for large angular difference)
        return vel_alignment
    def _reward_velocity_magnitude(self):
        # Calculate the magnitude of the base linear velocity in the horizontal plane
        vel_magnitude = torch.norm(self.base_lin_vel[:, :2], dim=1)
        
        # Reward based on velocity magnitude
        magnitude_reward = vel_magnitude
        
        return magnitude_reward


    
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    def _reward_delta_yaw(self):
        # Penalize high delta_yaw values
        rew = torch.exp(-torch.abs(self.commands[:, 3] - self.compute_yaw()))
        return rew

    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        # import ipdb;ipdb.set_trace()
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_goal_vel(self):
        target_vec = heading_vector_from_yaw(self.commands[:, 3])
        norm = torch.norm(target_vec, dim=-1, keepdim=True)
        target_vec_norm = target_vec / (norm + 1e-5)
        cur_vel = self.base_lin_vel[:, :2]
        # rew = torch.minimum(torch.sum(target_vec_norm * cur_vel, dim=-1), self.commands[:, 0]) / (self.commands[:, 0] + 1e-5)
        # import ipdb;ipdb.set_trace()
        vel_rel_heading = torch.sum(target_vec_norm * cur_vel, dim=-1)
        # import ipdb;ipdb.set_trace()
        lin_vel_error = torch.square(vel_rel_heading - self.commands[:, 0])
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
        # return rew
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    # def _reward_three_feet_on_ground(self):
    # # Reward for having exactly 3 feet on the ground
    # # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact

    # # Count the number of feet in contact with the ground
    #     num_contacts = torch.sum(contact_filt, dim=1)

    # # Reward if exactly 3 feet are in contact with the ground
    #     rew_three_feet = (num_contacts == 3).float()

    # # Add scaling to the reward based on the command's norm
    #     rew_three_feet *= torch.norm(self.commands[:, :2], dim=1) > 0.1  # no reward for zero command

    # # Reset air time for feet in contact
    #     self.feet_air_time *= ~contact_filt

    #     return rew_three_feet

    # def _reward_no_leg_dragging(self):
    #     # Identify legs in contact with the ground
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact

    #     # Increment a "contact time" counter for each foot in contact
    #     self.leg_contact_time += self.dt * contact_filt

    #     # Penalize legs that have been dragging (in contact for too long, e.g., >1 second)
    #     dragging_penalty = torch.sum((self.leg_contact_time > 1.0).float(), dim=1) * -1.0

    #     # Reset the contact time for feet that are no longer in contact with the ground
    #     self.leg_contact_time *= ~contact_filt

    #     # Apply a scaling based on the command's norm (no penalty for zero movement commands)
    #     dragging_penalty *= torch.norm(self.commands[:, :2], dim=1) > 0.1

    #     return dragging_penalty

    # def _reward_no_leg_hovering(self):
    #     # Identify legs in contact with the ground
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)
    #     self.last_contacts = contact

    #     # Increment a "contact time" counter for each foot in contact
    #     self.leg_contact_time += self.dt * contact_filt

       
    #     # Penalize legs that are hovering (not in contact for too long, e.g., >0.5 seconds)
    #     hover_time = self.dt * ~contact_filt
    #     hovering_penalty = torch.sum((hover_time > 1.5).float(), dim=1) * -1.0

        

    #     # Reset the contact time for feet that are no longer in contact with the ground
    #     self.leg_contact_time *= ~contact_filt

    #     # Apply a scaling based on the command's norm (no penalty for zero movement commands)
    #     total_penalty = (hovering_penalty) * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    #     return total_penalty
    # def _reward_balanced_leg_contact(self):
    #     avg_contact_time = self.leg_contact_time.mean(dim=1, keepdim=True) + 1e-6  # Avoid division issues
    #     deviation_from_avg = torch.abs(self.leg_contact_time - avg_contact_time)
        
    #     # Use tanh instead of exp to avoid overflow issues
    #     balanced_contact_reward = torch.tanh(-deviation_from_avg.mean(dim=1) * 0.1)

    #     return balanced_contact_reward







    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    # def _reward_no_standing_still(self):
    #     """
    #     Penalizes the robot for prolonged periods of inactivity where all feet remain stationary.
    #     Encourages continuous motion by evaluating foot dynamics.
    #     """
    #     # Identify feet in contact with the ground
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts)  # Filter unreliable contacts
    #     self.last_contacts = contact

    #     # Calculate air time for feet
    #     air_time = self.dt * ~contact_filt

    #     # Check if all feet are stationary (no air time or contact change for all feet)
    #     all_feet_stationary = torch.all(contact_filt, dim=1)

    #     # Penalize if the robot remains stationary (all feet stationary and no motion commands)
    #     standing_penalty = all_feet_stationary.float() * -1.0

    #     # Scale penalty based on movement command norm
    #     standing_penalty *= (torch.norm(self.commands[:, :2], dim=1) > 0.1).float()

    #     return standing_penalty

