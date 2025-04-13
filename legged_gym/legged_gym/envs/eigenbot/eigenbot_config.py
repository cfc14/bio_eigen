from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch
import numpy as np

class EigenbotRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'bendy_joint_M1_S1': -np.pi/4,   # [rad]
            'bendy_joint_M2_S2': 0.0,   # [rad]
            'bendy_joint_M3_S3': np.pi/4,
            'bendy_joint_M4_S4': -np.pi/4,
            'bendy_joint_M5_S5': 0.0,
            'bendy_joint_M6_S6': np.pi/4,
            'bendy_joint_M7_S7': np.pi/4,
            'bendy_joint_M8_S8': np.pi/4,
            'bendy_joint_M9_S9': np.pi/4,
            'bendy_joint_M10_S10': np.pi/4,
            'bendy_joint_M11_S11': np.pi/4,
            'bendy_joint_M12_S12': np.pi/4,
            'bendy_joint_M13_S13': np.pi/4,
            'bendy_joint_M14_S14': np.pi/4,
            'bendy_joint_M15_S15': np.pi/4,
            'bendy_joint_M16_S16': np.pi/4,
            'bendy_joint_M17_S17': np.pi/4,
            'bendy_joint_M18_S18': np.pi/4,
            
            }
    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # action_scale = 0.1
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/eigenbot/urdf/hexapod_v2.urdf'
        name = "eigenbot"
        foot_name = "foot_input"
        #penalize_contacts_on = ["bendy_input_M18_S18","bendy_input_M17_S17","bendy_input_M16_S16","bendy_input_M15_S15","bendy_input_M14_S14","bendy_input_M13_S13","bendy_input_M12_S12","bendy_input_M11_S11","bendy_input_M10_S10","bendy_input_M9_S9","bendy_input_M8_S8","bendy_input_M7_S7","bendy_input_M6_S6","bendy_input_M5_S5","bendy_input_M4_S4","bendy_input_M3_S3","bendy_input_M2_S2","bendy_input_M1_S1","static_elbow_M24_S24","static_elbow_M23_S23","static_elbow_M22_S22","static_elbow_M21_S21","static_elbow_M20_S20","static_elbow_M19_S19","foot_input_M25_S25","foot_input_M26_S26","foot_input_M27_S27","foot_input_M28_S28","foot_input_M29_S29","foot_input_M30_S30"]
        penalize_contacts_on = ["bendy_input_M18_S18","bendy_input_M17_S17","bendy_input_M16_S16",
                                "bendy_input_M15_S15","bendy_input_M14_S14","bendy_input_M13_S13",
                                "bendy_input_M12_S12","bendy_input_M11_S11","bendy_input_M10_S10",
                                "bendy_input_M9_S9","bendy_input_M8_S8","bendy_input_M7_S7",
                                "bendy_input_M6_S6","bendy_input_M5_S5","bendy_input_M4_S4",
                                "bendy_input_M3_S3","bendy_input_M2_S2","bendy_input_M1_S1"]#,
                                # "bendy_output_M18_S18","bendy_output_M17_S17","bendy_output_M16_S16",
                                # "bendy_output_M15_S15","bendy_output_M14_S14","bendy_output_M13_S13",
                                # "bendy_output_M12_S12","bendy_output_M11_S11","bendy_output_M10_S10",
                                # "bendy_output_M9_S9","bendy_output_M8_S8","bendy_output_M7_S7",
                                # "bendy_output_M6_S6","bendy_output_M5_S5","bendy_output_M4_S4",
                                # "bendy_output_M3_S3","bendy_output_M2_S2","bendy_output_M1_S1"]
        terminate_after_contacts_on = ["base_link"]
        self_collisions = 0# 1 to disable, 0 to enable...bitwise filter
        # Add this to the constructor of the LeggedRobot class
        

  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        # base_height_target = 0.25
        base_height_target = 0.25
        soft_torque_limit=1.
        soft_dof_vel_limit= 2.5 #1.5
        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002# -0.00005 # -0.0002
            dof_pos_limits = -10.0
            
    class env(LeggedRobotCfg.env):
        num_actions=18

class EigenbotRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
