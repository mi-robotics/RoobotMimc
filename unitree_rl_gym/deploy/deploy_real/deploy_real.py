from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, transform_imu_data_full, transform_imu_data_from_pelvis_to_torso_quat
from common.remote_controller import RemoteController, KeyMap
from config import Config


from collections.abc import Sequence
from collections import deque

from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
    matrix_from_quat, subtract_frame_transforms, quat_conjugate
)

from dataclasses import dataclass, field
from typing import List
from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

import time
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import onnxruntime as ort

from enum import Enum # Import Enum for our state machine

# --- New Additions: State Machine and Configuration ---

# 1. Define the states for our motion controller

UNITREE_SDK_JOINT_ORDER = [
    # Left Leg (Indices 0-5)
    "left_hip_pitch_joint", #0
    "left_hip_roll_joint", #1
    "left_hip_yaw_joint", #2
    "left_knee_joint", #3
    "left_ankle_pitch_joint", #4
    "left_ankle_roll_joint", #5

    # Right Leg (Indices 6-11)
    "right_hip_pitch_joint", #6
    "right_hip_roll_joint", #7
    "right_hip_yaw_joint", #8
    "right_knee_joint", #9
    "right_ankle_pitch_joint", #10
    "right_ankle_roll_joint", #11

    # Waist (Indices 12-14)
    "waist_yaw_joint", #12
    "waist_pitch_joint", #13
    "waist_roll_joint", #14

    # Left Arm (Indices 15-21)
    "left_shoulder_pitch_joint", #15
    "left_shoulder_roll_joint", #16
    "left_shoulder_yaw_joint", #17
    "left_elbow_joint", #18
    "left_wrist_pitch_joint", #19
    "left_wrist_roll_joint", #20
    "left_wrist_yaw_joint", #21

    # Right Arm (Indices 22-28)
    "right_shoulder_pitch_joint", #22
    "right_shoulder_roll_joint", #23
    "right_shoulder_yaw_joint", #24
    "right_elbow_joint", #25
    "right_wrist_pitch_joint", #26
    "right_wrist_roll_joint", #27
    "right_wrist_yaw_joint", #28
]

class MotionState(Enum):
    WARMING_UP = 0
    TRANSITION_TO_MOTION = 1
    EXECUTING_MOTION = 2
    TRANSITION_TO_DEFAULT = 3
    ENDED = 4
    SAVE = 5

class ObservationManager:
    """
    Manages observation collection, noise, and history buffering to replicate
    the Isaac Lab ObsTerm functionality in a standard MuJoCo environment.
    """
    def __init__(self, 
                 controller,
                 history_length: int = 10,
                 default_angles = None,
                 mj_to_policy=None
                 ):
        """
        Args:
            model: The MuJoCo MjModel object.
            data: The MuJoCo MjData object.
            history_length: The number of past steps to stack for each observation term.
        """
        self.controller: Controller = controller
        self.history_length = history_length
        self.default_angles = default_angles
        self.mj_to_policy = mj_to_policy
        self.mj_waist_ids = [12,13,14]

        self.num_dof = 29  # Number of actuated degrees of freedom

        # --- Initialize history buffers (deques) ---
        # Deques are efficient for fixed-length history
        self.base_ang_vel_hist = deque(maxlen=self.history_length)
        self.joint_pos_hist = deque(maxlen=self.history_length)
        self.joint_vel_hist = deque(maxlen=self.history_length)
        self.actions_hist = deque(maxlen=self.history_length)
        
        self.reset()

    def reset(self):
        """
        Resets all history buffers, filling them with zeros.
        This should be called at the beginning of each episode.
        """
 
        zero_base_ang_vel = np.zeros(3)
        zero_joint_pos = np.zeros(self.num_dof)
        zero_joint_vel = np.zeros(self.num_dof)
        zero_actions = np.zeros(self.num_dof) # Assuming action dim == num_dof

        for _ in range(self.history_length):
            self.base_ang_vel_hist.append(zero_base_ang_vel)
            self.joint_pos_hist.append(zero_joint_pos)
            self.joint_vel_hist.append(zero_joint_vel)
            self.actions_hist.append(zero_actions)


    def update(self, last_action: np.ndarray):
        """
        Fetches current state, adds noise, and updates the history buffers.
        Call this once per simulation step.

        Args:
            last_action: The action applied in the most recent step.
        """
        # 1. Fetch raw data from MuJoCo
        # Using local frame velocities to better match Isaac Lab's convention
  
        quat = self.controller.low_state.imu_state.quaternion
        ang_vel = np.array([self.controller.low_state.imu_state.gyroscope], dtype=np.float32)
        
        joint_pos = np.zeros(self.num_dof)
        joint_vel = np.zeros(self.num_dof)
        for i in range(len(self.mj_to_policy)):
            joint_pos[i] = self.controller.low_state.motor_state[self.mj_to_policy[i]].q
            joint_vel[i] = self.controller.low_state.motor_state[self.mj_to_policy[i]].dq
        joint_pos_rel = joint_pos - self.default_angles
        joint_vel_rel = joint_vel
        # 3. Append to history buffers
        self.base_ang_vel_hist.append(ang_vel)
        self.joint_pos_hist.append(joint_pos_rel)
        self.joint_vel_hist.append(joint_vel_rel)
        self.actions_hist.append(last_action.copy()) # Action has no noise

    def get_observation(self) -> np.ndarray:
        """
        Flattens and concatenates all history buffers to form the final
        observation vector.

        Returns:
            A 1D numpy array containing the full, stacked observation.
        """
        obs = np.concatenate([
            np.array(self.base_ang_vel_hist).flatten(),
            np.array(self.joint_pos_hist).flatten(),
            np.array(self.joint_vel_hist).flatten(),
            np.array(self.actions_hist).flatten()
        ])
        return obs
    


@dataclass
class MotionCommandCfg():
    """zonfiguration for the motion command."""
    motion_file: str = ''
    anchor_body: str = ''
    body_names: List[str] = field(default_factory=list)
    motion_start: int = 0
    run_length: int = 0


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]
    
class MotionCommand():
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, device='cpu'):
        self.device = device
        self.cfg = cfg
        self.robot_body_names = ['pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
        self.robot_anchor_body_index = self.robot_body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor([ self.robot_body_names.index(n) for n in self.cfg.body_names], device=self.device)

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.ones(1, dtype=torch.long, device=self.device)*cfg.motion_start
        self.stop_step = self.time_steps+cfg.run_length


    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]
    
    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot_q
    
    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        root_q = self.motion.body_quat_w[[0], self.motion_anchor_body_index]
        base_heading = yaw_quat(root_q)
        out_q = quat_mul(
            quat_conjugate(base_heading),
            self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index], 
            )
        return out_q #removed any inital heading in motions
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    def _set_q(self, q):
        
        self.robot_q = torch.from_numpy(q).unsqueeze(0)

    def _update_command(self):
        self.time_steps += 1

        print(self.time_steps, torch.where(self.time_steps >= self.motion.time_step_total -1)[0])

        env_ids = torch.where(self.time_steps >= self.motion.time_step_total -1)[0]  
        env_ids = torch.cat((env_ids, torch.where(self.time_steps >= self.stop_step)[0] ))

        return len(env_ids)>0
      



class Controller:
    def __init__(self) -> None:
        onnx_model_path = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-24_16-21-37_lefan_walk_padded/exported/policy.onnx"    
        # Step 2: Create an Inference Session
        try:
            # Use GPU provider if available, otherwise fall back to CPU
            self.session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        except ort.SessionCreationError:
            # Fallback to CPU if GPU is not available
            self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        model_meta = self.session.get_modelmeta()
        self.kds = np.array(model_meta.custom_metadata_map['joint_damping'].split(','), dtype=float)  
        self.kps = np.array(model_meta.custom_metadata_map['joint_stiffness'].split(','), dtype=float) 
        self.action_scale = np.array(model_meta.custom_metadata_map['action_scale'].split(','), dtype=float) 
        self.default_angles = np.array(model_meta.custom_metadata_map['default_joint_pos'].split(','), dtype=float) 
        policy_joint_names = model_meta.custom_metadata_map['joint_names'].split(',')    
        effort_limit = np.array(model_meta.custom_metadata_map['joint_effort_limit'].split(','), dtype=float) 
        output_shape = self.session.get_outputs()[0].shape
        num_actions = output_shape
        input_shape = self.session.get_inputs()[0].shape
        num_obs = input_shape

        self.mj_to_policy = [UNITREE_SDK_JOINT_ORDER.index(n) for n in policy_joint_names]    
        self.policy_to_mj = [policy_joint_names.index(n) for n in UNITREE_SDK_JOINT_ORDER]

        # self.config = config
        self.remote_controller = RemoteController()

        # Initializing process variables
        self.obs_manager = ObservationManager(
            history_length=1,
            default_angles=self.default_angles,
            mj_to_policy=self.mj_to_policy
        )

        config = {
            'motion_file': '/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/tmp/walk1_subject1.npz',
            'anchor_body': 'torso_link',
            'body_names': [
                'pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
                'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
            ]
        }
        motion_cfg = MotionCommandCfg(
            motion_file=config['motion_file'],
            anchor_body=config['anchor_body'],
            body_names=config['body_names'],
            motion_start=150,
            run_length=200
        )
        self.command_manager = MotionCommand(motion_cfg) 
        self.current_state = MotionState.WARMING_UP
        self.warmup_time = 100
        self.transition_time = 100
        self.motion_ended = False


        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()

        self.counter = 0
        self.control_step_counter = 0

        self.config = {
            'control_dt': 0.02 #50hz
        }

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        self.wait_for_low_state()

        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
     

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config['control_dt'])
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config['control_dt'])


  
    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config['control_dt'])
        
        dof_idx = self.mj_to_policy
        kps = self.kps
        kds = self.kds
        default_pos = self.default_angles.copy()
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config['control_dt'])

    # ########################################################################TODO
    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.mj_to_policy)):
                motor_idx = self.mj_to_policy[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.send_cmd(self.low_cmd)
            time.sleep(self.config['control_dt'])

    def update_fsm(self):
        if self.current_state == MotionState.WARMING_UP:
            print(f'WARMING UP ----------------- {self.control_step_counter}')
            self.task_command_angles = np.copy(self.default_angles) # Hold default pose
            self.task_command_vels = np.zeros_like(self.task_command_angles)
            self.task_motion_ori = torch.tensor([[1,0,0,0]])

            if self.control_step_counter >= self.warmup_time:
                self.task_command_angles  = self.command_manager.command[0, :29].numpy()
                self.first_task_command_angle = self.task_command_angles.copy()
            
                # Change state and reset counter for interpolation
                self.current_state = MotionState.TRANSITION_TO_MOTION
                self.interpolation_counter = 0

        # --- State: TRANSITION_TO_MOTION ---
        elif self.current_state == MotionState.TRANSITION_TO_MOTION:
            print(f'TRANSITION_TO_MOTION----------------- {self.control_step_counter}')
            alpha = min(self.interpolation_counter / self.transition_time, 1.0)
            self.task_command_angles = (1 - alpha) * self.default_angles + alpha * self.first_task_command_angle
            self.interpolation_counter += 1
            
            if self.interpolation_counter >= self.transition_time:
                print("Transition complete. Executing motion.")
                self.current_state = MotionState.EXECUTING_MOTION
                

        elif self.current_state == MotionState.EXECUTING_MOTION:
            print(f'EXECUTING_MOTION ----------------- {self.control_step_counter}')
            self.task_command_angles = self.command_manager.command[0, :29].numpy()
            self.task_command_vels = self.command_manager.command[0, 29:].numpy()
            self.task_motion_ori = self.command_manager.anchor_quat_w
            # Check for the end condition
            if self.motion_ended:
                self.final_task_command_angle = self.task_command_angles.copy()
                
                self.current_state = MotionState.TRANSITION_TO_DEFAULT
                self.interpolation_counter = 0

        # --- State: TRANSITION_TO_DEFAULT ---
        elif self.current_state == MotionState.TRANSITION_TO_DEFAULT:
            print(f'TRANSITION_TO_DEFAULT ----------------- {self.control_step_counter}')
            alpha = min(self.interpolation_counter / self.transition_time, 1.0)
            # Interpolate from the last policy pose back to the default
            self.task_command_angles = (1 - alpha) * self.final_task_command_angle + alpha *  np.copy(self.default_angles)
            self.task_command_vels = np.zeros_like(self.task_command_angles)
            self.task_motion_ori = torch.tensor([[1,0,0,0]])
            self.interpolation_counter += 1

            if self.interpolation_counter >= self.transition_time:
                print("Transition to default complete. Motion ended.")
                self.ended_counter = 0
                self.current_state = MotionState.ENDED
        
        # --- State: ENDED ---
        elif self.current_state == MotionState.ENDED:
            print(f'ENDED ----------------- {self.control_step_counter}')
            self.task_command_angles = np.copy(self.default_angles) # Hold default pose
            self.ended_counter += 1

            if self.ended_counter > 100:
                self.current_state = MotionState.SAVE

        quat = self.controller.low_state.imu_state.quaternion
        # Assummes pelvis imu
        # "waist_yaw_joint",
        # "waist_pitch_joint",
        # "waist_roll_joint",
        waist_yaw = self.low_state.motor_state[12].q
        waist_pitch = self.low_state.motor_state[13].q
        waist_roll = self.low_state.motor_state[14].q

        quat = transform_imu_data_from_pelvis_to_torso_quat(
            waist_roll=waist_roll, 
            waist_pitch=waist_pitch, 
            waist_yaw=waist_yaw, 
            imu_quat=quat, 
        )

        self.command_manager._set_q(q)

        self.task_command = torch.cat([torch.tensor(self.task_command_angles).unsqueeze(0), torch.tensor(self.task_command_vels).unsqueeze(0)], dim=1)
        _, ori = subtract_frame_transforms(
            self.command_manager.robot_anchor_pos_w,
            self.command_manager.robot_anchor_quat_w,
            self.command_manager.anchor_pos_w,
            self.task_motion_ori,
        )
        mat = matrix_from_quat(ori)

        self.motion_anchor_ori_b = mat[..., :2].reshape(mat.shape[0], -1)
    
    def run(self):
        self.counter += 1
        self.control_step_counter += 1
        
        self.update_fsm()
        self.obs_manager.update(self.action.copy()[0])

        # Get the action from the policy network
        proprio = torch.from_numpy(self.obs_manager.get_observation())
        obs = torch.cat((
            self.task_command, 
            self.motion_anchor_ori_b, 
            proprio
        ), dim=-1)

        # policy inference
        # Create a dictionary to map input names to your data
        input_dict = {
            "obs": obs.numpy().astype(np.float32),
            "time_step": torch.tensor([[0]]).float().numpy().astype(np.float32)
        }

        self.action = self.session.run(['actions'], input_dict)[0]
        
        # filtered_action = filtered_action * 0. + action * 1
        target_dof_pos = (self.action * self.action_scale + self.default_angles)[0]

        # Build low cmd
        for i in range(len(self.mj_to_policy)):
            motor_idx = self.mj_to_policy[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.send_cmd(self.low_cmd)

        if self.current_state == MotionState.EXECUTING_MOTION:
            self.motion_ended = self.command_manager._update_command()

        time.sleep(self.config['control_dt'])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
  
    args = parser.parse_args()


    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller()

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
