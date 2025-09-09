import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import onnxruntime as ort
import numpy as np
import os


from collections.abc import Sequence

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

from enum import Enum # Import Enum for our state machine

# --- New Additions: State Machine and Configuration ---

# 1. Define the states for our motion controller
class MotionState(Enum):
    WARMING_UP = 0
    TRANSITION_TO_MOTION = 1
    EXECUTING_MOTION = 2
    TRANSITION_TO_DEFAULT = 3
    ENDED = 4
    SAVE = 5

import numpy as np
from scipy.spatial.transform import Rotation as R
def transform_imu_data_from_pelvis_to_torso_quat(
    waist_roll,
    waist_pitch,
    waist_yaw,
    pelvis_quat
):

    # Convert pelvis quaternion to rotation matrix
    R_pelvis = R.from_quat([pelvis_quat[1], pelvis_quat[2], pelvis_quat[3], pelvis_quat[0]]).as_matrix()
    R_waist = R.from_euler('xyz', [waist_roll, waist_pitch, waist_yaw]).as_matrix()
    R_torso = np.dot(R_pelvis, R_waist)
    # --- 3. Conert orientation back to quaternion ---
    torso_quat = R.from_matrix(R_torso).as_quat()[[3, 0, 1, 2]]

    return torso_quat


class JointDataLogger:
    """
    A class to log and visualize time-series data for multiple robot joints.

    This logger records target position, actual position, velocity, and torque
    for a specified number of joints. Data can be saved to a CSV file and
    plotted in three separate graphs.
    """

    def __init__(self, num_joints: int, torque_limits):
        """
        Initializes the logger.

        Args:
            num_joints (int): The number of joints to log (e.g., 29).
        """
        if num_joints <= 0:
            raise ValueError("Number of joints must be a positive integer.")
            
        self.num_joints = num_joints
        self.joint_names = [f'joint_{i}' for i in range(num_joints)]
        
        self.log_data = []
        self.start_time = time.time()
        self.torque_limits = torque_limits

        print(f"JointDataLogger initialized for {self.num_joints} joints.")

    def record_snapshot(self, target_positions, actual_positions, velocities, torques):
        """
        Records a single snapshot of data for all joints.

        Args:
            target_positions (list or np.ndarray): Array of target joint positions.
            actual_positions (list or np.ndarray): Array of recorded joint positions.
            velocities (list or np.ndarray): Array of joint velocities.
            torques (list or np.ndarray): Array of joint torques.
        """
        # Ensure input data has the correct dimensions
        if not all(len(arr) == self.num_joints for arr in [target_positions, actual_positions, velocities, torques]):
            raise ValueError(f"All input data arrays must have length {self.num_joints}")

        timestamp = time.time() - self.start_time
        
        snapshot = {'timestamp': timestamp}
        for i in range(self.num_joints):
            joint_name = self.joint_names[i]
            snapshot[f'{joint_name}_target_pos'] = target_positions[i]
            snapshot[f'{joint_name}_actual_pos'] = actual_positions[i]
            snapshot[f'{joint_name}_vel'] = velocities[i]
            snapshot[f'{joint_name}_torque'] = torques[i]
            
        self.log_data.append(snapshot)

    def save_to_csv(self, filename: str = "joint_log.csv"):
        """
        Saves the logged data to a CSV file.

        Args:
            filename (str): The name of the file to save the data to.
        """
        if not self.log_data:
            print("Warning: No data to save.")
            return

        df = pd.DataFrame(self.log_data)
        df.to_csv(filename, index=False)
        print(f"Data successfully saved to {filename}")

    def plot_data_per_joint(self, save_plots: bool = False, file_prefix: str = "joint_diagnostics"):
        """
        Generates figures with a grid of subplots, one for each joint.

        Creates three separate figures for position, velocity, and torque.

        Args:
            save_plots (bool): If True, saves the plots to PNG files instead of
                               displaying them interactively.
            file_prefix (str): The prefix for the saved plot filenames.
        """
        if not self.log_data:
            print("Warning: No data to plot.")
            return

        df = pd.DataFrame(self.log_data)
        timestamps = df['timestamp']

        # Determine the grid layout for the subplots
        n_cols = 5
        n_rows = math.ceil(self.num_joints / n_cols)

        # --- 1. Position Plots ---
        fig_pos, axes_pos = plt.subplots(n_rows, n_cols, figsize=(20, 15), sharex=True, sharey=True)
        fig_pos.suptitle('Joint Positions (Target vs. Recorded)', fontsize=20)
        axes_pos_flat = axes_pos.flatten()
        
        for i in range(self.num_joints):
            ax = axes_pos_flat[i]
            joint_name = self.joint_names[i]
            ax.plot(timestamps, df[f'{joint_name}_target_pos'], linestyle='--', label='Target')
            ax.plot(timestamps, df[f'{joint_name}_actual_pos'], linestyle='-', label='Actual')
            ax.set_title(joint_name, fontsize=10)
            ax.grid(True)
            ax.legend()
        
        # --- 2. Velocity Plots ---
        fig_vel, axes_vel = plt.subplots(n_rows, n_cols, figsize=(20, 15), sharex=True, sharey=True)
        fig_vel.suptitle('Joint Velocities', fontsize=20)
        axes_vel_flat = axes_vel.flatten()

        for i in range(self.num_joints):
            ax = axes_vel_flat[i]
            joint_name = self.joint_names[i]
            ax.plot(timestamps, df[f'{joint_name}_vel'])
            ax.set_title(joint_name, fontsize=10)
            ax.grid(True)
            
        # --- 3. Torque Plots ---
        fig_tor, axes_tor = plt.subplots(n_rows, n_cols, figsize=(20, 15), sharex=True, sharey=False)
        fig_tor.suptitle('Joint Torques', fontsize=20)
        axes_tor_flat = axes_tor.flatten()

        for i in range(self.num_joints):
            ax = axes_tor_flat[i]
            joint_name = self.joint_names[i]
            limit = self.torque_limits[i]
            padding = limit * 0.10
            ax.set_ylim(-limit - padding, limit + padding)
            ax.axhline(y=self.torque_limits[i], color='green', linestyle='--')            
            ax.axhline(y=-self.torque_limits[i], color='green', linestyle='--')
            ax.plot(timestamps, df[f'{joint_name}_torque'])
            ax.set_title(joint_name, fontsize=10)
            ax.grid(True)

        # Hide any unused subplots in the grid
        for fig_axes in [axes_pos_flat, axes_vel_flat, axes_tor_flat]:
            for i in range(self.num_joints, len(fig_axes)):
                fig_axes[i].set_visible(False)
        
        # Add shared labels
        for fig in [fig_pos, fig_vel, fig_tor]:
             fig.text(0.5, 0.04, 'Time (s)', ha='center', va='center', fontsize=16)
        fig_pos.text(0.06, 0.5, 'Position (radians)', ha='center', va='center', rotation='vertical', fontsize=16)
        fig_vel.text(0.06, 0.5, 'Velocity (rad/s)', ha='center', va='center', rotation='vertical', fontsize=16)
        fig_tor.text(0.06, 0.5, 'Torque (Nm)', ha='center', va='center', rotation='vertical', fontsize=16)
        
        plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])

        if save_plots:
            pos_filename = f"{file_prefix}_positions.png"
            vel_filename = f"{file_prefix}_velocities.png"
            tor_filename = f"{file_prefix}_torques.png"
            
            fig_pos.savefig(pos_filename)
            fig_vel.savefig(vel_filename)
            fig_tor.savefig(tor_filename)
            
            print(f"Plots saved to {pos_filename}, {vel_filename}, and {tor_filename}")
            plt.close('all') # Close figures to prevent them from showing
        else:
            print("Displaying interactive plots...")
            plt.show()

    def plot_torque_individually(self, save_dir: str = "torque_plots"):
        """
        Generates and saves a large, individual plot for each joint's torque.

        Args:
            save_dir (str): The name of the directory where the plot images
                            will be saved. The directory will be created if
                            it does not exist.
        """
        if not self.log_data:
            print("Warning: No data to plot.")
            return

        df = pd.DataFrame(self.log_data)
        timestamps = df['timestamp']

        # Create the output directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"Created directory: {save_dir}")

        print(f"Generating {self.num_joints} individual torque plots...")
        for i in range(self.num_joints):
            joint_name = self.joint_names[i]

            # Create a new, single figure for each joint
            fig, ax = plt.subplots(figsize=(12, 7))

            # Plot the torque data
            ax.plot(timestamps, df[f'{joint_name}_torque'], label='Measured Torque')
            
     
            limit = self.torque_limits[i]
            padding = limit * 0.15 # Use slightly more padding for a better look
            
            ax.set_ylim(-limit - padding, limit + padding)
            ax.axhline(y=limit, color='r', linestyle='--', label=f'Limit (±{limit} Nm)')
            ax.axhline(y=-limit, color='r', linestyle='--')

            # Add titles and labels for clarity
            ax.set_title(f'Torque Analysis for {joint_name}', fontsize=16)
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Torque (Nm)', fontsize=12)
            ax.grid(True)
            ax.legend()
            
            # Construct filename and save the plot
            filename = os.path.join(save_dir, f'torque_{joint_name}.png')
            fig.savefig(filename, dpi=150) # dpi=150 gives good quality
            
            # Close the figure to free up memory
            plt.close(fig)

        print(f"Successfully saved {self.num_joints} torque plots to the '{save_dir}' directory.")




    
 






import mujoco
import numpy as np
from collections import deque
import mujoco.viewer

class ObservationManager:
    """
    Manages observation collection, noise, and history buffering to replicate
    the Isaac Lab ObsTerm functionality in a standard MuJoCo environment.
    """
    def __init__(self, 
                 model: mujoco.MjModel, 
                 data: mujoco.MjData, 
                 history_length: int = 10,
                 default_angles = None,
                 mj_to_policy=None,
                 use_noise=False
                 ):
        """
        Args:
            model: The MuJoCo MjModel object.
            data: The MuJoCo MjData object.
            history_length: The number of past steps to stack for each observation term.
        """
        self.model = model
        self.data = data
        self.history_length = history_length
        self.default_angles = default_angles
        self.mj_to_policy = mj_to_policy
        self.use_noise = use_noise

        # --- Identify joint indices ---
        # Assuming the first 7 qpos/qvel elements are for the floating base (pos + quat)
        # and the rest are for actuated joints. Adjust if your model is different.
        self.num_dof = model.nv - 6  # Number of actuated degrees of freedom
        self.actuated_joint_qpos_indices = np.arange(7, model.nq)
        self.actuated_joint_qvel_indices = np.arange(6, model.nv)
        
        # Store default joint positions for 'joint_pos_rel'
        self.default_qpos = model.qpos0[self.actuated_joint_qpos_indices].copy()

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
        base_ang_vel =  self.data.sensor("base_gyro").data.astype(np.float32)
        joint_pos_rel = self.data.qpos[7:][self.mj_to_policy] - self.default_angles
        joint_vel_rel = self.data.qvel[6:][self.mj_to_policy] 

        if self.use_noise:
            base_ang_vel = base_ang_vel + np.random.uniform(-0.2, 0.2, size=3)
            joint_pos_rel = joint_pos_rel + np.random.uniform(-0.01, 0.01, size=self.num_dof)
            joint_vel_rel = joint_vel_rel + np.random.uniform(-0.5, 0.5, size=self.num_dof)
        
        # 3. Append to history buffers
        self.base_ang_vel_hist.append(base_ang_vel)
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
        self.robot_q = torch.from_numpy(q).unsqueeze(0).float()

    def _update_command(self):
        self.time_steps += 1

        print(self.time_steps, torch.where(self.time_steps >= self.motion.time_step_total -1)[0])

        env_ids = torch.where(self.time_steps >= self.motion.time_step_total -1)[0]  
        env_ids = torch.cat((env_ids, torch.where(self.time_steps >= self.stop_step)[0] ))

        return len(env_ids)>0
      



def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")


    # Step 1: Define the path to your ONNX model
    # /home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_11-48-16_xagile_test_run

    # onnx_model_path = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-15_14-45-29_dry_run/exported/policy.onnx" ##  BEST      
    onnx_model_path = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-24_16-21-37_lefan_walk_padded/exported/policy.onnx"    

    # onnx_model_path = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/logs/rsl_rl/g1_flat/2025-08-20_11-48-16_xagile_test_run/exported/policy.onnx"

    xml_path = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1_mine.xml"
    
    # Step 2: Create an Inference Session
    try:
        # Use GPU provider if available, otherwise fall back to CPU
        session = ort.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    except ort.SessionCreationError:
        # Fallback to CPU if GPU is not available
        session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

    print(f"ONNX Runtime providers: {session.get_providers()}")

    # Step 3: Get input and output details
    # This is crucial for knowing the shape and data type of the input and output tensors.
    input_name = session.get_inputs()[0].name

    output_name = session.get_outputs()[0].name    
    output_shape = session.get_outputs()[0].shape

    input_shape = session.get_inputs()[0].shape
    input_dtype = session.get_inputs()[0].type


    print(f"Input Name: {input_name}, Shape: {input_shape}, Dtype: {input_dtype}")
    print(f"Output Name: {output_name}, Shape {output_shape}")

    model_meta = session.get_modelmeta()

    print("--- Custom Metadata from the ONNX Runtime Session ---")
    for key, value in model_meta.custom_metadata_map.items():
        print(f"{key}: {value}")

    kds = np.array(model_meta.custom_metadata_map['joint_damping'].split(','), dtype=float)  
    kps = np.array(model_meta.custom_metadata_map['joint_stiffness'].split(','), dtype=float) 
    action_scale = np.array(model_meta.custom_metadata_map['action_scale'].split(','), dtype=float) 
    default_angles = np.array(model_meta.custom_metadata_map['default_joint_pos'].split(','), dtype=float) 
    policy_joint_names = model_meta.custom_metadata_map['joint_names'].split(',')    
    effort_limit = np.array(model_meta.custom_metadata_map['joint_effort_limit'].split(','), dtype=float) 
   
    print(policy_joint_names, len(policy_joint_names))
    print([(policy_joint_names[i], i) for i in range(len(policy_joint_names)) if 'waist' in policy_joint_names[i]])
    input()

    num_actions = output_shape
    num_obs = input_shape

    # /home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/tmp/0-KIT_3_walk_6m_straight_line04_poses.npz
    # /home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/tmp/walk1_subject1.npz'
    config = {
        'sim': {
            'dt': 0.005
        },
        'decimation': 4,
        'motion_file': '/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/tmp/walk1_subject1.npz',
        'anchor_body': 'torso_link',
        'body_names': [
            'pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link', 'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link', 'torso_link',
            'left_shoulder_roll_link', 'left_elbow_link', 'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link', 'right_wrist_yaw_link',
        ]
    }

    simulation_dt = config["sim"]["dt"]
    control_decimation = config["decimation"]
    
    motion_cfg = MotionCommandCfg(
        motion_file=config['motion_file'],
        anchor_body=config['anchor_body'],
        body_names=config['body_names'],
        motion_start=150,
        run_length=200
    )

    simulation_duration = config["simulation_duration"] = 1000

    command_manager = MotionCommand(motion_cfg) 



    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    filtered_action = action.copy()
    target_dof_pos = default_angles.copy()

    counter = 0
    control_step_counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Corrected way to get the joint names
    # We loop through all the joints using their index (from 0 to njnt-1)
    joint_names = []
    for i in range(m.njnt):
        # Use mj_id2name to get the name for the given joint index
        # We specify the object type as 'mjOBJ_JOINT'
        name_bytes = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        # The name is returned as a byte string, so we decode it
        name_str = name_bytes
        joint_names.append(name_str)

    joint_names = joint_names[1:]

    mj_to_policy = [joint_names.index(n) for n in policy_joint_names]    
    policy_to_mj = [policy_joint_names.index(n) for n in joint_names]

    d.qpos = np.concatenate(
        [
            np.array([0,0, 0.76], dtype=np.float32),
            np.array([1,0,0,0], dtype=np.float32),
            default_angles[policy_to_mj],
        ]
    )
    mujoco.mj_forward(m, d)

    motion_ended = False
    warmed_up = False
    warm_up_time = 100
    transition_time = 100


    obs_manager = ObservationManager(
        m, d, history_length=1,
        default_angles=default_angles,
        mj_to_policy=mj_to_policy,
        use_noise=False
    )

    logger = JointDataLogger(num_joints=29, torque_limits=effort_limit)
    current_state = MotionState.WARMING_UP
    SAVED=False

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.cam.elevation = -20
        # Set the camera azimuth (rotate around the model)
        viewer.cam.azimuth = 150
        viewer.cam.distance = 4
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
      

          

            counter +=1
            if counter % control_decimation == 0:
                control_step_counter += 1

                target_positions = action[0, :] * action_scale[:] + default_angles[:]
                actual_positions = d.qpos[7:][mj_to_policy]
                velocities = d.qvel[6:][mj_to_policy]
                torques = d.ctrl[mj_to_policy]
                # Record the data for this timestep
                logger.record_snapshot(target_positions, actual_positions, velocities, torques)



                if current_state == MotionState.WARMING_UP:
                    print(f'WARMING UP ----------------- {control_step_counter}')
                    task_command_angles = np.copy(default_angles) # Hold default pose
                    task_command_vels = np.zeros_like(task_command_angles)
                    task_motion_ori = torch.tensor([[1,0,0,0]])

                    if control_step_counter >= warm_up_time:
                        task_command_angles  = command_manager.command[0, :29].numpy()
                        first_task_command_angle = task_command_angles.copy()
                   
                        # Change state and reset counter for interpolation
                        current_state = MotionState.TRANSITION_TO_MOTION
                        interpolation_counter = 0

                # --- State: TRANSITION_TO_MOTION ---
                elif current_state == MotionState.TRANSITION_TO_MOTION:
                    print(f'TRANSITION_TO_MOTION----------------- {control_step_counter}')
                    alpha = min(interpolation_counter / transition_time, 1.0)
                    task_command_angles = (1 - alpha) * default_angles + alpha * first_task_command_angle
                    interpolation_counter += 1
                    
                    if interpolation_counter >= transition_time:
                        print("Transition complete. Executing motion.")
                        current_state = MotionState.EXECUTING_MOTION
                        

                elif current_state == MotionState.EXECUTING_MOTION:
                    print(f'EXECUTING_MOTION ----------------- {control_step_counter}')
                    task_command_angles = command_manager.command[0, :29].numpy()
                    task_command_vels = command_manager.command[0, 29:].numpy()
                    task_motion_ori = command_manager.anchor_quat_w
                    # Check for the end condition
                    if motion_ended:
                        final_task_command_angle = task_command_angles.copy()
                        
                        current_state = MotionState.TRANSITION_TO_DEFAULT
                        interpolation_counter = 0

                # --- State: TRANSITION_TO_DEFAULT ---
                elif current_state == MotionState.TRANSITION_TO_DEFAULT:
                    print(f'TRANSITION_TO_DEFAULT ----------------- {control_step_counter}')
                    alpha = min(interpolation_counter / transition_time, 1.0)
                    # Interpolate from the last policy pose back to the default
                    task_command_angles = (1 - alpha) * final_task_command_angle + alpha *  np.copy(default_angles)
                    task_command_vels = np.zeros_like(task_command_angles)
                    task_motion_ori = torch.tensor([[1,0,0,0]])
                    interpolation_counter += 1

                    if interpolation_counter >= transition_time:
                        print("Transition to default complete. Motion ended.")
                        ended_counter = 0
                        current_state = MotionState.ENDED
                
                # --- State: ENDED ---
                elif current_state == MotionState.ENDED:
                    print(f'ENDED ----------------- {control_step_counter}')
                    task_command_angles = np.copy(default_angles) # Hold default pose
                    ended_counter += 1

                    if ended_counter > 100:
                        current_state = MotionState.SAVE





                #################################                
                #################################
                #################################
                #################################
                #################################

    
                
                
                q = d.sensor("base_quat").data.astype(np.float32)

                pos = d.qpos[7:][mj_to_policy]
                waist_yaw = pos[2].item()
                waist_pitch = pos[8].item()
                waist_roll = pos[5].item()

                q = transform_imu_data_from_pelvis_to_torso_quat(
                    waist_roll=waist_roll, 
                    waist_pitch=waist_pitch, 
                    waist_yaw=waist_yaw, 
                    pelvis_quat=q, 
                )
        
                command_manager._set_q(q)

                task_command = torch.cat([torch.tensor(task_command_angles).unsqueeze(0), torch.tensor(task_command_vels).unsqueeze(0)], dim=1)
                _, ori = subtract_frame_transforms(
                    command_manager.robot_anchor_pos_w,
                    command_manager.robot_anchor_quat_w,
                    command_manager.anchor_pos_w,
                    task_motion_ori,
                )
                mat = matrix_from_quat(ori)

                motion_anchor_ori_b = mat[..., :2].reshape(mat.shape[0], -1)



                # if motion_ended or not warmed_up:
                #     task_command = torch.cat([torch.tensor(default_angles).unsqueeze(0), torch.zeros(1, default_angles.shape[0])], dim=1)
                #     motion_anchor_pos_b = torch.zeros_like(motion_anchor_pos_b)
                #     _, ori = subtract_frame_transforms(
                #         command_manager.robot_anchor_pos_w,
                #         command_manager.robot_anchor_quat_w,
                #         command_manager.anchor_pos_w,
                #         torch.tensor([[1,0,0,0]]),
                #     )
                #     mat = matrix_from_quat(ori)
                #     motion_anchor_ori_b = mat[..., :2].reshape(mat.shape[0], -1)
                
 
              
                # 3. Move the tensor to the GPU and add the batch dimension
                obs_manager.update(action.copy()[0])

                # 2. Create a PyTorch tensor from the NumPy array (it will be on CPU)
                proprio_cpu = torch.from_numpy(obs_manager.get_observation())
                proprio = proprio_cpu.unsqueeze(0)

                obs = torch.cat((
                    task_command, 
                    motion_anchor_ori_b, 
                    proprio
                ), dim=-1)

                # policy inference
                # Create a dictionary to map input names to your data
                input_dict = {
                    "obs": obs.numpy().astype(np.float32),
                    "time_step": torch.tensor([[0]]).float().numpy().astype(np.float32)
                }

                action = session.run(['actions'], input_dict)[0]

                # filtered_action = filtered_action * 0. + action * 1
                target_dof_pos = (action * action_scale + default_angles)[0]

                if current_state == MotionState.EXECUTING_MOTION:
                    motion_ended = command_manager._update_command()
              
                if current_state == MotionState.SAVE and not SAVED:
                    SAVED=True
                    logger.save_to_csv("robot_joint_data.csv")
                    logger.plot_torque_individually()
                    logger.plot_data_per_joint(save_plots=True)



            tau = pd_control(target_dof_pos[policy_to_mj], d.qpos[7:], kps[policy_to_mj], np.zeros_like(kds), d.qvel[6:], kds[policy_to_mj])
       
            torque_limits = m.actuator_ctrlrange
            min_torques = torque_limits[:, 0]
            max_torques = torque_limits[:, 1]
      
            clipped_tau = np.clip(tau, -effort_limit[policy_to_mj], effort_limit[policy_to_mj])

            d.ctrl[:] = clipped_tau
            mujoco.mj_step(m, d)


         
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
