from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
    quat_rotate,
    matrix_from_quat
)

import joblib
import time

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def to_tan_norm(quat):
    mat = matrix_from_quat(quat)
    return mat[..., :2].reshape(mat.shape[0], -1)

class MultiMultionLoader:

    def __init__(self, 
                 motion_file: str, 
                 body_indexes: Sequence[int], 
                 num_envs: int,
                 device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"

        self.motion_data = joblib.load(motion_file)
        self.num_envs = num_envs
        self.num_motions = len(self.motion_data)
        self.device = device
        self._body_indexes = body_indexes

        #TODO add to configs -- need to reconsider this for the data collection
        self.max_motion_length = 100

        self.sample_motions()

    def sample_motions(self, motion_ids=None):
        start_time = time.perf_counter()

        self.fps = []
        self.lengths = []

        joint_pos = []
        joint_vel = []
        body_pos_w = []
        body_quat_w = []
        body_lin_vel_w = []
        body_ang_vel_w = []


        if motion_ids is None:
            sampled_indices = torch.randint(low=0, high=self.num_motions, size=(self.num_envs,))
        else:
            sampled_indices = motion_ids
            
        for i, motion_idx in enumerate(sampled_indices):
            print(i, len(self.motion_data[motion_idx]['joint_pos']))
            if len(self.motion_data[motion_idx]['joint_pos']) > self.max_motion_length:
                start_idx = np.random.randint(0, len(self.motion_data[motion_idx]['joint_pos'])-self.max_motion_length )
                end_idx = start_idx + self.max_motion_length
            else:
                start_idx = 0
                end_idx = len(self.motion_data[motion_idx]['joint_pos'])

            joint_pos.append(torch.from_numpy(self.motion_data[motion_idx]['joint_pos'][start_idx:end_idx]).to(self.device, dtype=torch.float32))            
            joint_vel.append(torch.from_numpy(self.motion_data[motion_idx]['joint_vel'][start_idx:end_idx]).to(self.device, dtype=torch.float32))

            body_pos_w.append(torch.from_numpy(self.motion_data[motion_idx]['body_pos_w'][start_idx:end_idx]).to(self.device, dtype=torch.float32)[:, self._body_indexes])            
            body_quat_w.append(torch.from_numpy(self.motion_data[motion_idx]['body_quat_w'][start_idx:end_idx]).to(self.device, dtype=torch.float32)[:, self._body_indexes])
            body_lin_vel_w.append(torch.from_numpy(self.motion_data[motion_idx]['body_lin_vel_w'][start_idx:end_idx]).to(self.device, dtype=torch.float32)[:, self._body_indexes])
            body_ang_vel_w.append(torch.from_numpy(self.motion_data[motion_idx]['body_ang_vel_w'][start_idx:end_idx]).to(self.device, dtype=torch.float32)[:, self._body_indexes])

            self.lengths.append(end_idx-start_idx)

            self.fps.append(50) # TODO this need to be correct

        self.joint_pos = torch.cat(joint_pos)
        self.joint_vel = torch.cat(joint_vel)     

        self._body_pos_w = torch.cat(body_pos_w)
        self._body_quat_w = torch.cat(body_quat_w)
        self._body_lin_vel_w = torch.cat(body_lin_vel_w)
        self._body_ang_vel_w = torch.cat(body_ang_vel_w)

        self.lengths = torch.tensor(self.lengths)

        lengths_shifted = self.lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts =  lengths_shifted.cumsum(0).to(dtype=torch.long, device=self.device)
        self.lengths = self.lengths.to(self.device)

        # 3. Record the end time
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Sampling new motions took: {elapsed_time:.4f} seconds")

        return 
    
    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w



class MultiMotionCommand(CommandTerm):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
 
        self.env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion = MultiMultionLoader(self.cfg.motion_file, self.body_indexes, num_envs=env.num_envs, device=self.device)
        self.num_future_steps = self.cfg.num_future_steps
        self.future_fps = self.cfg.future_fps
        self.motion_fps = self.motion.fps[0]
        self.use_futures = self.cfg.use_futures
   
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_idxs = self.motion.length_starts.clone()
        self.frame_idxs_futures = self._calc_futures_frame_idxs()

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        #### HUMANOID OBS PRE COMPUTE
        self.robot_root_h =  torch.zeros((self.num_envs,1), dtype=torch.float32, device=self.device)        
        self.robot_local_body_pos =  torch.zeros((self.num_envs,len(cfg.body_names)*3), dtype=torch.float32, device=self.device)
        self.robot_local_body_rots =  torch.zeros((self.num_envs, len(cfg.body_names)*6), dtype=torch.float32, device=self.device)
        self.robot_local_body_vel =  torch.zeros((self.num_envs,len(cfg.body_names)*3), dtype=torch.float32, device=self.device)
        self.robot_local_body_ang_vel =  torch.zeros((self.num_envs,len(cfg.body_names)*3), dtype=torch.float32, device=self.device)

        self.motion_local_body_pos =  torch.zeros((self.num_envs,self.num_future_steps*len(cfg.body_names)*3), dtype=torch.float32, device=self.device)
        self.motion_relative_body_pos =  torch.zeros((self.num_envs, self.num_future_steps*len(cfg.body_names)*3), dtype=torch.float32, device=self.device)
        self.motion_local_body_rots =  torch.zeros((self.num_envs, self.num_future_steps*len(cfg.body_names)*6), dtype=torch.float32, device=self.device)
        self.motion_relative_body_rots =  torch.zeros((self.num_envs, self.num_future_steps*len(cfg.body_names)*6), dtype=torch.float32, device=self.device)
   
        self.last_quat_w = torch.zeros((self.num_envs, 4), device=self.device)        
        self.quat_w = torch.zeros((self.num_envs, 4), device=self.device)


        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gt_err"] = torch.zeros(self.num_envs, device=self.device)



    def _calc_futures_frame_idxs(self):
        if self.use_futures:
            # size -> (num_future_steps, )
            time_offsets = (
                torch.arange(
                    1, self.num_future_steps + 1, device=self.device, dtype=torch.long
                )
                * (1/self.future_fps)*self.motion_fps
            )

            # time steps : size -> (num envs,)
            # size -> (num envs, num_future_steps)
            future_times = self.time_steps.unsqueeze(1)+time_offsets.unsqueeze(0)

            # size -> (num_envs * num_future_steps, )
            clipped_time =  torch.minimum(
                future_times.view(-1), 
                self.motion.lengths.unsqueeze(-1).tile([1, self.num_future_steps]).view(-1)-1
            )

            # size -> (num_envs * num_future_steps, )
            frame_index_future = clipped_time + self.motion.length_starts.unsqueeze(-1).tile([1, self.num_future_steps]).view(-1)
         
            assert (clipped_time < self.motion.lengths.unsqueeze(-1).tile([1, self.num_future_steps]).view(-1)).all()
        else:
            frame_index_future = self.frame_idxs.clone()

        return frame_index_future.long()
    


    ########################################### BASE VARS END    

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.frame_idxs]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.frame_idxs]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_idxs] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_idxs]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_idxs]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_idxs]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_idxs, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_idxs, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_idxs, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_idxs, self.motion_anchor_body_index]
    
    ########################################### BASE VARS END







    ########################################### VARS WITH FUTURES 

    @property
    def joint_pos_futures(self) -> torch.Tensor:
        return self.motion.joint_pos[self.frame_idxs_futures]

    @property
    def joint_vel_futures(self) -> torch.Tensor:
        return self.motion.joint_vel[self.frame_idxs_futures]

    @property
    def body_pos_w_futures(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_idxs_futures] \
            + self._env.scene.env_origins.unsqueeze(1).tile([1, self.num_future_steps, 1]).flatten(0,1)[:, None, :]

    @property
    def body_quat_w_futures(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_idxs_futures]

    @property
    def body_lin_vel_w_futures(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_idxs_futures]

    @property
    def body_ang_vel_w_futures(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_idxs_futures]

    @property
    def anchor_pos_w_futures(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.frame_idxs_futures, self.motion_anchor_body_index] \
            + self._env.scene.env_origins.unsqueeze(1).tile([1, self.num_future_steps, 1]).flatten(0,1)

    @property
    def anchor_quat_w_futures(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.frame_idxs_futures, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w_futures(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.frame_idxs_futures, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w_futures(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.frame_idxs_futures, self.motion_anchor_body_index]

    ########################################### VARS WITH FUTURES END 







    ########################################### ROBOT STATE VARS  


    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    ########################################### ROBOT STATE VARS END
    

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        self.metrics["gt_err"] =  torch.norm(self.body_pos_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)


    def resample_motion(self):
        self.motion.sample_motions()
        self._resample_command(torch.arange(self.num_envs, device=self.device, dtype=torch.long))
        self._update_command()
   
        return 

    def _resample_command(self, env_ids: Sequence[int], phase=None):

        if phase is None:
            phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)


        self.time_steps[env_ids] = (phase * (self.motion.lengths[env_ids] - 1)).long()
        self.frame_idxs[env_ids] = self.time_steps[env_ids] + self.motion.length_starts[env_ids]

        start_indices = env_ids * self.num_future_steps
        env_offsets = torch.arange(0, self.num_future_steps, device=self.device, dtype=torch.long)
        fut_env_ids = start_indices.unsqueeze(-1) + env_offsets
        fut_env_ids = fut_env_ids.view(-1)
        self.frame_idxs_futures[fut_env_ids] = self._calc_futures_frame_idxs()[fut_env_ids]

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )

        #reset in the deault pose
        if not self.env.disable_default_pose_reset:
            default_pose_mask = torch.rand(env_ids.shape) < self.cfg.p_init_default_pose
            joint_pos[env_ids[default_pose_mask]] = self.robot.data.default_joint_pos[env_ids[default_pose_mask]]        
            joint_vel[env_ids[default_pose_mask]] = 0.
            root_pos[env_ids[default_pose_mask]][:, 2] = 0.8      
            root_ori[env_ids[default_pose_mask]] = self.robot.data.default_root_state[env_ids[default_pose_mask], 3:7]        
            root_lin_vel[env_ids[default_pose_mask]] = self.robot.data.default_root_state[env_ids[default_pose_mask], 7:10]
            root_ang_vel[env_ids[default_pose_mask]] = self.robot.data.default_root_state[env_ids[default_pose_mask], 10:13]


        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

        self.last_quat_w[env_ids] = root_ori[env_ids]
        self.quat_w[env_ids] = root_ori[env_ids]

    def _update_command(self):
        self.time_steps += 1
        clipped_time =  torch.minimum(
                self.time_steps.view(-1), 
                self.motion.lengths-1
            )
        self.frame_idxs = clipped_time + self.motion.length_starts
        self.frame_idxs_futures = self._calc_futures_frame_idxs()

        if not self.env.disable_resets:
            env_ids = torch.where(self.time_steps >= self.motion.lengths)[0]
            self._resample_command(env_ids)


        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = anchor_pos_w_repeat - robot_anchor_pos_w_repeat
        delta_pos_w[..., :2] = 0.0
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = (
            robot_anchor_pos_w_repeat + delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)
        )

        asset = self.env.scene['robot']
        self.last_quat_w[:]= self.quat_w[:]
        self.quat_w = asset.data.root_quat_w

        self.compute_robot_humanoid_obs_full()
        self.compute_mimic_obs_full()


    
    def compute_robot_humanoid_obs_full(self):

        root_pos = self.robot_anchor_pos_w
        root_rot = self.robot_anchor_quat_w

        body_pos = self.robot_body_pos_w
        body_rot = self.robot_body_quat_w
        body_vel = self.robot_body_lin_vel_w        
        body_ang_vel = self.robot_body_ang_vel_w


        root_h = root_pos[:, 2:3]
        heading_rot = quat_inv(yaw_quat(root_rot))


        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(
            heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
            heading_rot_expand.shape[2],
        )

        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(
            local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
        )

        flat_local_body_pos = quat_rotate(
            flat_heading_rot, flat_local_body_pos
        )

        local_body_pos = flat_local_body_pos.reshape(
            local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
        )

        flat_body_rot = body_rot.reshape(
            body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
        )
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(
            body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
        )

        root_rot_obs = to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(
            body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
        )
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(
            body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
        )

        flat_body_ang_vel = body_ang_vel.reshape(
            body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
        )
        flat_local_body_ang_vel = quat_rotate(
            flat_heading_rot, flat_body_ang_vel
        )
        local_body_ang_vel = flat_local_body_ang_vel.reshape(
            body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
        )

        self.robot_root_h[:] = root_h[:]     
        self.robot_local_body_pos[:] = local_body_pos[:]
        self.robot_local_body_rots[:] = local_body_rot_obs[:]
        self.robot_local_body_vel[:] =  local_body_vel[:]
        self.robot_local_body_ang_vel[:] = local_body_ang_vel[:]


   
        return 
    

    def compute_mimic_obs_full(self):
        num_envs = self.num_envs
        num_future_steps = self.num_future_steps

        cur_gt = self.robot_body_pos_w
        cur_gr = self.robot_body_quat_w
        root_pos = self.robot_anchor_pos_w.unsqueeze(1).tile([1, self.num_future_steps, 1]).flatten(0,1)
        root_rot = self.robot_anchor_quat_w.unsqueeze(1).tile([1, self.num_future_steps, 1]).flatten(0,1)

        flat_target_pos = self.body_pos_w_futures
        flat_target_rot = self.body_quat_w_futures

        expanded_body_pos = cur_gt.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_gt.shape[1:]
        )
        expanded_body_rot = cur_gr.unsqueeze(1).expand(
            num_envs, num_future_steps, *cur_gr.shape[1:]
        )

        flat_cur_pos = expanded_body_pos.reshape(flat_target_pos.shape)
        flat_cur_rot = expanded_body_rot.reshape(flat_target_rot.shape)


        heading_inv_rot = quat_inv(yaw_quat(root_rot))

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat(
            (1, flat_cur_pos.shape[1], 1)
        )
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(
            heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1],
            heading_inv_rot_expand.shape[2],
        )

        root_pos_expand = root_pos.unsqueeze(-2)

        """target"""
        # target body pos   [N, 3xB]
        flat_target_body_pos = (flat_target_pos - root_pos_expand).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
        )
        flat_target_body_pos = quat_rotate(
            flat_heading_inv_rot, flat_target_body_pos
        )
        target_body_pos = flat_target_body_pos.reshape(num_envs, num_future_steps, -1)

        flat_target_body_pos_rel = (flat_target_pos - flat_cur_pos).reshape(
            flat_target_pos.shape[0] * flat_target_pos.shape[1], flat_target_pos.shape[2]
        )
        flat_target_body_pos_rel = quat_rotate(
            flat_heading_inv_rot, flat_target_body_pos_rel
        )
        target_body_pos_rel = flat_target_body_pos_rel.reshape(
            num_envs, num_future_steps, -1
        )

        # target body rot   [N, 6xB]
        target_body_rot = quat_mul(
            heading_inv_rot_expand, flat_target_rot
        )

        target_body_rot_obs = to_tan_norm(
            target_body_rot.view(-1, 4)
        ).reshape(num_envs, num_future_steps, -1)

        target_rel_body_rot = quat_mul(
            quat_inv(flat_cur_rot), flat_target_rot
        )
        target_rel_body_rot_obs = to_tan_norm(
            target_rel_body_rot.view(-1, 4)
        ).reshape(num_envs, num_future_steps, -1)

        # print(self.motion_relative_body_pos.size() )       
        # print(target_body_pos_rel.size())

        self.motion_local_body_pos[:] = target_body_pos.flatten(1,2)[:]
        self.motion_relative_body_pos[:] = target_body_pos_rel.flatten(1,2)[:]
        self.motion_local_body_rots[:] = target_body_rot_obs.flatten(1,2)[:]
        self.motion_relative_body_rots[:] = target_rel_body_rot_obs.flatten(1,2)[:]

        return 




    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    p_init_default_pose: float  =0.1

    num_future_steps: int = MISSING
    future_fps: int = MISSING
    use_futures: bool = MISSING

