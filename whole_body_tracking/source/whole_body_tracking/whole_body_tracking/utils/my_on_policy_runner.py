import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from rsl_rl.utils import store_code_state
import time
from collections import deque
import torch
from torch import nn
from typing import List, Tuple, Dict, Optional
from whole_body_tracking.tasks.tracking.mdp.multi_motion_commands import MultiMotionCommand
import joblib
from tqdm import tqdm
import numpy as np
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE
from torch.distributions import Normal
from isaaclab.managers import SceneEntityCfg

class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    

    @torch.no_grad()
    def calc_eval_metrics(self) -> Tuple[Dict, Optional[float]]:

        self.eval_mode()

        env = self.env.unwrapped
        env.set_reset_conf(disable=True)


        command_manager = env.command_manager.get_term('motion')
        num_motions = command_manager.motion.num_motions

        assert isinstance(command_manager, MultiMotionCommand)
        metrics = {
            # Track which motions are evaluated (within time limit)
            "evaluated": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
            "ground_contact": torch.zeros(num_motions, device=self.device, dtype=torch.bool),
        }
        for k in command_manager.metrics.keys():
            metrics[k] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_max"] = torch.zeros(num_motions, device=self.device)
            metrics[f"{k}_min"] = torch.ones(num_motions, device=self.device)

      
        iterations = (num_motions // self.env.num_envs) + 1
    
        motion_ids = torch.arange(num_motions, device=self.device, dtype=torch.long)

        all_motion_lengths = []

        contact_sensor = env.scene.sensors['contact_forces']
        sensor_cfg = SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "pelvis", '.*shoulder.*'
                ],
            )
        sensor_cfg.resolve(env.scene)
    

        for iter in tqdm(range(iterations)):
            ################# Deterimne the motion ids
            start_index = iter*self.env.num_envs
            end_index = (iter+1)*self.env.num_envs

            indices = torch.arange(start_index, end_index) % num_motions
            num_motions_this_iter = min(end_index, num_motions) - start_index 
            iter_motion_ids = motion_ids[indices]


            ################# Manually hard reset the env
            command_manager.motion.sample_motions(motion_ids=iter_motion_ids)
            
            #Assumes the motion data has the same dt as the env
            motion_lengths = command_manager.motion.lengths
            all_motion_lengths.append(motion_lengths[:num_motions_this_iter])

            max_len = motion_lengths.max().item()

            obs = self.env.reset()
            # Resets the robot state

            command_manager._resample_command(
                env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long),
                phase = torch.zeros((self.env.num_envs), device=self.device, dtype=torch.float32)
            )
            elapsed_time = 0
            # update articulation kinematics
            env.scene.write_data_to_sim()
            env.sim.forward()
            command_manager.compute_robot_humanoid_obs_full()
            command_manager.compute_mimic_obs_full()

            #################################### Get the initial observations and step set up vars
            obs, extras = self.env.get_observations()
            obs = self.obs_normalizer(obs)
            # Disable automatic reset to maintain consistency in evaluation.         


            for l in range(max_len-1):
                print(f'---- Step {l} ---- of {max_len} ---- ')
                actions = self.alg.policy.act_inference(obs) #[bs, num_actions]

                obs, rewards, dones, extras = self.env.step(actions)
                obs = self.obs_normalizer(obs)

                elapsed_time += 1
                # clip_done = elapsed_time >= command_manager.motion.lengths
                # clip_not_done = torch.logical_not(clip_done).cpu()
                # clip_not_done[num_motions_this_iter:] = 0

                # ground_contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.).any(dim=-1)
         
                # for k in command_manager.metrics.keys():
                #     value = command_manager.metrics[k]
             
                #     # Only update metrics for motions that are continuing.
          
                #     metrics[k][iter_motion_ids[clip_not_done]] += value[clip_not_done]

                #     metrics[f"{k}_max"][iter_motion_ids[clip_not_done]] = torch.maximum(
                #         metrics[f"{k}_max"][iter_motion_ids[clip_not_done]],
                #         value[clip_not_done],
                #     )
                #     metrics[f"{k}_min"][iter_motion_ids[clip_not_done]] = torch.minimum(
                #         metrics[f"{k}_min"][iter_motion_ids[clip_not_done]],
                #         value[clip_not_done],
                #     )

                # metrics['ground_contact'][iter_motion_ids[clip_not_done]] |= ground_contact[clip_not_done]

           

            
     

        ###############################        
        ################################
        ###############################
        ###############################

        to_log = {}
        for k in command_manager.metrics.keys():
            mean_tracking_errors = metrics[k] / (all_motion_lengths)
            to_log[f"eval/{k}"] = mean_tracking_errors.detach().mean().item()
            to_log[f"eval/{k}_max"] = metrics[f"{k}_max"].detach().mean().item()
            to_log[f"eval/{k}_min"] = metrics[f"{k}_min"].detach().mean().item()

        if "gt_err" in command_manager.metrics.keys():
            tracking_failures = ((metrics["gt_err_max"] > 0.5) | metrics['ground_contact']).float()
            to_log["eval/tracking_success_rate"] = 1.0 - tracking_failures.detach().mean().item()
            failed_motions = torch.nonzero(tracking_failures).flatten().tolist()
            to_log['failed_motions'] = failed_motions
            

        os.makedirs(f'{self.log_dir}/eval', exist_ok=True)
        joblib.dump(to_log, f'{self.log_dir}/eval/log.pkl')

        env.set_reset_conf(disable=False)
        self.env.reset()

        return to_log, to_log.get("eval/tracking_success_rate")



    @torch.no_grad()
    def collect_dataset(self, action_noise_level):
        self.eval_mode()

        env = self.env.unwrapped
        env.set_reset_conf(disable=True)


        command_manager = env.command_manager.get_term('motion')

        assert isinstance(command_manager, MultiMotionCommand)
        num_motions = command_manager.motion.num_motions

        iterations = (num_motions // self.env.num_envs) + 1
        
        motion_ids = torch.arange(num_motions, device=self.device, dtype=torch.long)
        for iter in tqdm(range(iterations)):
            start_index = iter*self.env.num_envs
            end_index = (iter+1)*self.env.num_envs

            indices = torch.arange(start_index, end_index) % num_motions
            iter_motion_ids = motion_ids[indices]
    
            
            output_data = []
            for eval_episode in range(5):
                command_manager.motion.sample_motions(motion_ids=iter_motion_ids)
            
                #Assumes the motion data has the same dt as the env
                motion_lengths = command_manager.motion.lengths
                max_len = motion_lengths.max().item()

                obs = self.env.reset()
                # Resets the robot state

                command_manager._resample_command(
                    env_ids = torch.arange(self.env.num_envs, device=self.device, dtype=torch.long),
                    phase = torch.zeros((self.env.num_envs), device=self.device, dtype=torch.float32)
                )
                elapsed_time = 0
                # update articulation kinematics
                env.scene.write_data_to_sim()
                env.sim.forward()
                command_manager.compute_robot_humanoid_obs_full()
                command_manager.compute_mimic_obs_full()

                obs, extras = self.env.get_observations()
                obs = self.obs_normalizer(obs)
                # Disable automatic reset to maintain consistency in evaluation.

                data_collection_obs = extras["observations"]['data_collection_obs']

                # must collect sensor obs, action-1
                data_collection_obs_all = torch.zeros((
                    self.env.num_envs, max_len,  data_collection_obs.size(1)
                ), dtype=data_collection_obs.dtype, device='cpu')

                data_collection_obs_all[:, elapsed_time, :] = data_collection_obs.detach().cpu()                  
 
                # The list of all active action term objects
                action_term = env.action_manager._terms['joint_pos']

                for l in range(max_len-1):
                    print(f'---- Step {l} ---- of {max_len} ---- ITER: {eval_episode}')
                    actions = self.alg.policy.act_inference(obs) #[bs, num_actions]

                    normal_dist = Normal(loc=actions, scale=action_noise_level/action_term._scale)
                    noised_actions = normal_dist.sample()

                    obs, rewards, dones, extras = self.env.step(noised_actions)
                    obs = self.obs_normalizer(obs)

                    data_collection_obs = extras["observations"]['data_collection_obs']  
                    #replace data collection action (noised actions) with the clean actions 
                    data_collection_obs[:, :29] = actions.clone()
   
                    elapsed_time += 1
                    clip_done = elapsed_time >= command_manager.motion.lengths
                    clip_not_done = torch.logical_not(clip_done).cpu()

                    data_collection_obs_all[clip_not_done, elapsed_time, :] = data_collection_obs.detach().cpu()[clip_not_done].clone()               
             
                        
                #Save the data 
                for env_id in tqdm(range(self.env.num_envs), desc='Creating motion dicts'):
                    d = {}
                    d['motion'] = data_collection_obs_all[env_id][:command_manager.motion.lengths[env_id], :].numpy()                    
                    d['motion_id'] = iter_motion_ids[env_id]
                    d['caption'] = 'Still need to impliment this'
                    d['length'] = command_manager.motion.lengths[env_id]
                    output_data.append(d)

            os.makedirs(f'{self.log_dir}/collection/v3/{action_noise_level}', exist_ok=True)
            joblib.dump(output_data, f'{self.log_dir}/collection/v3/{action_noise_level}/robot_sa_dataset_r5.pkl')
            print('Dataset saved ============================')


                

        env.disable_reset = False

        return 


