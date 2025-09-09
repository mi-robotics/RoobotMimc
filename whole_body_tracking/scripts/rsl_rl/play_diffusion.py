"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")

parser.add_argument(
    "--diffusion_loadrun", type=str, required=True, default=None, help="Diffusion model saved location"
)
parser.add_argument(
    "--diffusion_cp", type=int, required=True, default=None, help="Diffusion model saved location"
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

from data_loaders.dataset import extract_observation_terms
from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf




def load_diffusion_policy(diffusion_load_run, checkpoint, num_envs):

    from data_loaders.get_data import get_dataset_loader
    from data_loaders.dataset import get_features_dims, RobotStateActionDataset
    from utils.model_util import create_model_and_diffusion, load_saved_model
    from utils import dist_util
    from diffusion.guidance.velocity_controller import VelocityGuidance

    log_root = '/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/MDM/log'
    cfg_path = f'{log_root}/{diffusion_load_run}/.hydra/resolved_config.yaml'
    model_path = f'{log_root}/{diffusion_load_run}/model{checkpoint:09}.pt'
    cfg = OmegaConf.load(cfg_path)
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.dataset.load_data = False
    cfg.dataset.load_mean_std = True


    dataset = RobotStateActionDataset(cfg)
    model, diffusion = create_model_and_diffusion(cfg, dataset)
    load_saved_model(model, model_path, use_avg=True)
    model.to('cuda:0')
    model.eval() 

    sample_fn = diffusion.p_sample_loop

    action_dim = get_features_dims(cfg.dataset.action_data_keys)
    robot_dim = get_features_dims(cfg.dataset.context_data_keys) if cfg.model.prediction_type == 'context' else get_features_dims(cfg.dataset.state_data_keys)
    features_dim = action_dim + robot_dim
    motion_shape = (num_envs, 1, features_dim, cfg.model.pred_len)

    try:
        cond_fn = eval(cfg.model.guidance_fn)(
            mean=dataset.s_mean,
            std=dataset.s_std
        ) if cfg.model.guidance_fn is not None else None
    except Exception as e :
        print(e)
        input()
        cond_fn = None

    policy = lambda model_kwargs: sample_fn(
            model,
            motion_shape,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=None,
            cond_fn_x_start=None
        )
    return dataset, policy, cfg


def load_edm_policy(diffusion_load_run, checkpoint, num_envs):

    #CM imports
    from cm import dist_util
    from cm.script_util import (
        NUM_CLASSES,
        model_and_diffusion_defaults,
        create_diffusion,

    )
    from cm.random_util import get_generator
    from cm.karras_diffusion import karras_sample

    #MDM imports
    from data_loaders.dataset import get_features_dims, RobotStateActionDataset
    from utils.model_util import create_model

    log_root = '/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/consistency_models/log'
    cfg_path = f'{log_root}/{diffusion_load_run}/.hydra/resolved_config.yaml'
    model_path = f'{log_root}/{diffusion_load_run}/model{checkpoint:09}.pt'
    cfg = OmegaConf.load(cfg_path)
    cfg = EasyDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.dataset.load_data = False
    cfg.dataset.load_mean_std = True

    dataset = RobotStateActionDataset(cfg)
    diffusion = create_diffusion(cfg, dataset)
    model = create_model(cfg)
    model.to(dist_util.dev())
    model.eval() 


    action_dim = get_features_dims(cfg.dataset.action_data_keys)
    robot_dim = get_features_dims(cfg.dataset.context_data_keys) if cfg.model.prediction_type == 'context' else get_features_dims(cfg.dataset.state_data_keys)
    features_dim = action_dim + robot_dim
    motion_shape = (num_envs, 1, features_dim, cfg.model.pred_len)

    if cfg.sample.sampler == "multistep":
        assert len(cfg.sample.ts) > 0
        ts = tuple(int(x) for x in cfg.sample.ts.split(","))
    else:
        ts = None

    generator = get_generator('dummy')

    policy = lambda model_kwargs: karras_sample(
            diffusion,
            model,
            motion_shape,
            steps=cfg.sample.steps,
            model_kwargs=model_kwargs,
            device=dist_util.dev(),
            clip_denoised=False,
            sampler=cfg.sample.sampler,
            sigma_min=cfg.sample.sigma_min,
            sigma_max=cfg.sample.sigma_max,
            s_churn=cfg.sample.s_churn,
            s_tmin=cfg.sample.s_tmin,
            s_tmax=cfg.sample.s_tmax,
            s_noise=cfg.sample.s_noise,
            generator=generator,
            ts=ts,
        )

    return dataset, policy, cfg


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.events.push_robot = None 
    env_cfg.commands.motion.p_init_default_pose = 1
    env_cfg.terminations.anchor_pos = None    
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.ee_body_pos = None


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    dataset, policy, diff_cfg = load_diffusion_policy(args_cli.diffusion_loadrun, args_cli.diffusion_cp, args_cli.num_envs)
    print('Dataset, Diffusion Policy and Configs Loaded ============================')


    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    env_cfg.commands.motion.motion_file = "/home/mcarroll/Documents/cd-2/humanoid_tracking/data/lefan_walk/packaged.pkl"        
    # env_cfg.commands.motion.motion_file = "/home/mcarroll/Documents/cd-2/humanoid_tracking/whole_body_tracking/tmp/padded_walk1_subject1.npz"

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    obs, extras = env.get_observations()
    diff_obs = extras['observations']['data_collection_obs']

    c_mean, c_std = torch.from_numpy(dataset.c_mean).to(diff_obs.device), torch.from_numpy(dataset.c_std).to(diff_obs.device)
    s_mean, s_std = torch.from_numpy(dataset.s_mean).to(diff_obs.device), torch.from_numpy(dataset.s_std).to(diff_obs.device)
    a_mean, a_std = torch.from_numpy(dataset.a_mean).to(diff_obs.device), torch.from_numpy(dataset.a_std).to(diff_obs.device)


    init_c = extract_observation_terms(diff_obs, diff_cfg.dataset.context_data_keys)
    init_c = (init_c - c_mean )/ c_std
    init_s = extract_observation_terms(diff_obs, diff_cfg.dataset.state_data_keys)
    init_s = (init_s - s_mean )/ s_std
    init_a = extract_observation_terms(diff_obs, diff_cfg.dataset.action_data_keys)
    init_a = (init_a - a_mean )/ a_std


    historical_c = torch.zeros((1, diff_cfg.model.context_len, c_mean.shape[0]), device=diff_obs.device)    
    historical_s = torch.zeros((1, diff_cfg.model.context_len, s_mean.shape[0]), device=diff_obs.device)
    historical_a = torch.zeros((1, diff_cfg.model.context_len, a_mean.shape[0]), device=diff_obs.device)


    historical_c[:, -1, :] = init_c.float()
    historical_s[:, -1, :] = init_s.float()
    historical_a[:, -1, :] = init_a.float()

    action_dim = a_mean.shape[0]

    print(action_dim)
    print(init_c.size())    
    print(init_a.size())
    print(historical_c.size())
    print(historical_a.size())
    print(historical_c.permute(0, 2, 1).unsqueeze(1).size())


    timestep = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.no_grad():
            model_kwargs = {'y':{}}
            model_kwargs['y']['mask'] = torch.ones((args_cli.num_envs, 1, 1, diff_cfg.model.pred_len), device=diff_obs.device, dtype=torch.bool)
            model_kwargs['y']['prefix_c'] = historical_c.permute(0, 2, 1).unsqueeze(1).clone()            
            model_kwargs['y']['prefix_s'] = historical_s.permute(0, 2, 1).unsqueeze(1).clone()
            model_kwargs['y']['prefix_a'] = historical_a.permute(0, 2, 1).unsqueeze(1).clone()

            #TODO target vel -- need to make this editable
            if 'target_vel' in diff_cfg.model.cond_mode:
                model_kwargs['y']['target_vel'] = torch.tensor([[0., 0., 0.]], device=diff_obs.device)
            elif diff_cfg.model.guidance_fn == 'VelocityGuidance':
                model_kwargs['y']['target_vel_guidance'] = torch.tensor([[0., 0.0]], device=diff_obs.device)

            model_kwargs['y']['action_dim'] = action_dim
            # agent stepping
            model_output = policy(model_kwargs)


            action_trajectory = model_output[:, :, :action_dim, :] # [bs, njoints, nfeats, nframes] 
            action_trajectory = dataset.inv_transform_a(action_trajectory)
            state_trajectory = dataset.inv_transform_s( model_output[:, :, action_dim:, :])
     
            for a_i in range(1):

                actions = action_trajectory[:, 0, :, a_i] #bs,num_actions

                if timestep < 4:
                    actions*=0
                    print('00000000000000000000000000000000000000000000000000000000000000', timestep)
         
                obs, _, _, extras = env.step(actions)

                diff_obs = extras['observations']['data_collection_obs']
                c = extract_observation_terms(diff_obs, diff_cfg.dataset.context_data_keys)
                c = ((c - c_mean) / c_std).unsqueeze(1).float()
                s = extract_observation_terms(diff_obs, diff_cfg.dataset.state_data_keys)
                print(s[:, dataset.velocity_indexes])
                asset = env.unwrapped.scene['robot']
                print(asset.data.root_lin_vel_b[:,:2])
                print(state_trajectory[:, 0, :2, a_i])
                print()
                s = ((s - s_mean) / s_std).unsqueeze(1).float()
                a = extract_observation_terms(diff_obs, diff_cfg.dataset.action_data_keys)
                a = ((a - a_mean )/ a_std).unsqueeze(1).float()

                historical_c = torch.cat((historical_c[:, 1:, :], c), dim=1)
                historical_s = torch.cat((historical_s[:, 1:, :], s), dim=1)                
                historical_a = torch.cat((historical_a[:, 1:, :], a), dim=1)

                # env stepping
                timestep += 1
                if timestep == 4:
                    break
                if args_cli.video:
                    # Exit the play loop after recording one video
                    if timestep == args_cli.video_length:
                        break

    # close the simulator
    env.close()



def evaluate_velocity_control(env, dataset, diff_cfg, policy, save_dir):
    import joblib

    num_runs_total = 10000
    sample_refequency = 50
    execution_length = 8

    data_out = {
        'has_fallen': [],
        'episode_reward': [],
        'total_episodes': 0,
        'sample_refequency':sample_refequency,
        'execution_length':execution_length,
        'diff_cfg':diff_cfg
    }

    while data_out['total_episode'] < num_runs_total:
        env.reset()

        obs, extras = env.get_observations()
        diff_obs = extras['observations']['data_collection_obs']

        c_mean, c_std = torch.from_numpy(dataset.c_mean).to(diff_obs.device), torch.from_numpy(dataset.c_std).to(diff_obs.device)
        a_mean, a_std = torch.from_numpy(dataset.a_mean).to(diff_obs.device), torch.from_numpy(dataset.a_std).to(diff_obs.device)

        init_c = extract_observation_terms(diff_obs, diff_cfg.dataset.context_data_keys)
        init_c = init_c - c_mean / c_std
        init_a = extract_observation_terms(diff_obs, diff_cfg.dataset.action_data_keys)
        init_a = init_a - a_mean / a_std

        historical_c = torch.zeros((env.num_envs, diff_cfg.model.context_len, c_mean.shape[0]), device=diff_obs.device)
        historical_a = torch.zeros((env.num_envs, diff_cfg.model.context_len, a_mean.shape[0]), device=diff_obs.device)

        historical_c[:, -1, :] = init_c.float()
        historical_a[:, -1, :] = init_a.float()

        action_dim = a_mean.shape[0]

        episode_timedout = False

        velocity_cmd = torch.randn((env.num_envs, 3), device=diff_obs.device) * 0.5

        epsiode_stats = {
            'has_fallen': torch.zeros(diff_obs.num_envs, device=diff_obs.device, dtype=torch.float),
            'episode_reward': torch.zeros(diff_obs.num_envs, device=diff_obs.device, dtype=torch.float),
        }

        while not episode_timedout:

            # run everything in inference mode --> enable gradient calculation overrides
            with torch.no_grad():
                model_kwargs = {'y':{}}
                model_kwargs['y']['mask'] = torch.ones((args_cli.num_envs, 1, 1, diff_cfg.model.pred_len), device=diff_obs.device, dtype=torch.bool)
                model_kwargs['y']['prefix_c'] = historical_c.permute(0, 2, 1).unsqueeze(1)
                model_kwargs['y']['prefix_a'] = historical_a.permute(0, 2, 1).unsqueeze(1)

                if 'target_vel' in diff_cfg.model.cond_mode:
                    model_kwargs['y']['target_vel'] = velocity_cmd
                elif diff_cfg.model.guaidance_fn == 'VelocityGuidance':
                    model_kwargs['y']['target_vel_guidance'] = velocity_cmd

                model_kwargs['y']['action_dim'] = action_dim
                # agent stepping
                model_output = policy(model_kwargs)

                action_trajectory = model_output[:, :, :action_dim, :] # [bs, njoints, nfeats, nframes] 
                action_trajectory = dataset.inv_transform_a(action_trajectory)
                
                for a_i in range(execution_length):
                    actions = action_trajectory[:, 0, :, a_i] #bs,num_actions
            
                    obs, _, _, extras = env.step(actions)

                    diff_obs = extras['observations']['data_collection_obs']
                    c = extract_observation_terms(diff_obs, diff_cfg.dataset.context_data_keys)
                    c = (c - c_mean / c_std).unsqueeze(1).float()
                    a = extract_observation_terms(diff_obs, diff_cfg.dataset.action_data_keys)
                    a = (a - a_mean / a_std).unsqueeze(1).float()

                    historical_c = torch.cat((historical_c[:, 1:, :], c), dim=1)                
                    historical_a = torch.cat((historical_a[:, 1:, :], a), dim=1)

                    #TODO
                    has_fallen = None
                    s = extract_observation_terms(diff_obs, diff_cfg.dataset.state_data_keys)
                    velocity = s[:, dataset.velocity_indexes]

                    vel_error = torch.sum( torch.square(velocity_cmd-velocity),dim=1,)
                    reward = torch.exp(-vel_error)

                    epsiode_stats['has_fallen'] |= has_fallen
                    epsiode_stats['episode_reward'][epsiode_stats['has_fallen']] += reward[epsiode_stats['has_fallen']]


        data_out['has_fallen'].append(epsiode_stats['has_fallen'])
        data_out['epsiode_reward'].append(epsiode_stats['episode_reward'])



    data_out['has_fallen'] = torch.cat(data_out['has_fallen'])[:num_runs_total]
    data_out['episode_reward']  = torch.cat(data_out['episode_reward'])[:num_runs_total]
    data_out['total_episodes'] = num_runs_total

    joblib.dump(data_out, f'{save_dir}/evaluate_velocity_control.pkl')


    return 


def evaluate_perturb_walking():

    return 


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
