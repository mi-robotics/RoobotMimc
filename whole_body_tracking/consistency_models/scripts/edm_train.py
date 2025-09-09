"""
Train a diffusion model on images.
"""

import argparse

from cm import dist_util, logger

from cm.resample import create_named_schedule_sampler
from cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    create_diffusion
)
from cm.train_util import TrainLoop
import torch.distributed as dist


# MDM ========================================
from utils.model_util import create_model
from data_loaders.get_data import get_dataset_loader
from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation



import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from easydict import EasyDict

@hydra.main(config_path="../config", config_name="base_edm", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(config=cfg, f=f"{cfg['training']['save_dir']}/.hydra/resolved_config.yaml")
    cfg = EasyDict(cfg)

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    data = get_dataset_loader(cfg, device=dist_util.dev())
    diffusion = create_diffusion(cfg, data.dataset )
    model = create_model(cfg)
   
    
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(cfg.diffusion.schedule_sampler, diffusion)

    batch_size = cfg.training.batch_size


    logger.log("creating data loader...")

    train_platform_type = eval(cfg.train_platform_type)
    train_platform = train_platform_type(cfg.training.save_dir)
    train_platform.report_args(cfg, name='Args')

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=-1,
        lr=cfg.training.lr,
        ema_rate=cfg.training.ema_rate,
        log_interval=cfg.training.log_interval,
        save_interval=cfg.training.save_interval,
        resume_checkpoint=cfg.training.resume_checkpoint,
        use_fp16=cfg.training.use_fp16,
        fp16_scale_growth=cfg.training.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.training.weight_decay,
        lr_anneal_steps=cfg.training.lr_anneal_steps,
        num_steps=cfg.training.num_steps,
        train_platform=train_platform,
        cfg=cfg
    )

    trainer.run_loop()



if __name__ == "__main__":
    main()
