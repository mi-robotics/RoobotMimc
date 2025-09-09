# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json


from utils.fixseed import fixseed
from utils.parser_util import apply_rules
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion

from train.train_platforms import WandBPlatform, ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from easydict import EasyDict

@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    OmegaConf.save(config=cfg, f=f"{cfg['training']['save_dir']}/.hydra/resolved_config.yaml")
    cfg = EasyDict(cfg)
    apply_rules(cfg)
    fixseed(cfg.seed)

    train_platform_type = eval(cfg.train_platform_type)
    train_platform = train_platform_type(cfg.training.save_dir)
    train_platform.report_args(cfg, name='Args')

    if cfg.training.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')

    dist_util.setup_dist(cfg.device)

    print("creating data loader...")

    data = get_dataset_loader(cfg, device=dist_util.dev())

    print("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(cfg, data.dataset)
    model.to(dist_util.dev())

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training...")

    TrainLoop(cfg, train_platform, model, diffusion, data).run_loop()
    train_platform.close()

if __name__ == "__main__":
    main()
