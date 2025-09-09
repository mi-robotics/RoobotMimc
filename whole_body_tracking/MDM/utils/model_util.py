import torch
from model.mdm import MDM
from diffusion import gaussian_diffusion as gd
from diffusion.respace import SpacedDiffusion, space_timesteps

def load_model_wo_clip(model, state_dict):
    # assert (state_dict['sequence_pos_encoder.pe'][:model.sequence_pos_encoder.pe.shape[0]] == model.sequence_pos_encoder.pe).all()  # TEST
    # assert (state_dict['embed_timestep.sequence_pos_encoder.pe'][:model.embed_timestep.sequence_pos_encoder.pe.shape[0]] == model.embed_timestep.sequence_pos_encoder.pe).all()  # TEST
    del state_dict['sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    del state_dict['embed_timestep.sequence_pos_encoder.pe']  # no need to load it (fixed), and causes size mismatch for older models
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') or 'sequence_pos_encoder' in k for k in missing_keys])


def create_model_and_diffusion(cfg, dataset):
    model = MDM(cfg)
    diffusion = create_gaussian_diffusion(cfg, dataset)
    return model, diffusion

def create_model(cfg):
    model = MDM(cfg)
    return model

def create_gaussian_diffusion(cfg, dataset):
    # default params
    steps = cfg.diffusion.diffusion_steps

    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.

    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(
        cfg.diffusion.noise_schedule, 
        cfg.diffusion.diffusion_steps, 
        scale_betas=1.)
    
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not cfg.diffusion.predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not cfg.diffusion.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not cfg.diffusion.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        dataset=dataset,
        cfg=cfg
    )

def load_saved_model(model, model_path, use_avg: bool=False):  # use_avg_model
    state_dict = torch.load(model_path, map_location='cpu')
    # Use average model when possible
    if use_avg and 'model_avg' in state_dict.keys():
    # if use_avg_model:
        print('loading avg model')
        state_dict = state_dict['model_avg']
    else:
        if 'model' in state_dict:
            print('loading model without avg')
            state_dict = state_dict['model']
        else:
            print('checkpoint has no avg model, loading as usual.')
    load_model_wo_clip(model, state_dict)
    return model