from torch.utils.data import DataLoader
from .dataset import RobotStateActionDataset
import torch


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch, cfg):

    notnone_batches = [b for b in batch if b is not None]
    
    action_batch = torch.stack([b['inp_a'] for b in notnone_batches], dim=0)    
    context_batch = torch.stack([b['inp_c'] for b in notnone_batches], dim=0)
    state_batch = torch.stack([b['inp_s'] for b in notnone_batches], dim=0)

    cond = {'y': {
        'mask':torch.ones((action_batch.shape[0], 1, 1, action_batch.shape[-1]), device=action_batch.device, dtype=torch.bool)
    }}

    cond['y']['mask_actions'] = cond['y']['mask'].clone()   
    if cfg.model.action_pred_len > 0:
        cond['y']['mask_actions'][:, :, :, cfg.model.action_pred_len:] = 0

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'prefix_a' in notnone_batches[0]:
        prefix_batch = [b['prefix_a'] for b in notnone_batches]
        cond['y'].update({'prefix_a': collate_tensors(prefix_batch)})

    if 'prefix_c' in notnone_batches[0]:
        prefix_batch = [b['prefix_c'] for b in notnone_batches]
        cond['y'].update({'prefix_c': collate_tensors(prefix_batch)})

    if 'prefix_s' in notnone_batches[0]:
        prefix_batch = [b['prefix_s'] for b in notnone_batches]
        cond['y'].update({'prefix_s': collate_tensors(prefix_batch)})

    if 'orig_length' in notnone_batches[0]:
        cond['y'].update({'orig_length': torch.as_tensor([b['orig_length'] for b in notnone_batches])})

    if 'target_vel' in notnone_batches[0]:
        prefix_batch = [b['target_vel'] for b in notnone_batches]
        cond['y'].update({'target_vel': collate_tensors(prefix_batch)})


    return action_batch, context_batch, state_batch, cond

def get_dataset_loader(cfg, split='train', device='cpu'):
    dataset = RobotStateActionDataset(cfg)
    
    def collate_wrapper(batch):
        # 1. Adapt the raw data from the dataset
        adapted_data = [{
            'inp_a': torch.tensor(b[0].T).float().unsqueeze(0)[..., -cfg.model.pred_len:],
            'inp_c': torch.tensor(b[1].T).float().unsqueeze(0)[..., -cfg.model.pred_len:],
            'inp_s': torch.tensor(b[2].T).float().unsqueeze(0)[..., -cfg.model.pred_len:],
            'prefix_a': torch.tensor(b[0].T).float().unsqueeze(0)[..., :-cfg.model.pred_len],
            'prefix_c': torch.tensor(b[1].T).float().unsqueeze(0)[..., :-cfg.model.pred_len],
            'prefix_s': torch.tensor(b[2].T).float().unsqueeze(0)[..., :-cfg.model.pred_len],
            'target_vel': torch.tensor(b[3]).float(),
            'text': b[5],
            'orig_length': b[4],
        } for b in batch]
        
        # 2. Call your collate function with the adapted data and the config
        return collate(adapted_data, cfg)


    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate_wrapper
    )

    return loader