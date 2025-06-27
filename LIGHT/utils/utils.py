import os
import time
import torch
import numpy as np


def seed_everything(seed: int):
    # random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True


def load_model_weights(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if checkpoint.get("model_state_dict") is not None:
        checkpoint = checkpoint['model_state_dict']
    return checkpoint
    
    
def resume_checkpoint(model, optimizer, scheduler, device, args):
    if args.resume and os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        model_state_dict = checkpoint['model_state_dict']
        if list(model_state_dict.keys())[0].startswith('module.'):
            model_state_dict = {k[len("module."):]: v for k, v in model_state_dict.items()}
        model.load_state_dict(model_state_dict, strict=True)
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        # Fix optimizer device issue
        for state in optimizer.state.values():
            if isinstance(state, dict):
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        args.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming checkpoint from {args.checkpoint_path}, "
              f"starting from epoch {args.start_epoch}")
    else:
        args.start_epoch = 0