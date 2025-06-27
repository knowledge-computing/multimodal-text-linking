import os
import sys
import random
import numpy as np
import yaml
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import LinkingTrainDataset, LinkingTestDataset
from dataset.buildin import DATASET_META
from models.text_linking import LightTextLinking
from models.model_utils import get_processors
from utils.options import parse_args
from utils.utils import *


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def validate(model, data_loader, device, log_writer, epoch=None):
    model.eval()
    total_losses = defaultdict(int)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_data = {k: v.to(device) for k, v in batch.items()}
            _, losses = model(input_data, return_loss=True)
            for k, loss in losses.items():
                total_losses[k] += loss.item() / len(data_loader)
                
    val_loss = total_losses['base_loss']
    for k, loss in total_losses.items():
        print(f"Epoch {epoch + 1}: Validation {k}: {loss}")
        log_writer.add_scalar(f"Loss/Val_{k}", loss, epoch)
    return val_loss


def main():
    seed_everything(1234)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'GPU Device: {device}')
    log_writer = SummaryWriter(log_dir=args.log_dir)
    with open(args.save_config_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Configuration saved to {args.save_config_file}")

    ###
    ### dataloader
    tokenizer, image_processor = get_processors(args.pretrained_model_name)
    train_dataset = LinkingTrainDataset(tokenizer, image_processor, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = LinkingTestDataset(tokenizer, image_processor, args, mode='val', return_ori=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)    

    ###
    ### model
    model = LightTextLinking(args)        
    if args.light_pretrained_weights is not None:
        assert os.path.exists(args.light_pretrained_weights), "LIGHT pretrained weights must exists"
        print("... Loading pretrained weights for LIGHT ...")
        checkpoint = load_model_weights(args.light_pretrained_weights)
        checkpoint = {k[len("module.light."):]: v for k, v in checkpoint.items() if k.startswith('module.light')}    
        msg = model.light.load_state_dict(checkpoint, strict=False)
        print(msg) 

    total_params = count_parameters(model, trainable_only=False)
    print(f"Total parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=args.scheduler_patience, 
        factor=0.1, 
        threshold=0,
        min_lr=0.000001, 
        verbose=True)
    
    model.to(device)

    ###
    ### training
    best_val_loss, epochs_no_improve = np.inf, 0
    
    for epoch in range(args.num_epochs):
        model.train()
        train_dataset.reset()
        
        epoch_losses = defaultdict(int)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            input_data = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            _, losses = model(input_data, return_loss=True)

            total_loss = 0
            for k, loss in losses.items():
                epoch_losses[k] += loss.item() / len(train_dataloader)
                total_loss += loss
                 
            total_loss.backward()
            optimizer.step()

        for k, loss in epoch_losses.items():
            print(f"Epoch {epoch + 1}: Training {k}: {loss}")
            log_writer.add_scalar(f"Loss/Train_{k}", loss, epoch)

        ### save model
        if args.save_every_epoch > 0 and (epoch + 1) % args.save_every_epoch == 0:
            model_save_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1}: {model_save_path}")

        ### validate
        if (epoch + 1) % args.eval_every_epoch == 0:
            val_loss = validate(model, val_dataloader, device, log_writer=log_writer, epoch=epoch)
            if val_loss < best_val_loss:
                print(f"Loss decreases: {best_val_loss-val_loss}, Best model saved.")            
                model_save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_save_path)
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in val loss for {epochs_no_improve} epochs.")
                    
            scheduler.step(val_loss)

        ### terminate
        if epochs_no_improve >= args.patience or epoch == args.num_epochs - 1:
            base_comm = f"python inference.py --test_dataset MapText_test --out_file predict.json --model_dir {model_dir} --anno_path {anno_path} --img_dir {img_dir}"
            if args.val_dataset == "MapText_val":
                comm = base_comm.format(model_dir=args.output_dir, 
                                        anno_path=DATASET_META['MapText_test']['anno_path'], 
                                        img_dir=DATASET_META['MapText_test']['img_dir'])
                print(comm)
                os.system(comm)
            if args.val_dataset == "HierText_val":
                comm = base_comm.format(model_dir=args.output_dir, 
                                        anno_path=DATASET_META['HierText_test']['anno_path'], 
                                        img_dir=DATASET_META['HierText_test']['img_dir'])
                print(comm)
                os.system(comm)
            sys.exit(0)
            
        
    
if __name__ == '__main__':
    main()
