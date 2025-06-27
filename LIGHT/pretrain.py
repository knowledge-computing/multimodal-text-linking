import os
import sys
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler

from dataset.dataset_pretrain import PretrainDataset
from model.layoutlmv3 import LayoutLMv3Pretrain
from model.layoutlmv4 import LayoutLMv4Pretrain

from model.model_utils import get_processors
from utils.options import parse_args
from utils.utils import *
import utils.misc as misc

    
def validate(model, data_loader, device, log_writer, epoch=None):
    model.eval()
    total_losses = defaultdict(int)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_data = {k: v.to(device) for k, v in batch.items()}
            losses = model(input_data)
            for k, loss in losses.items():
                total_losses[k] += loss.item() / len(data_loader)

    for k, loss in total_losses.items():
        print(f"Epoch {epoch + 1}: Validation {k}: {loss}")
        log_writer.add_scalar(f"Loss/Val_{k}", loss, epoch)
    log_writer.flush()
    return 0


def main():
    args = parse_args()
    if args.distributed:
        misc.init_distributed_mode(args)
    
    misc.init_training(args)
    misc.seed_everything(1234)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'GPU Device: {device}')
    if not args.distributed or misc.is_main_process():
        log_writer = SummaryWriter(log_dir=args.log_dir)

    ###
    ### dataloader
    tokenizer, image_processor = get_processors(args.pretrained_model_name)
    train_dataset = PretrainDataset(tokenizer, image_processor, args)
    val_dataset = PretrainDataset(tokenizer, image_processor, args, mode='val')
    
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        sampler_val = torch.utils.data.SequentialSampler(val_dataset)
        
    train_dataloader = DataLoader(train_dataset, 
                                  sampler=sampler_train, 
                                  batch_size=args.batch_size)
    val_dataloader = DataLoader(val_dataset, 
                                sampler=sampler_val, 
                                batch_size=args.batch_size)   

    if args.base_model == 'layoutLMv3':
        model = LayoutLMv3Pretrain(args).to(device)
    elif args.base_model == 'layoutLMv4':
        model = LayoutLMv4Pretrain(args).to(device)
    else:
        raise NotImplementedError
    model_without_ddp = model
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[args.gpu],
            find_unused_parameters=True)
        model_without_ddp = model.module
    
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.distributed:
        total_steps = int(args.num_samples_per_epoch * args.num_epochs / (args.batch_size * 2))
    else:
        total_steps = int(args.num_samples_per_epoch * args.num_epochs / args.batch_size)
        
    total_steps = int(args.num_samples_per_epoch * args.num_epochs)
    print("Total number of Steps:", total_steps)  # Should print: 960
    warmup_ratio = args.warmup_ratio  # 4.8% warm-up
    warmup_steps = int(total_steps * warmup_ratio)  # Compute warm-up steps
    print("Warm-up Steps:", warmup_steps)  # Should print: 960
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    
    ###
    ### model
    if args.base_model == 'layoutLMv4' and args.poly_pretrained_weights is not None:
        assert os.path.exists(args.poly_pretrained_weights), "Poly pretrained weights must exists" 
        print("... Loading pretrained weights for poly_encoder ...")
        checkpoint = load_model_weights(args.poly_pretrained_weights)
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
        msg = model_without_ddp.layoutLMv4.poly_encoder.load_state_dict(checkpoint, strict=False)
        print(msg)
    
    ###
    ### resume
    resume_checkpoint(model_without_ddp, optimizer, scheduler, device, args)
        
    ###
    ### training
    for epoch in range(args.start_epoch, args.num_epochs):
        model.train()
        train_dataset.reset()
    
        epoch_losses = defaultdict(int)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):

            input_data = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            losses = model(input_data)

            total_loss = 0
            for k, loss in losses.items():
                epoch_losses[k] += loss.item() / len(train_dataloader)
                total_loss += loss
                 
            total_loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            
            if misc.is_main_process():
                scheduler.step()        
            
        if misc.is_main_process():
            ### print
            for k, loss in epoch_losses.items():
                log_writer.add_scalar(f"Loss/Train_{k}", loss, epoch)
                print(f"Epoch {epoch + 1}: Training {k}: {loss}")    

            ### save model
            if args.save_every_epoch > 0 and (epoch + 1) % args.save_every_epoch == 0:
                model_save_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, model_save_path)
                print(f"Model saved at epoch {epoch + 1}: {model_save_path}")

            if epoch + 1 == args.num_epochs:
                model_save_path = os.path.join(args.checkpoint_dir, f"final_model.pth")
                torch.save(model.state_dict(), model_save_path)

            ### validate
            if (epoch + 1) % args.eval_every_epoch == 0:
                validate(model, val_dataloader, device, log_writer, epoch=epoch)         
                
    
if __name__ == '__main__':
    main()

    
# CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=14476 pretrain.py --config configs/pretrain_layoutlmv4.yaml