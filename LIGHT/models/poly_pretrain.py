import cv2
import random
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from transformers import LayoutLMv3Model, LayoutLMv3Tokenizer, LayoutLMv3Config
from transformers import BertConfig, BertModel

from models.model_utils import MLP, PCAMaskEncoding, AdditiveAttention, MultiHeadSelfAttention


class PolyPretrain(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        task="pretrain",
        args=None,
    ):
        super(PolyPretrain, self).__init__()
        self.task = task
        self.padding_token_id = args.padding_token_id if args is not None else -100
        self.mask_token_id = args.mask_token_id if args is not None else -1
        self.emb_dim = emb_dim
        self.num_layers = 6
        self.num_heads = 8
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.coord_embedding = nn.Linear(1, emb_dim, bias=False)
        self.position_embedding = nn.Embedding(16 * 2 + 1, emb_dim)
        self.bert_config = BertConfig(
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            intermediate_size=emb_dim * 4,
            max_position_embeddings=16 * 2 + 1)

        self.encoder = BertModel(config=self.bert_config)        
        
        if self.task == "pretrain":
            dim = emb_dim // 2
            self.reconstruct_mlp = MLP(emb_dim, 1, [dim], nonlinearity='relu')
            self.center_x_mlp = MLP(emb_dim, 1, [dim], nonlinearity='relu')
            self.center_y_mlp = MLP(emb_dim, 1, [dim], nonlinearity='relu')
            self.height_mlp = MLP(emb_dim, 1, [dim], nonlinearity='relu')
            self.angle_mlp = MLP(emb_dim, 1, [dim], nonlinearity='relu')
            self.dist_mlp = MLP(emb_dim, emb_dim, [emb_dim], nonlinearity='relu')
            self.loss_fn = nn.MSELoss()
            self.ce_loss_fn = nn.CrossEntropyLoss()  
        else:
            self.mlp = MLP(emb_dim, emb_dim, [emb_dim], nonlinearity='relu')
            self.layer_norm = nn.LayerNorm(emb_dim)

    def add_cls_token(self, input_embeddings, attn_mask=None):
        B, N, C = input_embeddings.shape
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        embeddings = torch.cat([cls_tokens, input_embeddings], dim=1)  
        if attn_mask is not None:
            attn_mask = torch.cat([torch.ones(B, 1, device=input_embeddings.device), attn_mask], dim=1)            
        return embeddings, attn_mask
        
    def add_position_embeddings(self, input_embeddings):
        B, N, C = input_embeddings.shape
        position_ids = torch.arange(N, device=input_embeddings.device).unsqueeze(0).repeat(B, 1)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = input_embeddings + position_embeddings  
        return embeddings
            
    def forward(self, input_data):
        sequences = input_data["sequences"]
        if len(sequences.shape) > 2:
            ori_B, ori_N, _ = sequences.shape
            sequences = sequences.flatten(0, 1)
        
        # if the whole polygon is padding, remove
        valid_mask = (sequences != self.padding_token_id).any(dim=1)
        sequences = sequences[valid_mask]
        
        # find paddings in each polygon
        seq_attn_mask = sequences != self.padding_token_id
        input_mask = sequences == self.mask_token_id
        B, N = sequences.shape

        # (B, seq_len, hidden_size)
        seq_embeddings = self.coord_embedding(sequences.unsqueeze(-1))  
        embeddings, attn_mask = self.add_cls_token(seq_embeddings, seq_attn_mask)
        embeddings = self.add_position_embeddings(embeddings)        
        outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attn_mask)
        polygon_embedding = outputs.last_hidden_state
        
        all_losses = {}
        if self.task == "pretrain":    
            reconstuct = self.reconstruct_mlp(polygon_embedding[:, 1:, :]).squeeze()
            labels = input_data["gt_sequences"].flatten(0, 1)
            labels = labels[valid_mask] 
            all_losses["base_loss"] = self.loss_fn(reconstuct[input_mask], labels[input_mask])

            cls_embedding = polygon_embedding[:, 0, :]
            
            angles = self.angle_mlp(cls_embedding).squeeze()
            labels = input_data["angles"].flatten(0, 1)
            all_losses["angle_loss"] = self.loss_fn(angles, labels[labels != self.padding_token_id])
            
            center_x = self.center_x_mlp(cls_embedding).squeeze()
            labels = input_data["center_x"].flatten(0, 1)
            all_losses["cx_loss"] = self.loss_fn(center_x, labels[labels != self.padding_token_id])

            center_y = self.center_y_mlp(cls_embedding).squeeze()
            labels = input_data["center_y"].flatten(0, 1)
            all_losses["cy_loss"] = self.loss_fn(center_y, labels[labels != self.padding_token_id])
            
            heights = self.height_mlp(cls_embedding).squeeze()
            labels = input_data["char_heights"].flatten(0, 1)
            all_losses["height_loss"] = self.loss_fn(heights, labels[labels != self.padding_token_id])

            ## create positive samples
            pos_sequences = input_data["gt_sequences"].flatten(0, 1)
            pos_sequences = pos_sequences[valid_mask]
            pos_embeddings = self.coord_embedding(pos_sequences.unsqueeze(-1))  
            embeddings, _ = self.add_cls_token(pos_embeddings, attn_mask=None)
            embeddings = self.add_position_embeddings(embeddings)
            outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attn_mask)
            positive = outputs.last_hidden_state[:, 0, :]        

            dists = self.dist_mlp(cls_embedding) # N x C
            dists = F.normalize(dists, dim=-1)
            prev, dis_loss, nce_loss = 0, 0, 0
            
            non_nan_B = 0
            temperature = 0.1
            for b in range(ori_B):
                ### distance 
                m = input_data['m'][b]
                emb = dists[prev: prev + m, :]
                sim = emb @ emb.T / temperature
                mask = torch.eye(sim.size(0), device=sim.device).bool()
                sim.masked_fill_(mask, float('-inf'))

                labels = input_data["dist_matrix"][b, :m]
                lo = self.ce_loss_fn(sim, labels) 
                if not torch.isnan(lo):
                    non_nan_B += 1
                    dis_loss += lo
                
                # contrast
                nce_loss += self.compute_CL_loss(cls_embedding[prev: prev + m, :], positive[prev: prev + m, :])
                prev += m
                     
            all_losses["dis_loss"] = dis_loss / non_nan_B 
            all_losses["nce_loss"] = nce_loss / ori_B 
            return all_losses
            
        else:
            embeddings = polygon_embedding[:, 0, :]
            embeddings = self.mlp(embeddings)
            embeddings = self.layer_norm(embeddings)
            return embeddings
            
    def compute_CL_loss(self, anchor, positive):
        temperature = 0.1
        positive = F.normalize(positive, p=2, dim=-1) 
        anchor = F.normalize(anchor, p=2, dim=-1) 
        
        negatives = []
        for i in range(10):
            permuted_indices = torch.randperm(anchor.size(0))
            negatives.append(anchor[permuted_indices])
        negative = torch.stack(negatives, dim=1)
        
        pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # Shape: (112,)
        anchor_expanded = anchor.unsqueeze(1)  # Shape: (112, 1, 768)
        neg_sim = torch.sum(anchor_expanded * negative, dim=-1) / temperature  # Shape: (112, 5)

        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # Shape: (112, 6)
        labels = torch.zeros(all_sim.shape, dtype=torch.float).to(anchor.device)
        labels[:, 0] = 1

        log_softmax_sim = F.log_softmax(all_sim, dim=1)
        return -torch.sum(labels * log_softmax_sim) / anchor.size(0)
