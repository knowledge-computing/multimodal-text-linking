from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from transformers import LayoutLMv3Config
from transformers import BertModel, BertConfig
from transformers import BeitForMaskedImageModeling

from models.losses import NCELoss, FocalLoss
from models.model_utils import MLP, TokenEncoder
from models.light import LightModel


class LightTextLinking(nn.Module):
    def __init__(self, args):
        super(LightTextLinking, self).__init__()
        self.emb_dim = 768
        self.aux_losses = args.aux_losses
        self.embedding_components = args.embedding_components
        self.max_position_embeddings = getattr(args, "max_position_embeddings", 1280)

        self.config = LayoutLMv3Config(max_position_embeddings=self.max_position_embeddings)
        self.light = LightModel(self.config)
        self.token_encoder = TokenEncoder(self.emb_dim)
        
        self.predecessor_mlp = MLP(self.emb_dim, self.emb_dim, 
                                   [self.emb_dim, self.emb_dim], 
                                   nonlinearity='relu')
        self.successor_mlp = MLP(self.emb_dim, self.emb_dim, 
                                 [self.emb_dim, self.emb_dim], 
                                 nonlinearity='relu')
            
        self.loss_fn = nn.CrossEntropyLoss() 
        self.focal_loss_fn = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)

    def forward(self, input_data, return_loss=True):
        B = input_data['labels'].shape[0]
        attn_mask = input_data['attention_mask']

        outputs, poly_embedding = self.light(
            input_ids=input_data['input_ids'],
            bbox=input_data['bbox'],
            attention_mask=attn_mask,
            pixel_values=input_data['pixel_values'],
            polygons=input_data['polygons']
        )

        token_embeddings = outputs.last_hidden_state[:, :1000]
            
        all_losses = defaultdict(int)
        all_logits = {'logits': [], 'bi_logits': []}
        for batch_idx in range(B):
            first_token_indices = input_data['first_token_indices'][batch_idx]
            first_token_indices = first_token_indices[first_token_indices != -999]

            embeddings = token_embeddings[batch_idx][first_token_indices]
            embeddings = self.token_encoder(embeddings)
            
            pred_embeddings = self.predecessor_mlp(embeddings)
            succ_embeddings = self.successor_mlp(embeddings)
        
            dot_products = torch.matmul(pred_embeddings, succ_embeddings.T)
            bi_dot_products = torch.matmul(succ_embeddings, pred_embeddings.T)
            all_logits['logits'].append(dot_products)
            all_logits['bi_logits'].append(bi_dot_products)

            if return_loss:
                lables_B = [input_data['labels'][batch_idx][i] for i in first_token_indices]
                losses = self.compute_sample_losses(lables_B, dot_products, bi_dot_products)
                all_losses['base_loss'] += losses['base_loss']
                if 'bidirection' in self.aux_losses:
                    all_losses['bidirection'] += losses['bidirection']
                if 'focal' in self.aux_losses:
                    all_losses['focal_loss'] += losses['focal_loss']
            
        return all_logits, all_losses


    def compute_sample_losses(self, labels, dot_products, bi_dot_products):
        losses = {}
        label_to_index = {label.item(): idx for idx, label in enumerate(labels)}
        target_next_indices, targer_prev_indices = [], []

        target_matrix_succ = torch.zeros_like(dot_products, dtype=torch.float32)
        target_matrix_prev = torch.zeros_like(dot_products, dtype=torch.float32)

        for i, label in enumerate(labels):
            label = label.item()
            a, b = str(label).split('000', 1)
            a, b = int(a), int(b)
            next_label = int(f"{a}000{b + 1}")
            prev_label = int(f"{a}000{b - 1}")
    
            if label_to_index.get(next_label) is not None:
                target_next_indices.append(label_to_index[next_label])
                target_matrix_succ[i, label_to_index[next_label]] = 1.0 
            else:
                target_next_indices.append(label_to_index[label])

            if label_to_index.get(prev_label) is not None:
                targer_prev_indices.append(label_to_index[prev_label])
                target_matrix_prev[i, label_to_index[prev_label]] = 1.0 
            else:
                targer_prev_indices.append(label_to_index[label])
            
        target_next_indices = torch.tensor(target_next_indices, dtype=torch.long).to(dot_products.device)
        loss = self.loss_fn(dot_products, target_next_indices)        
        losses['base_loss'] = loss

        if 'bidirection' in self.aux_losses:
            targer_prev_indices = torch.tensor(targer_prev_indices, dtype=torch.long).to(dot_products.device)
            loss = self.loss_fn(bi_dot_products, targer_prev_indices)
            losses['bidirection'] = loss

        if 'focal' in self.aux_losses:
            losses['focal_loss'] = 0
            focal_loss = self.focal_loss_fn(dot_products, target_matrix_succ)
            losses['focal_loss'] += focal_loss.mean() * 100

            if 'bidirection' in self.aux_losses:
                focal_loss = self.focal_loss_fn(bi_dot_products, target_matrix_prev)
                losses['focal_loss'] += focal_loss.mean() * 100

        return losses

