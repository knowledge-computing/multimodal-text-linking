import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import logging
from transformers import LayoutLMv3Model, LayoutLMv3Config
from transformers import BertModel, BertConfig
from transformers import BeitForMaskedImageModeling
from model.model_utils import MLP
from models.light import LightModel


class LightPretrain(nn.Module):
    def __init__(self, args):
        super(LightPretrain, self).__init__()
        self.emb_dim = self.config.hidden_size
        self.token_padding_max_length = args.token_padding_max_length
        self.max_position_embeddings = getattr(args, "max_position_embeddings", 1280)

        self.image_tokenizer = BeitForMaskedImageModeling.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.config = LayoutLMv3Config(max_position_embeddings=self.max_position_embeddings)
        self.light = LightModel.from_pretrained(args.pretrained_model_name,
                                                config=self.config,
                                                ignore_mismatched_sizes=True)

        self.token_mlp = MLP(self.emb_dim, 50265, [self.emb_dim], nonlinearity='relu')
        self.visual_mlp = MLP(self.emb_dim, 8192, [self.emb_dim], nonlinearity='relu')
        self.ce_loss_func = nn.CrossEntropyLoss()
        
        self.add_wpa_loss = args.add_wpa_loss
        if self.add_wpa_loss:
            self.wpa_mlp = MLP(self.emb_dim * 2, 1, [self.emb_dim, self.emb_dim], nonlinearity='relu')        
            self.bce_loss_func = nn.BCELoss()
        
    def compute_loss(self, logits, labels, mask):
        active_loss = mask.view(-1) == 1        
        active_logits = logits.view(-1, logits.size(-1))[active_loss, :]
        active_labels = labels.view(-1)[active_loss]
        return self.ce_loss_func(active_logits, active_labels)

    def forward(self, input_data):
        with torch.no_grad():
            mim_labels = self.image_tokenizer.forward(pixel_values=input_data['pixel_values']).logits.argmax(dim=-1)

        attn_mask = input_data['attention_mask']
        mlm_mask = input_data['mlm_mask']
        mim_mask = input_data['mim_mask']
        outputs, _ = self.light(
                input_ids=input_data['input_ids'],
                bbox=input_data['bbox'],
                attention_mask=attn_mask,
                pixel_values=input_data['masked_pixel_values'],
                polygons=input_data['polygons']
        )

        last_hidden_state = outputs.last_hidden_state
        # token_embeddings: bx512x768
        # visual_embeddings: bx196x768
        token_embeddings = last_hidden_state[:, :self.token_padding_max_length]       
        visual_embeddings = last_hidden_state[:, self.token_padding_max_length: self.token_padding_max_length+196] 
        token_logits = self.token_mlp(token_embeddings)
        visual_logits = self.visual_mlp(visual_embeddings)
        mlm_loss = self.compute_loss(token_logits, input_data['mlm_labels'], mlm_mask)
        mim_loss = self.compute_loss(visual_logits, mim_labels, mim_mask)
        
        #####################################################################
        ### WPA Loss
        if not self.add_wpa_loss:
            return {
                'mlm_loss': mlm_loss,
                'mim_loss': mim_loss
            }
        else:
            wpa_loss = 0
            image_id_labels = input_data['image_id_labels'] # 8 x 512
            wpa_labels = input_data['wpa_labels']
            for b in range(image_id_labels.shape[0]):
                mask = wpa_labels[b] > -100
                image_id_labels_B = image_id_labels[b][mask]
                matched_visual_embeddings = visual_embeddings[b][image_id_labels_B]
                matched_token_embeddings = token_embeddings[b][mask]
                if matched_token_embeddings.shape[0] == 0:
                    continue

                wpa_logits = self.wpa_mlp(torch.cat([matched_visual_embeddings, matched_token_embeddings], dim=-1))       
                wpa_loss += self.bce_loss_func(nn.Sigmoid()(wpa_logits), wpa_labels[b][mask][:, None])

            wpa_loss /= image_id_labels.shape[0]
            return {
                'mlm_loss': mlm_loss,
                'mim_loss': mim_loss,
                'wpa_loss': wpa_loss
            }
            