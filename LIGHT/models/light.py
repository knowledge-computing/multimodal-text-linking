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

def torch_int(x):
    """
    Casts an input to a torch int64 tensor if we are in a tracing context, otherwise to a Python int.
    """
    # if not is_torch_available():
    #     return int(x)

    import torch

    return x.to(torch.int64) if torch.jit.is_tracing() and isinstance(x, torch.Tensor) else int(x)


class LightModel(LayoutLMv3Model):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config):
        super().__init__(config)

        #####################
        self.args = None
        self.poly_encoder = PolyEncoder(emb_dim=config.hidden_size)        
        self.poly_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.poly_dropout = nn.Dropout(config.hidden_dropout_prob)
        #####################
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        polygons: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
            device = inputs_embeds.device
        elif pixel_values is not None:
            batch_size = len(pixel_values)
            device = pixel_values.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds or pixel_values")

        if input_ids is not None or inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(((batch_size, seq_length)), device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            if bbox is None:
                bbox = torch.zeros(tuple(list(input_shape) + [4]), dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                bbox=bbox,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            #####################
            poly_embedding = self.poly_encoder({"sequences": polygons})
            embedding_output = self.poly_layer_norm(embedding_output + poly_embedding)
            embedding_output = self.poly_dropout(embedding_output)
            #####################

        final_bbox = final_position_ids = None
        patch_height = patch_width = None
        if pixel_values is not None:
            patch_height, patch_width = (
                torch_int(pixel_values.shape[2] / self.config.patch_size),
                torch_int(pixel_values.shape[3] / self.config.patch_size),
            )
            visual_embeddings = self.forward_image(pixel_values)
            visual_attention_mask = torch.ones(
                (batch_size, visual_embeddings.shape[1]), dtype=torch.long, device=device
            )
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, visual_attention_mask], dim=1)
            else:
                attention_mask = visual_attention_mask

            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                if self.config.has_spatial_attention_bias:
                    visual_bbox = self.calculate_visual_bbox(device, dtype=torch.long, batch_size=batch_size)
                    if bbox is not None:
                        final_bbox = torch.cat([bbox, visual_bbox], dim=1)
                    else:
                        final_bbox = visual_bbox

                visual_position_ids = torch.arange(
                    0, visual_embeddings.shape[1], dtype=torch.long, device=device
                ).repeat(batch_size, 1)
                if input_ids is not None or inputs_embeds is not None:
                    position_ids = torch.arange(0, input_shape[1], device=device).unsqueeze(0)
                    position_ids = position_ids.expand(input_shape)
                    final_position_ids = torch.cat([position_ids, visual_position_ids], dim=1)
                else:
                    final_position_ids = visual_position_ids

            if input_ids is not None or inputs_embeds is not None:
                embedding_output = torch.cat([embedding_output, visual_embeddings], dim=1)
            else:
                embedding_output = visual_embeddings

            embedding_output = self.LayerNorm(embedding_output)
            embedding_output = self.dropout(embedding_output)
        elif self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
            if self.config.has_spatial_attention_bias:
                final_bbox = bbox
            if self.config.has_relative_attention_bias:
                position_ids = self.embeddings.position_ids[:, : input_shape[1]]
                position_ids = position_ids.expand_as(input_ids)
                final_position_ids = position_ids

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, None, device, dtype=embedding_output.dtype
        )

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            bbox=final_bbox,
            position_ids=final_position_ids,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_height=patch_height,
            patch_width=patch_width,
        )

        sequence_output = encoder_outputs[0]

        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), poly_embedding


class PolyEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(PolyEncoder, self).__init__()
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
        self.norm = nn.LayerNorm(emb_dim)

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
        
        seq_attn_mask = sequences >= 0.

        # (B, seq_len, hidden_size)
        seq_embeddings = self.coord_embedding(sequences.unsqueeze(-1))  
        embeddings, attn_mask = self.add_cls_token(seq_embeddings, seq_attn_mask)
        embeddings = self.add_position_embeddings(embeddings)        
        outputs = self.encoder(inputs_embeds=embeddings, attention_mask=attn_mask)
        polygon_embedding = outputs.last_hidden_state        
        embeddings = polygon_embedding[:, 0, :]
        return self.norm(embeddings.reshape(ori_B, ori_N, -1))
            