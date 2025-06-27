import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LayoutLMv3Processor, LayoutLMv3TokenizerFast, LayoutLMv3ImageProcessor


def embed_l2_normalize(embed, dim = -1):
    '''
        embedding L2 normalize
        reference: https://github.com/gengchenmai/csp/blob/main/main/losses.py
    '''
    norm = torch.norm(embed, dim = dim, keepdim = True)
    return embed / norm

def get_nonlinear_layer(nonlinearity):
    if nonlinearity == 'identity':
        return nn.Identity()
    elif nonlinearity == 'sine':
        return Sine()
    elif nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'selu':
        return nn.SELU(inplace=True)
    elif nonlinearity == 'softplus':
        return nn.Softplus()
    elif nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif nonlinearity == 'gelu':
        return nn.GELU()
    elif nonlinearity == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2)
    else:
        raise NotImplementedError


class MLP(nn.Module):
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        hidden_dims,
        nonlinearity='relu',
    ):
        super().__init__()        
        layers = []
        last_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(get_nonlinear_layer(nonlinearity))            
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.reshape(-1, x.shape[-1]))
        return x.view(*shape, -1)


def get_processors(pretrained_model_name):
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
        pretrained_model_name, 
        padding='max_length', 
        is_split_into_words=True, 
        truncation=True,
        only_label_first_subword=True
    )
    image_processor = LayoutLMv3ImageProcessor.from_pretrained(
        pretrained_model_name, 
        apply_ocr=False
    )
    return tokenizer, image_processor





class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)  # To compute scalar attention score

    def forward(self, query, key, value, mask=None):
        # Shape: (batch_size, seq_len, hidden_dim)
        # Compute additive attention scores
        # W_q * Q + W_k * K + b
        query_proj = self.W_q(query).unsqueeze(2)  # (batch_size, seq_len_q, 1, hidden_dim)
        key_proj = self.W_k(key).unsqueeze(1)      # (batch_size, 1, seq_len_k, hidden_dim)
        
        # Broadcasting and applying tanh activation
        scores = torch.tanh(query_proj + key_proj)  # (batch_size, seq_len_q, seq_len_k, hidden_dim)
        scores = self.v(scores).squeeze(-1)         # (batch_size, seq_len_q, seq_len_k)
        
        # Apply mask: mask has shape (batch_size, seq_len_k)
        if mask is not None:
            # mask == 0 → mask this position, mask == 1 → keep this position
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        # Compute attention weights using softmax
        attn_weights = F.softmax(scores, dim=-1)  # Shape: (batch_size, seq_len_q, seq_len_k)
        
        # Compute weighted sum of values
        attn_output = torch.bmm(attn_weights, value)  # Shape: (batch_size, seq_len_q, hidden_dim)
        
        return attn_output, attn_weights


VALUE_MAX = 0.05
VALUE_MIN = 0.01


@torch.no_grad()
class PCAMaskEncoding(nn.Module):
    """
    To do the mask encoding of PCA.
        components_: (tensor), shape (n_components, n_features) if agnostic=True
                                else (n_samples, n_components, n_features)
        explained_variance_: Variance explained by each of the selected components.
                            (tensor), shape (n_components) if agnostic=True
                                        else (n_samples, n_components)
        mean_: (tensor), shape (n_features) if agnostic=True
                          else (n_samples, n_features)
        agnostic: (bool), whether class_agnostic or class_specific.
        whiten : (bool), optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
        sigmoid: (bool) whether to apply inverse sigmoid before transform.
    """
    def __init__(self, mask_size):
        super().__init__()
        self.agnostic = True 
        self.whiten = True 
        self.sigmoid = True 
        self.dim_mask = 60
        self.mask_size = 28 

        if self.agnostic:
            components_path = '/users/8/lin00786/work/MapText/text_linking/coco_2017_train_class_agnosticTrue_whitenTrue_sigmoidTrue_60_siz28.npz'
            parameters = np.load(components_path)
            self.components = nn.Parameter(torch.from_numpy(parameters['components_c'][0]).float(),requires_grad=False)
            self.explained_variances = nn.Parameter(torch.from_numpy(parameters['explained_variance_c'][0]).float(), requires_grad=False)
            self.means = nn.Parameter(torch.from_numpy(parameters['mean_c'][0]).float(),requires_grad=False)

        else:
            raise NotImplementedError

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        # In case of overflow
        value_random = VALUE_MAX * torch.rand_like(x)
        value_random = torch.where(value_random > VALUE_MIN, value_random, VALUE_MIN * torch.ones_like(x))
        x = torch.where(x > value_random, 1 - value_random, value_random)
        # inverse sigmoid
        y = -1 * torch.log((1 - x) / x)
        return y

    def encoder(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : Original features(tensor), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_transformed : Transformed features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.mask_size**2, print("The original mask_size of input"
                                                      " should be equal to the supposed size.")

        if self.sigmoid:
            X = self.inverse_sigmoid(X)

        if self.agnostic:
            if self.means is not None:
                X_transformed = X - self.means
            X_transformed = torch.matmul(X_transformed, self.components.T)
            if self.whiten:
                X_transformed /= torch.sqrt(self.explained_variances)
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        return X_transformed

    def decoder(self, X, is_train=False):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Parameters
        ----------
        X : Encoded features(tensor), shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original original features(tensor), shape (n_samples, n_features)
        """
        assert X.shape[1] == self.dim_mask, print("The dim of transformed data "
                                                  "should be equal to the supposed dim.")

        if self.agnostic:
            if self.whiten:
                components_ = self.components * torch.sqrt(self.explained_variances.unsqueeze(1))
            X_transformed = torch.matmul(X, components_)
            if self.means is not None:
                X_transformed = X_transformed + self.means
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        if is_train:
            pass
        else:
            if self.sigmoid:
                X_transformed = torch.sigmoid(X_transformed)
            else:
                X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)

        return X_transformed


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, 
                                               num_heads=num_heads, 
                                               dropout=dropout,
                                               batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, q, k, v):
        # Self-attention
        attn_output, _ = self.attention(q, k, v)
        # Residual connection and normalization
        x = self.norm(q + attn_output)
        return x



class CrossAttentionDecoderLayer(nn.Module):
    def __init__(
        self, 
        emb_dim=256, 
        d_ffn=1024, 
        num_heads=8,
        dropout=0.1 
    ):
        super(CrossAttentionDecoderLayer, self).__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, 
                                                num_heads=num_heads, 
                                                dropout=dropout,
                                                batch_first=True)    
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_dim)

        self.linear1 = nn.Linear(emb_dim, d_ffn)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, emb_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout1(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, emb1, emb2):
        emb1_2, _ = self.cross_attn(emb1, emb2, emb2)
        emb1_2 = emb1_2 + self.dropout(emb1)
        emb1_2 = self.norm(emb1_2)
        emb1_2 = self.forward_ffn(emb1_2)
        return emb1_2
        

class CrossAttentionDecoder(nn.Module):
    def __init__(
        self, 
        emb_dim=256, 
        d_ffn=1024, 
        num_heads=8,
        dropout=0.1  
    ):
        super(CrossAttentionDecoder, self).__init__()
        self.cross_attn1 = CrossAttentionDecoderLayer(emb_dim, 
                                                      d_ffn, 
                                                      num_heads, 
                                                      dropout)
        self.cross_attn2 = CrossAttentionDecoderLayer(emb_dim, 
                                                      d_ffn, 
                                                      num_heads, 
                                                      dropout)
    def forward(self, emb1, emb2):
        new_emb1 = self.cross_attn1(emb1, emb2)
        new_emb2 = self.cross_attn2(emb2, emb1)
        return new_emb1, new_emb2



class TokenEncoder(nn.Module):
    def __init__(self, emb_dim):
        super(TokenEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.mlp = MLP(emb_dim, emb_dim, [emb_dim], nonlinearity='relu')
        self.layer_norm = nn.LayerNorm(emb_dim)
            
    def forward(self, token_embeddings):
        embeddings = self.mlp(token_embeddings)
        return self.layer_norm(embeddings)















