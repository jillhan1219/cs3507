import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Model as GPT2ModelBase, GPT2Attention as GPT2AttentionBase
from typing import Optional, Tuple, Union
#from causal_product import causal_dot_product
from tools.mamba import MixerModel

# mamba
class MAMBADecoder(torch.nn.Module):
    def __init__(self, config):
        super(MAMBADecoder, self).__init__()
        self.gpt2 = GPT2Model(config)
        self.mamba = MixerModel(d_model= config.n_embd, n_layer=12)
        self.norm = nn.LayerNorm(config.n_embd)

    def forward(self, input_ids, attention_mask=None):
        embeddings = self.gpt2.wte(input_ids) + self.gpt2.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
        
        hidden_state = self.gpt2.drop(embeddings)
        
        outputs = self.mamba(hidden_state)
        outputs = self.norm(outputs)
        
        return outputs

# LPE embedding
class GPT2ModelWithLearnablePositionalEncoding(GPT2ModelBase):
    def __init__(self, config):
        super().__init__(config)
        self.wpe = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.init_weights()

    def forward(self, input_ids, **kwargs):
        input_shape = input_ids.size()
        position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        return super().forward(inputs_embeds=hidden_states, **kwargs)

# attention
# nystrom with random landmarks
class NystromAttention(nn.Module):
    def __init__(self, query_dimensions, num_landmarks=64, eps=1e-6):
        super(NystromAttention, self).__init__()
        self.query_dimensions = query_dimensions
        self.num_landmarks = num_landmarks
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, num_heads, seq_length, dim = queries.size()

        # Select landmarks
        num_landmarks = min(self.num_landmarks, seq_length)
        indices = torch.randint(0, seq_length, (num_landmarks,), device=queries.device)
        Q_landmarks = queries[:, :, indices, :]  # Shape: (batch_size, num_heads, num_landmarks, dim)
        K_landmarks = keys[:, :, indices, :]  # Shape: (batch_size, num_heads, num_landmarks, dim)

        # Compute the matrix forms
        # F_tilde = softmax(Q K_tilde^T / sqrt(d_q))
        F_tilde = torch.einsum('bhqd,bhkd->bhqk', queries / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), K_landmarks)
        F_tilde = F_tilde.softmax(dim=-1)

        # B_tilde = softmax(Q_tilde K^T / sqrt(d_q))
        B_tilde = torch.einsum('bhqd,bhkd->bhqk', Q_landmarks / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), keys)
        B_tilde = B_tilde.softmax(dim=-1)

        # A_tilde = softmax(Q_tilde K_tilde^T / sqrt(d_q))
        A_tilde = torch.einsum('bhqd,bhkd->bhqk', Q_landmarks / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), K_landmarks)
        A_tilde = A_tilde.softmax(dim=-1)

        # Compute the inverse of A_tilde
        A_tilde_inv = torch.inverse(A_tilde + self.eps * torch.eye(num_landmarks, device=queries.device).unsqueeze(0).unsqueeze(0))

        # Compute the final approximation
        context = torch.einsum('bhqk,bhkl,bhlm->bhqm', F_tilde, A_tilde_inv, B_tilde)
        context = torch.einsum('bhqm,bhmd->bhqd', context, values)

        return context, (F_tilde, A_tilde, B_tilde)

'''
# nystrom with average pooling
class NystromAttention(nn.Module):
    def __init__(self, query_dimensions, num_landmarks=64, eps=1e-6):
        super(NystromAttention, self).__init__()
        self.query_dimensions = query_dimensions
        self.num_landmarks = num_landmarks
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool1d(num_landmarks)

    def forward(self, queries, keys, values, attn_mask=None):
        batch_size, num_heads, seq_length, dim = queries.size()

        # Reshape and average pooling to select landmarks
        Q_pooled = queries.permute(0, 1, 3, 2).reshape(-1, dim, seq_length)  # Shape: (batch_size*num_heads, dim, seq_length)
        K_pooled = keys.permute(0, 1, 3, 2).reshape(-1, dim, seq_length)  # Shape: (batch_size*num_heads, dim, seq_length)

        Q_landmarks = self.avg_pool(Q_pooled).reshape(batch_size, num_heads, dim, self.num_landmarks).permute(0, 1, 3, 2)
        K_landmarks = self.avg_pool(K_pooled).reshape(batch_size, num_heads, dim, self.num_landmarks).permute(0, 1, 3, 2)

        # Compute the matrix forms
        # F_tilde = softmax(Q K_tilde^T / sqrt(d_q))
        F_tilde = torch.einsum('bhqd,bhkd->bhqk', queries / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), K_landmarks)
        F_tilde = F_tilde.softmax(dim=-1)

        # B_tilde = softmax(Q_tilde K^T / sqrt(d_q))
        B_tilde = torch.einsum('bhqd,bhkd->bhqk', Q_landmarks / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), keys)
        B_tilde = B_tilde.softmax(dim=-1)

        # A_tilde = softmax(Q_tilde K_tilde^T / sqrt(d_q))
        A_tilde = torch.einsum('bhqd,bhkd->bhqk', Q_landmarks / torch.sqrt(torch.tensor(dim, dtype=torch.float32)), K_landmarks)
        A_tilde = A_tilde.softmax(dim=-1)

        # Compute the inverse of A_tilde
        A_tilde_inv = torch.inverse(A_tilde + self.eps * torch.eye(self.num_landmarks, device=queries.device).unsqueeze(0).unsqueeze(0))

        # Compute the final approximation
        context = torch.einsum('bhqk,bhkl,bhlm->bhqm', F_tilde, A_tilde_inv, B_tilde)
        context = torch.einsum('bhqm,bhmd->bhqd', context, values)

        return context, (F_tilde, A_tilde, B_tilde)
'''

# linear attention
# 分别对Q和K进行softmax
class SimplifiedLinearAttention(nn.Module):
    """Implement attention using dot product of Q and K with softmax applied to Q and K separately.
    
    Given the queries, keys and values as Q, K, V instead of computing
    
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    
    we compute
    
        V' = softmax(Q, dim=1).mm(softmax(K, dim=2).t()).mm(V).
    
    Arguments
    ---------
        query_dimensions: int, the dimensions of the queries
        eps: float, a small number to ensure the numerical stability of the denominator (default: 1e-6)
    """
    def __init__(self, query_dimensions, eps=1e-6):
        super(SimplifiedLinearAttention, self).__init__()
        self.query_dimensions = query_dimensions
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask=None):
        # Apply softmax to Q and K along the respective dimensions
        Q = F.softmax(queries, dim=2)
        K = F.softmax(keys, dim=1)
        
        # Compute the attention weights using dot product of Q and K
        attention_weights = torch.einsum('nhqd,nhkd->nhqk', Q, K)

        if attn_mask is not None:
            attention_weights = attention_weights.masked_fill(attn_mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_weights, dim=-1)

        # Compute the attention output
        context = torch.einsum('nhqk,nhkd->nhqd', attention_weights, values)

        return context, attention_weights

# linear attention
# 使用核函数
class LinearAttention(nn.Module):
    """Implement attention using dot product of feature maps in O(N D^2) complexity.
    
    Given the queries, keys and values as Q, K, V instead of computing
    
        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),
    
    we make use of a feature map function Φ(.) and perform the following computation
    
        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).
    
    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.
    
    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            EluFeatureMap(query_dimensions)
        )
        self.eps = eps

    def forward(self, queries, keys, values, attn_mask=None, head_mask=None):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        #print(f'Q:{Q.shape}')
        K = self.feature_map.forward_keys(keys)
        #print(f'K:{K.shape}')

        # Compute the attention weights using dot product of Q and K
        attention_weights = torch.einsum('nhqd,nhkd->nhqk', Q, K)

        #print(f'attention mask shape:{attn_mask.shape}')
        #print(f'attention weights shape:{attention_weights.shape}')
        if attn_mask is not None:
            attention_weights = attention_weights.masked_fill(attn_mask == 0, float('-inf'))

        attention_weights = F.softmax(attention_weights, dim=-1)

        # Compute the attention output
        #print(f'V:{values.shape}')
        context = torch.einsum('nhqk,nhkd->nhqd', attention_weights, values)

        return context, attention_weights


class EluFeatureMap(nn.Module):
    def __init__(self, query_dimensions):
        super(EluFeatureMap, self).__init__()
        self.query_dimensions = query_dimensions

    def forward_queries(self, queries):
        return F.elu(queries) + 1

    def forward_keys(self, keys):
        return F.elu(keys) + 1

    def new_feature_map(self, device):
        pass

class CustomGPT2Attention(GPT2AttentionBase):
    def __init__(self, config, attention_type='linear'):
        super().__init__(config)
        self.attention_type = attention_type
        if attention_type == 'linear1':
            self.custom_attention = LinearAttention(config.hidden_size)
        elif attention_type == 'linear2':
            self.custom_attention = SimplifiedLinearAttention(config.hidden_size)
        elif attention_type == 'nystrom':
            self.custom_attention = NystromAttention(config.hidden_size)
        else:
            raise ValueError("Unsupported attention type")

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            #attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
            #print('here')
            attn_output, attn_weights = self.custom_attention(query, key, value, attention_mask )

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class CustomGPT2Model(GPT2Model):
    def __init__(self, config, attention_type='linear'):
        super().__init__(config)
        for i, layer in enumerate(self.h):
            layer.attn = CustomGPT2Attention(config, attention_type=attention_type)


