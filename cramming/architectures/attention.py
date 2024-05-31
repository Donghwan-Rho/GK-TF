"""Attention modules. The final model uses "self-attention", but other options were tried and are still documented here."""
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange
import logging
from .attention_modified import SeqFirstSelfAttention_modified
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn.functional as F

log = logging.getLogger(__name__)

def subtraction_gaussian_kernel_torch(q, k):
    k = k.transpose(-1, -2)
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    return matA_square + matB_square - 2. * (q @ k)

def get_attention_mechanism(
    idx,
    hidden_size,
    cfg_attention,
):
    # print('cfg_attention', cfg_attention)
    # print('cfg_attention.type', cfg_attention.type)
    cfg_attention.type = cfg_attention['type']
    
    if cfg_attention.type == "self-attention":
        mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)  # neox
    elif cfg_attention.type == "self-attention-modified":
        mechanism = SeqFirstSelfAttention_modified(hidden_size, cfg_attention)  # neox
    elif cfg_attention.type == "pytorch":
        # Sanity check 1: [Warning: This includes the output projection twice...]
        mechanism = SelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "pytorch-seqfirst":
        # Sanity check 1: [Warning: This includes the output projection twice...]
        mechanism = SeqFirstSelfAttentionPyTorch(hidden_size, cfg_attention)  # torch default
    elif cfg_attention.type == "huggingface":
        mechanism = BertAttentionWrapper(hidden_size, cfg_attention)  # always includes bias!
    elif cfg_attention.type == "flash-attention-impl":  # the fast implementation called flash
        mechanism = FlashMultiHeadAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "fourier":
        mechanism = FourierMixing(hidden_size, cfg_attention)
    elif cfg_attention.type == "fourier-experimental":
        mechanism = FourierMixingParametrized(hidden_size, cfg_attention)
    elif cfg_attention.type == "flash":  # flash from transformer quality in linear time
        mechanism = FLASH(hidden_size, cfg_attention)
    elif cfg_attention.type == "tuformer":
        mechanism = TuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "funnel":  # dont use this with a normal seq->seq model
        mechanism = FunnelAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "seqfirst_tuformer":
        mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "seqfirst2_tuformer":
        mechanism = SeqFirstTuFormAttention(hidden_size, cfg_attention)
    elif cfg_attention.type == "none":
        mechanism = Identity(hidden_size)
    elif cfg_attention.type == "fourier-hybrid":
        if idx in cfg_attention.hybrid_layers:
            mechanism = SeqFirstSelfAttention(hidden_size, cfg_attention)
        else:
            mechanism = FourierMixing(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism


class Identity(torch.nn.Module):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size):
        super().__init__()
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return hidden_states


class BertAttentionWrapper(BertSelfAttention):
    """mini wrapper around BERT attention from huggingface for sanity checks."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        class config:
            pass

        config.hidden_size = hidden_size
        config.num_attention_heads = cfg_attention.num_attention_heads
        config.attention_probs_dropout_prob = cfg_attention.dropout_prob
        config.is_decoder = False

        super().__init__(config)
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return super().forward(hidden_states, attention_mask)[0]


class SelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=cfg_attention.dropout_prob,
            batch_first=True,
            bias=False,
            add_bias_kv=cfg_attention.qkv_bias,
        )

        # Do something terrible to patch the fact that the output projection is somewhere else in our code:
        del self.attn.out_proj.weight
        del self.attn.out_proj.bias
        self.attn.out_proj.register_buffer("weight", torch.eye(hidden_size))
        self.attn.out_proj.register_buffer("bias", torch.zeros(hidden_size))
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask, need_weights=False)[0]


class SeqFirstSelfAttentionPyTorch(torch.nn.Module):
    """Minimal wrapper around pytorch self attention."""

    __constants__ = ["LAYOUT"]
    LAYOUT = "[S B H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            hidden_size,
            cfg_attention.num_attention_heads,
            dropout=cfg_attention.dropout_prob,
            batch_first=False,
            bias=False,
            add_bias_kv=cfg_attention.qkv_bias,
        )

        # Do something terrible to patch the fact that the output projection is somewhere else in our code:
        del self.attn.out_proj.weight
        del self.attn.out_proj.bias
        self.attn.out_proj.register_buffer("weight", torch.eye(hidden_size))
        self.attn.out_proj.register_buffer("bias", torch.zeros(hidden_size))
        self.output_dim = hidden_size

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        return self.attn(hidden_states, hidden_states, hidden_states, attn_mask=attention_mask, need_weights=False)[0]


class LegacySeqFirstSelfAttention(torch.nn.Module):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def __init__(self, hidden_size: int, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        # print('self.hidden_per_head', self.hidden_per_head)
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())

        # Strided linear layer.
        # Linear(768, 2304)
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size # 768
        # print('self.query_key_value', self.query_key_value)
        # print('self.output_dim', self.output_dim)
        if cfg_attention.rotary_embedding == "sanity":
            # print('cfg_attention.rotary_embedding == "sanity":')
            self.rotary_emb = RotarySanityCheck(self.hidden_per_head, seq_dim=0)
        elif cfg_attention.rotary_embedding == "v2":
            # print('cfg_attention.rotary_embedding == "v2":')
            self.rotary_emb = RotaryEleutherAI(self.hidden_per_head)
        elif cfg_attention.rotary_embedding == "llama":
            # print('cfg_attention.rotary_embedding == "llama":')
            self.rotary_emb = RotaryLLAMA(self.hidden_per_head)
        elif cfg_attention.rotary_embedding:
            # print('cfg_attention.rotary_embedding:')
            self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            # print('self.rotary_emb = None')
            # 여기
            self.rotary_emb = None

        if cfg_attention.sequence_op == "torch-softmax":
            self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "torch-relu":
            self.sequence_op = TorchReLU(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "torch-relu-norm":
            self.sequence_op = TorchReLU_Norm(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "torch-norm":
            self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "exp":
            self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "poly":
            self.sequence_op = Polynorm(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "none":
            self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsum":
            self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
        elif cfg_attention.sequence_op == "cumsumexp":
            self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
        else:
            raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout: float = cfg_attention.dropout_prob

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # print('output_size', output_size)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )  # this looks crazy but beta=0 below skips the values of this tensor [so beta is NOT optional...]

        # kernel로 수정 필요
        ################################
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result, # input
            query_layer.transpose(0, 1),  # [b * np, sq, hn] # batch 1
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk] # batch 2
            beta=0.0, # beta
            alpha=self.norm_factor, # alpha
        )# output = beta * input + alpha * (batch 1) @ (batch 2)
        ################################

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])

        # ===========================
        # Attention probs and dropout
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.sequence_op(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        # attention_probs = self.attention_dropout(attention_probs)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1)) # softmax와 V 행렬곱

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer = self.query_key_value(hidden_states) # 128 128 2304
        # log.info('mixed_x_layer: {}'.format(mixed_x_layer))
        # torch._dynamo.config.verbose=True
        # torch._dynamo.config.suppress_errors = True
        # print('\n============== Legacy forward ==============')
        # print('Legacy forward hidden_states', hidden_states.shape) # 128 128 768
        # print('Legacy forward mixed_x_layer', mixed_x_layer.shape)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        # new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_per_head)
        mixed_x_layer = mixed_x_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
        )
        # print('after view Legacy forward mixed_x_layer', mixed_x_layer.shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)
        # print('Legacy forward query_layer', query_layer.shape) # 128 128 12 64
        
        if self.rotary_emb is not None:
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)

        # ==================================
        # Attention computation
        # ==================================
        
        # context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        
        
        # Ordinary
        #######################################
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        #######################################
        # # Ordinary-matmul
        # #######################################
        # context_layer, matmul_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # #######################################
        # # Heatmap
        # #######################################
        # context_layer, matmul_result, heatmap_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # # context_layer, matmul_result, norm_sum = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # context_layer, matmul_result, query_norm, key_norm = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # #######################################
        
        # # print('legacy forward context_layer', context_layer[0].shape, context_layer[1].shape)
        # # print('Legacy forward context_layer', context_layer.shape)
        # # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()
        # # print('after permute Legacy forward context_layer', context_layer.shape)

        # # [sq, b, np, hn] --> [sq, b, hp]
        # # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size)
        # # print('after view Legacy forward context_layer', context_layer.shape)
        # # print('============== Legacy forward end ==============\n')
        
        # Ordinary-matmul
        # #######################################
        return context_layer
        # #######################################
        # # Ordinary-matmul
        # # #######################################
        # return context_layer, matmul_result
        # # #######################################
        # # Heatmap
        # #######################################
        # return context_layer, matmul_result, heatmap_result
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # # return context_layer, matmul_result, norm_sum
        # return context_layer, matmul_result, query_norm, key_norm
        # #######################################

class SeqFirstSelfAttention(LegacySeqFirstSelfAttention):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # print('self', self)
        query_layer = query_layer.to(dtype=torch.float)
        key_layer = key_layer.to(dtype=torch.float)
        value_layer = value_layer.to(dtype=torch.float)
        '''Linear attention'''
        ##############################################################################
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # print('Seq attention output_size', output_size)
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print('after view Seq attention query_layer', query_layer.shape)
        # print('query norm', torch.norm(query_layer, p=float('inf')).item())
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print('key norm', torch.norm(key_layer, p=float('inf')).item())
        # print('after view Seq attention key_layer', key_layer.shape)
        value_layer = value_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print('lin value_layer', value_layer.shape)
        # print('value norm', torch.norm(value_layer, p=float('inf')).item())
        
        query_layer = query_layer.transpose(0, 1)
        key_layer = key_layer.transpose(0, 1)
        value_layer = value_layer.transpose(0, 1)
        # print('after transpose q', query_layer.shape)
        # print('after transpose k', key_layer.shape)
        # print('k shape', key_layer.shape[1])
        
        kernel_ftn = torch.nn.ReLU()
        query_layer = kernel_ftn(query_layer)
        key_layer = kernel_ftn(key_layer)
        
        # kernel_ftn = torch.nn.ELU()
        # query_layer = kernel_ftn(query_layer) + 1
        # key_layer = kernel_ftn(key_layer) + 1
        
        # value_layer = relu(value_layer)
        print('after relu')
        print('query norm', torch.norm(query_layer, p=float('inf')).item())
        print('q min', torch.min(query_layer).item())
        print('key norm', torch.norm(key_layer, p=float('inf')).item())
        print('k min', torch.min(key_layer).item())
        print('value norm', torch.norm(value_layer, p=float('inf')).item())
        print('v min', torch.min(value_layer).item())
        
        final_output_size = (output_size[0], output_size[1], output_size[2], query_layer.shape[2])
        # print('final output_size', final_output_size)
        
        kTv = torch.einsum('Bsm,Bsn->Bmn', key_layer, value_layer)
        # print('kTv', kTv.shape)
        print('kTv', torch.norm(kTv, p=float('inf')).item())
        print('kTv min', torch.min(kTv).item())
        
        
        # Denominator
        min_eps = 1e-6
        # max_eps = 1e+2
        k_col_sum = torch.sum(key_layer, axis=1) # sum K_j^T
        if torch.isnan(k_col_sum).any():
            print('nan')
            return
        # print(float('nan'))
        qk_col = torch.einsum('Bsm,Bm->Bs', query_layer, k_col_sum) # Q*sum K_j^T
        if torch.isnan(qk_col).any():
            print('nan')
            return
        # denom = 1 / (torch.clamp(qk_col, min=min_eps, max=max_eps)) # denominator
        denom = 1. / (torch.clamp(qk_col, min=min_eps) + min_eps) # denominator
        if torch.isnan(denom).any():
            print('nan')
            return
        print('k_col_sum max', torch.max(k_col_sum).item())
        print('k_col_sum', torch.norm(k_col_sum, p=float('inf')).item())
        print('k_col_sum min', torch.min(k_col_sum).item())
        print('qk_col', torch.norm(qk_col, p=float('inf')).item())
        print('qk_col min', torch.min(qk_col).item())
        print('denom', torch.norm(denom, p=float('inf')).item())
        print('denom min', torch.min(denom).item())
        # denom = 1. / torch.sqrt(torch.tensor([key_layer.shape[1]]))
        # denom = denom.to('cuda')
        
        qkTv = torch.einsum('Bsm,Bmn->Bsn', query_layer, kTv)
        print('qkTv max', torch.max(qkTv).item())
        print('qkTv', torch.norm(qkTv, p=float('inf')).item())
        if torch.isnan(qkTv).any():
            print('nan')
            return
        attn_output = torch.einsum('Bsn,Bs->Bsn', qkTv, denom)
        if torch.isnan(attn_output).any():
            print('nan')
            return
        # attn_output = qkTv * denom
        
        # print('attn_output', attn_output.shape)
        print('attn_output', torch.norm(attn_output, p=float('inf')).item())
        print('attn_output min', torch.min(attn_output).item())
        attn_output = attn_output.view(*final_output_size)
        # print('final attn_output', attn_output.shape)
        # print('attn_output', torch.norm(attn_output, p=float('inf')).item())
        
        
        return attn_output
        ##############################################################################
        
        # '''Softmax attention(Ordinary)'''
        # ##############################################################################
      
        # # ===================================
        # # Raw attention scores. [b, np, s, s]
        # # ===================================
        # # [b, np, sq, sk]
        # # print('\n============== Seq attention ==============')
        # # print('Seq attention query_layer', query_layer.shape)
        # # print('Seq attention key_layer', key_layer.shape)
        # # print('Seq attention value_layer', value_layer.shape)
        
        # output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # print('Seq attention output_size', output_size)
        # # [sq, b, np, hn] -> [sq, b * np, hn]
        # query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print('after view Seq attention query_layer', query_layer.shape)
        # key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # # print('after view Seq attention key_layer', key_layer.shape)
        
        # #######################################
        # # this better be fused in a clever way:
        # # QK^T
        # # print('\n============== matmul ==============')
        # # print('before transpose query_layer', query_layer.shape)
        # # print('after transpose query_layer', query_layer.transpose(0, 1).shape)
        # # print('before transpose key_layer', key_layer.shape)
        # # print('after transpose 1 key_layer', key_layer.transpose(0, 1).shape)
        # # print('after transpose 2 key_layer', key_layer.transpose(0, 1).transpose(1, 2).shape)        
        # # gpu 0, self.norm_factor: 0.0361
        
        # # # L2 normalization scheme
        # # #######################################
        # # query_layer_matmul = F.normalize(query_layer_matmul, p=2.0, dim=2)
        # # key_layer_matmul = F.normalize(key_layer_matmul, p=2.0, dim=2)
        
        # # query_layer_matmul *= 10
        # # key_layer_matmul *= 10
        # # #######################################
        
        # # # Q, K L2 normalization scheme
        # # #######################################
        # # query_layer_matmul = query_layer.transpose(0, 1)
        # # key_layer_matmul = key_layer.transpose(0, 1)
        # # print('query_layer_matmul', query_layer_matmul.shape)
        # # print('key_layer_matmul', key_layer_matmul.shape)
        
        # # query_norm = torch.norm(query_layer_matmul, p=2, dim=2)
        # # # print('query_norm', query_norm[0])
        # # key_norm = torch.norm(key_layer_matmul, p=2, dim=2)
        # # # print('query norm mean', torch.mean(query_norm))
        # # # print('key norm mean', torch.mean(key_norm))
        
        # # # query_norm = query_norm - 8.4
        # # # key_norm = key_norm - 8.4
        
        # # # query_norm = torch.square(query_norm)
        # # # key_norm = torch.square(key_norm)
        
        # # # query_norm_sum = torch.sum(query_norm)
        # # # key_norm_sum = torch.sum(key_norm)
        # # # norm_sum = query_norm_sum + key_norm_sum
        # # #######################################
        
        # # print('seq att matmul 전 query', query_layer_matmul.shape)
        # # print('seq att matmul 전 key', key_layer_matmul.shape)
        
        # # print('1/self.norm_factor', 1/self.norm_factor) # 8
        
        # query_layer_before_bmm = query_layer.transpose(0, 1)
        # key_layer_before_bmm = key_layer.transpose(0, 1)
        # print('matmul 전 q', query_layer_before_bmm.shape)
        # print('matmul 전 k', key_layer_before_bmm.shape)
        # print('matmul 전 v', value_layer.shape)
        
        # matmul_result = torch.bmm(query_layer_before_bmm, key_layer_before_bmm.transpose(1, 2)) * self.norm_factor
        # print('seq att matmul_result', matmul_result.shape)
        # # print('matmul_result.get_device()', matmul_result.get_device())
        # # print('self.norm_factor', self.norm_factor)
        # # print('Seq attention matmul_result', matmul_result.shape)
        # # print('============== matmul end ==============\n')
        # #######################################
        
        # # #######################################
        # # shape_0 = query_layer.shape[1] # 32
        # # shape_1 = query_layer.shape[0] # 128
        # # shape_2 = query_layer.shape[0] # 128
        # # query_layer_ = query_layer.transpose(0, 1)
        # # key_layer_ = key_layer.transpose(0, 1)
        
        # # matmul_result = torch.zeros(shape_0, shape_1, shape_2)
        # # matmul_result = matmul_result.to('cuda')
        
        # # for b in range(shape_0):
        # #     for row in range(shape_1):
        # #         for col in range(shape_2):
        # #             matmul_result[b][row][col] = torch.norm(query_layer_[b][row] - key_layer_[b][col]) ** 2
        
        # # matmul_result *= -self.norm_factor * 0.5
        # # print('matmul_result', matmul_result.shape)
        # #######################################
        
        
        # # change view to [b, np, sq, sk]
        # attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        # print('seq att attention_scores', attention_scores.shape)
        # # print('seq att head 0 att score', attention_scores[0][0])

        # # # Heatmap
        # # #######################################
        # # heatmap_result = torch.squeeze(attention_scores, 0)
        # # heatmap_result = torch.mean(heatmap_result, dim=0)
        # # # print('heatmap_results', heatmap_results.shape)
        # # #######################################
        
        # # ===========================
        # # Attention probs and dropout
        # # ===========================
        # # attention scores and attention mask [b, np, sq, sk]
        # # sequence: softmax 등 적용
        # attention_probs = self.sequence_op(attention_scores, attention_mask)
        # print('seq att attention_probs', attention_probs.shape)
        # # This is actually dropping out entire tokens to attend to, which might
        # # seem a bit unusual, but is taken from the original Transformer paper.
        # # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        # # attention_probs = self.attention_dropout(attention_probs)
        
        # attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        # # attention_probs = torch.nn.functional.dropout(attention_probs, p=0, training=training)
        
        # # =========================
        # # Context layer. [sq, b, hp]
        # # =========================

        # # value_layer -> context layer.
        # # [sk, b, np, hn] --> [b, np, sq, hn]

        # # context layer shape: [b, np, sq, hn]
        # output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # print('Seq attention output_size', output_size)

        # # change view [sk, b * np, hn]
        # value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # print('Seq attention value_layer', value_layer.shape)

        # # change view [b * np, sq, sk]
        # attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # print('after view Seq attention attention_probs', attention_probs.shape)
        # # matmul: [b * np, sq, hn]
        # # softmax 결과(prob dist)와 V 곱함
        # context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        # print('Seq attention context_layer', context_layer.shape)
        
        # # change view [b, np, sq, hn]
        # context_layer = context_layer.view(*output_size)
        # print('after view Seq attention context_layer', context_layer.shape)
        # # print('============== Seq attention end ==============\n')
        
        # # Ordinary
        # # #######################################
        # return context_layer
        # # #######################################
        # # # Ordinary-matmul
        # # # #######################################
        # # return context_layer, matmul_result
        # # # #######################################
        # # # Heatmap
        # # #######################################
        # # return context_layer, matmul_result, heatmap_result
        # # #######################################
        # # # Q, K L2 normalization scheme
        # # #######################################
        # # # return context_layer, matmul_result, norm_sum
        # # return context_layer, matmul_result, query_norm, key_norm
        # # #######################################
        # ##############################################################################
    
class FlashMultiHeadAttention(torch.nn.Module):
    """Wrapper for flash MHA."""

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        from flash_attn.flash_attention import FlashMHA

        self.flash_mha = FlashMHA(
            hidden_size,
            cfg_attention.num_attention_heads,
            bias=cfg_attention.qkv_bias,
            batch_first=True,
            attention_dropout=cfg_attention.dropout_prob,
            causal=cfg_attention.causal_attention,
        )
        hidden_per_head = hidden_size // self.flash_mha.num_heads
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(hidden_per_head, seq_dim=1))
            else:
                self.rotary_emb = Rotary(hidden_per_head, seq_dim=1)
        else:
            self.rotary_emb = None

        self.flash_mha.out_proj = None
        self.output_dim = hidden_size

    @torch.jit.ignore  # This jit.ignore call is ignored?
    def flash_inner(self, qkv):
        return self.flash_mha.inner_attn(qkv, key_padding_mask=None, need_weights=False, causal=self.flash_mha.causal)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # def forward(self, x, key_padding_mask=None, need_weights=False):
        """x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim)
        key_padding_mask: bool tensor of shape (batch, seqlen)

        Returns only the rearranged, unprojected output
        """
        qkv = self.flash_mha.Wqkv(hidden_states)
        if self.rotary_emb is not None:
            query, key, value = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads).unbind(dim=2)
            query, key = self.rotary_emb(query, key)
            qkv = torch.stack([query.type(qkv.dtype), key.type(qkv.dtype), value.type(qkv.dtype)], dim=2)
        else:
            qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.flash_mha.num_heads)
        context, attn_weights = self.flash_inner(qkv)
        return rearrange(context, "b s h d -> b s (h d)")


class FunnelAttention(SeqFirstSelfAttention):
    """Self-attention layer abstract class.

    This is a funnel crammed into the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    Self-attention layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT", "attention_dropout", "length_factor"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def __init__(self, hidden_size, cfg_attention, length_factor=1.0):
        super().__init__(hidden_size, cfg_attention)
        self.length_factor: float = length_factor

        # Strided linear layers
        del self.query_key_value
        self.query = torch.nn.Linear(self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.key_value = torch.nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=cfg_attention.qkv_bias)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================

        # ==================================
        #  Pool or unpool states
        # ==================================
        sq, b = hidden_states.shape[0], hidden_states.shape[1]

        # [sq, b, h] -> [sq * F, b, h]
        new_seq_length = int(sq * self.length_factor)
        if self.length_factor < 1:
            query_states = hidden_states.view(int(1 / self.length_factor), new_seq_length, b, self.hidden_size).mean(dim=0)
        elif self.length_factor > 1:
            query_states = hidden_states.repeat_interleave(int(self.length_factor), dim=0, output_size=new_seq_length)
        else:
            query_states = hidden_states

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        query_layer = self.query(query_states).view(new_seq_length, b, self.num_attention_heads, self.hidden_per_head)
        mixed_x_layer = self.key_value(hidden_states).view(sq, b, self.num_attention_heads, 2 * self.hidden_per_head)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 2, dim=3)

        if self.rotary_emb is not None:
            query_layer = self.rotary_emb.single_forward(query_layer)
            key_layer = self.rotary_emb.single_forward(key_layer)

        # ==================================
        # Attention computation
        # ==================================
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_seq_length, context_layer.shape[1], self.hidden_size)
        return context_layer


class TuFormAttention(torch.nn.Module):
    """Self-attention layer abstract class.

    This is a simplification of the tuformer implementationfrom
    https://github.com/xliu1231/fairseq_tuformer/blob/main/fairseq/modules/tuckerhead_attention.py

    THSA layer takes input with size [Batch, Seq, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.hidden_size = hidden_size
        self.rdim = getattr(cfg_attention, "rdim", hidden_size)
        self.register_buffer("norm_factor", torch.tensor(self.rdim).rsqrt())

        # Strided linear layer.
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.rdim, bias=cfg_attention.qkv_bias)
        self.c_proj = torch.nn.Linear(self.rdim, self.rdim, bias=cfg_attention.qkv_bias)
        self.output_dim = self.rdim

        if cfg_attention.rotary_embedding:
            raise ValueError("Have to think about dimensions here.")

        if cfg_attention.sequence_op == "torch-softmax":
            self.sequence_op = torch.jit.script(TorchSoftmax(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "torch-norm":
            self.sequence_op = torch.jit.script(TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "none":
            self.sequence_op = torch.jit.script(ScaledIdentity(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "cumsum":
            self.sequence_op = torch.jit.script(Cumsum(cfg_attention.seq_op_in_fp32))
        elif cfg_attention.sequence_op == "cumsumexp":
            self.sequence_op = torch.jit.script(CumsumExp(cfg_attention.seq_op_in_fp32))
        else:
            raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")

        self.attention_dropout = torch.nn.Dropout(cfg_attention.dropout_prob, inplace=False)  # cannot be inplace
        self.first_rearrange = Rearrange("b s l r -> (b r) s l", r=self.rdim)
        self.second_rearrange = Rearrange("(b r) s l -> b r s l", r=self.rdim)

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("bsr, blr -> bslr", query_layer, key_layer))

        attention_scores = self.sequence_op(self.first_rearrange(attention_scores), attention_mask)
        attention_scores = self.attention_dropout(attention_scores)

        return torch.einsum("brsl, blr -> bsr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class SeqFirstTuFormAttention(TuFormAttention):
    """Self-attention layer abstract class.

    Seq-first variant 1

    THSA layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__(hidden_size, cfg_attention)
        self.first_rearrange = Rearrange("b s l r -> (b r) s l", r=self.rdim)
        self.second_rearrange = Rearrange("(b r) s l -> b r s l", r=self.rdim)

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("sbr, lbr -> bslr", query_layer, key_layer))

        attention_scores = self.sequence_op(self.first_rearrange(attention_scores), attention_mask)
        attention_scores = self.attention_dropout(attention_scores)
        return torch.einsum("brsl, lbr -> sbr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class SeqFirstTuFormAttention2(TuFormAttention):
    """Self-attention layer abstract class.

    Seq-first variant 2

    THSA layer takes input with size [Seq, Batch, Hidden]
    and returns output of the same size.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT: str = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__(hidden_size, cfg_attention)
        self.first_rearrange = Rearrange("s l b r -> s l (b r)", r=self.rdim)
        self.second_rearrange = Rearrange("s l (b r) -> s l b r", r=self.rdim)
        if cfg_attention.sequence_op != "torch-softmax":
            raise ValueError("Not implemented")

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None):
        """tuformer attention in batch first implementation (hopefully)"""
        attention_scores = self.c_proj(torch.einsum("sbr, lbr -> slbr", query_layer, key_layer))

        attention_scores = self.first_rearrange(attention_scores).softmax(dim=1)
        attention_scores = self.attention_dropout(attention_scores)
        return torch.einsum("slbr, lbr -> sbr", self.second_rearrange(attention_scores), value_layer)

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        mixed_x_layer = self.query_key_value(hidden_states)  # b s 3r

        # 3 [ b s r]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.rdim] * 3, dim=-1)
        context_layer = self.attention(query_layer, key_layer, value_layer, attention_mask)  # b s r
        return context_layer


class FourierMixing(torch.nn.Module):
    """Fourier mixing layer as described in the FNet paper.
    Layer takes input with size [Batch, Seq, Hidden] and returns output of the same size.
    This function can take an attention mask as input, but will ignore it.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention):
        super().__init__()
        self.fft_op_in_fp32 = True  # Always necessary (atleast on pytorch 1.12)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(hidden_size, seq_dim=1))
            else:
                self.rotary_emb = Rotary(hidden_size, seq_dim=0)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Forward will take an attention mask but ignore it!"""

        if self.rotary_emb is not None:
            # full rotary (mostly on for compatibility, no guarantees on this being non-terrible)
            cos, sin = self.rotary_emb.get_cos_sin_cache(hidden_states)
            hidden_states = (hidden_states * cos[:, 0]) + (self.rotary_emb.rotate_half(hidden_states) * sin[:, 0])

        if self.fft_op_in_fp32:
            hidden_state_dtype = hidden_states.dtype
            hidden_states = hidden_states.float()
        else:
            hidden_state_dtype = None

        # Implementation 1:
        # hidden_states = torch.fft.fft(torch.fft.fft(hidden_states, dim=0, , norm="ortho"), dim=2, , norm="ortho").real
        # Implementation 2:
        hidden_states = torch.fft.fftn(hidden_states, dim=(1, 2), norm="ortho").real  # could also cast into angle?

        if self.fft_op_in_fp32:
            hidden_states = hidden_states.to(hidden_state_dtype)

        return hidden_states


class FourierMixingParametrized(torch.nn.Module):
    """Fourier mixing layer as described in the FNet paper.
    Layer takes input with size [Seq, batch, Hidden] and returns output of the same size.
    This function can take an attention mask as input, but will ignore it.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[S B H]"

    def __init__(
        self,
        hidden_size,
        cfg_attention,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        self.fft_op_in_fp32 = True  # Always necessary (atleast on pytorch 1.12)

        # linear layer.
        self.projection = torch.nn.Linear(2 * self.hidden_size, self.hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size
        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(self.hidden_per_head, seq_dim=0))
            else:
                self.rotary_emb = Rotary(self.hidden_per_head, seq_dim=0)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        """Forward will take an attention mask but ignore it!"""

        # [S, B, (np * hn)] --> [S, B, np, hn]
        head_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, self.hidden_per_head)
        if self.rotary_emb is not None:
            # full rotary (mostly on for compatibility, no guarantees on this being non-terrible)
            cos, sin = self.rotary_emb.get_cos_sin_cache(head_states)
            hidden_states = (head_states * cos[:, 0]) + (self.rotary_emb.rotate_half(head_states) * sin[:, 0])

        if self.fft_op_in_fp32:
            hidden_state_dtype = hidden_states.dtype
            head_states = head_states.float()
        else:
            hidden_state_dtype = None

        # Implementation 2:
        complex_scores = torch.fft.fftn(head_states, dim=(2, 3), norm="ortho")
        # complex [S, B, np, hn] -> [S, B, 2 * np * hn]
        # need to restride for this :<
        head_states = torch.view_as_real(complex_scores).reshape(hidden_states.shape[0], hidden_states.shape[1], -1)

        if self.fft_op_in_fp32:
            head_states = head_states.to(hidden_state_dtype)

        hidden_states = self.projection(head_states)

        return hidden_states


class FLASH(torch.nn.Module):
    """FLASH as described in Transformer Quality in Linear Time.
    This is FLASH-QUAD, as we're not too interested in long-range sequences here.
    """

    __constants__ = ["LAYOUT"]
    LAYOUT = "[B S H]"

    def __init__(self, hidden_size, cfg_attention, expansion_factor: int = 2, s: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.e = hidden_size * expansion_factor
        self.s = s
        self.uv_projection = torch.nn.Linear(hidden_size, 2 * self.e + self.s, bias=cfg_attention.qkv_bias)
        self.nonlin = torch.nn.SiLU(inplace=False)
        self.gamma = torch.nn.Parameter(torch.randn(2, s) * 0.02)
        self.beta = torch.nn.Parameter(torch.zeros(2, s))

        self.out_projection = torch.nn.Linear(self.e, hidden_size, bias=cfg_attention.qkv_bias)
        self.output_dim = hidden_size

        if cfg_attention.rotary_embedding:
            if cfg_attention.low_level_fusion:
                self.rotary_emb = torch.jit.script(Rotary(self.s, seq_dim=1))
            else:
                self.rotary_emb = Rotary(self.s, seq_dim=1)
        else:
            self.rotary_emb = None

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Inputs of shape [B, S, H]. Implementation directly based on FLASH pseudocode (see paper appendix)"""
        u_v_base = self.nonlin(self.uv_projection(inputs))
        u, v, base = torch.split(u_v_base, [self.e, self.e, self.s], dim=-1)
        base = torch.einsum("...r,hr->...hr", base, self.gamma) + self.beta
        if self.rotary_emb is not None:
            base = self.rotary_emb.single_forward(base)
        query, key = torch.unbind(base, dim=2)

        attention_scores = query.matmul(key.transpose(1, 2)) / inputs.shape[1]
        squared_scores = torch.nn.functional.relu(attention_scores).pow(2)
        return self.out_projection(u * torch.einsum(" bnm,bme->bne ", squared_scores, v))


class TorchSoftmax(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        # print('Torchsoftmax')
        # print('softmax inputs', inputs.dtype)
        # print('attention_mask', attention_mask.shape)
        if attention_mask is not None:
            # print('attention_mask', attention_mask.shape)
            inputs = inputs + attention_mask
            # print('after inputs', inputs.shape)
        probs = torch.softmax(inputs, dim=-1).to(dtype=input_dtype)
        # print('softmax probs', probs.dtype)
        return probs

class TorchReLU(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs + attention_mask    #In Softmax you add  -infty and apply softmax?
        # torch._dynamo.config.suppress_errors = True
        outputs = torch.nn.functional.relu(inputs).to(dtype=input_dtype)
        return outputs

class TorchReLU_Norm(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        if attention_mask is not None:
            inputs = inputs + attention_mask    #In Softmax you add  -infty and apply softmax?
        # torch._dynamo.config.suppress_errors = True
        outputs = torch.nn.functional.relu(inputs).to(dtype=input_dtype)
        outputs = outputs / (torch.sum(outputs, dim=-1, keepdim=True) + 1e-7)
        
        return outputs




class TorchNormalize(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        print('TorchNorm')
        print('inputs', inputs.shape)
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        if attention_mask is not None:
            print('attention_mask', attention_mask.shape)
            # print('attention_mask', attention_mask)
            inputs[attention_mask != 0] = 0
            print('after inputs', inputs.shape)

        norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms
    
class Polynorm(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, poly_type = 'sigmoid', norm_type = 2, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))
        self.poly_type = poly_type
        self.norm_type = norm_type

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        activ =  lambda x: x**2

        if self.poly_type == 'quadratic':
            activ = lambda x : x**2
        elif self.poly_type == 'cubic':
            activ = lambda x : x**3
        elif self.poly_type == 'tanh':
            activ = lambda x : x - x**3/3 + 2*x**5/15
        elif self.poly_type == 'sigmoid':
            activ = lambda x : 1/2 + x/4 - x ** 3 / 48 + x ** 5 /480 
        # elif self.poly_type == 'gelu':
        #     activ = lambda x : 

        inputs = activ(inputs)

        if attention_mask is not None:
            inputs[attention_mask != 0] = 0

        if self.norm_type == 0:
            norms = torch.nn.functional.layer_norm(inputs, inputs.shape[1:], eps=1e-05)
        elif self.norm_type == 1:
            norms = inputs / (torch.sum(inputs, dim=-1, keepdim=True) + 1e-7)
        elif self.norm_type == 2:
            norms = inputs

        norms = (norms * self.seq_gamma + self.seq_beta).to(dtype=input_dtype)
        return norms
    
class Exp(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.seq_gamma = torch.nn.Parameter(torch.ones(1, num_attention_heads, 1, 1))
        self.seq_beta = torch.nn.Parameter(torch.zeros(1, num_attention_heads, 1, 1))

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        activ =  lambda x: 10 * torch.exp(x)
        
        outputs = activ(inputs)

        if attention_mask is not None:
            inputs[attention_mask != 0] = 0
        return outputs
    
class ScaledIdentity(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs * torch.as_tensor(inputs.shape[2]).rsqrt()).to(dtype=input_dtype)


class Cumsum(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.cumsum(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)


class CumsumExp(torch.nn.Module):
    def __init__(self, seq_op_in_fp32):
        super().__init__()
        self.seq_op_in_fp32 = True  # Required as of pytorch 1.13

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        """Sequence-scaled input cumulative sum."""
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
        return (inputs.logcumsumexp(dim=-1) * pow(inputs.shape[2], -0.5)).to(dtype=input_dtype)
