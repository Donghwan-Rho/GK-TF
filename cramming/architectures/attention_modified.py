"""Attention modules. The final model uses "self-attention", but other options were tried and are still documented here."""
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention

from .embeddings import Rotary, RotarySanityCheck, RotaryEleutherAI, RotaryLLAMA
from typing import Optional
from einops.layers.torch import Rearrange
from einops import rearrange
import logging
import math
import torch.nn.functional as F
from torch.cuda.amp import autocast

log = logging.getLogger(__name__)

def subtraction_gaussian_kernel_torch(q, k):
    # print('\n========= Start subtraction_gaussian_kernel_torch =========')
    # print(f'q: {q.shape}') # b*12, 128, 64
    # print(f'q: {q.dtype}') # b*12, 128, 64
    # print(f'k: {k.shape}') # b*12, 128, 64
    k = k.transpose(-1, -2) 
    # print('in subtract q', q.shape) # 1536, 128, 64
    # print('in subtract k', k.shape) # 1536, 128, 64
    # print('q@k', (q@k).shape) # 1536, 128, 128
    # print('gau q', q.dtype)
    # [B, H, H1*W1, C] @ [C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matA_square = q ** 2. @ torch.ones(k.shape[-2:]).cuda()
    # print('matA_square', matA_square.dtype)
    # [H1*W1, C] @ [B, H, C, H2*W2] -> [B, H, H1*W1, H2*W2]
    matB_square = torch.ones(q.shape[-2:]).cuda() @ k ** 2.
    
    # print(f'matA_square: {matA_square.shape}') # b*12, 128, 64
    # print(f'matA_square: {matA_square.dtype}') # b*12, 128, 64
    # print(f'matB_square: {matB_square.shape}') # b*12, 128, 64
    # print('q**2', torch.norm(matA_square, p=float('inf')).item())
    # print('k**2', torch.norm(matB_square, p=float('inf')).item())
    # print('q@k', torch.norm((q@k), p=float('inf')).item())
    # print('========= End subtraction_gaussian_kernel_torch =========\n')
    return matA_square + matB_square - 2. * (q @ k)

def get_attention_mechanism(
    idx,
    hidden_size,
    cfg_attention,
):
    # print('cfg_attention', cfg_attention)
    # print('cfg_attention.type', cfg_attention.type)
    cfg_attention.type = cfg_attention['type']
    
    if  cfg_attention.type == "self-attention-modified":
        mechanism = SeqFirstSelfAttention_modified(hidden_size, cfg_attention)
    else:
        raise ValueError(f"Invalid attention type {cfg_attention.type} given.")
    return mechanism

class LegacySeqFirstSelfAttention_modified(torch.nn.Module):
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
        # print(f'LegacySeqFirstSelfAttention_modified cfg_attention: {cfg_attention}')
        self.hidden_size = hidden_size
        self.num_attention_heads = cfg_attention.num_attention_heads
        # 64
        self.hidden_per_head = self.hidden_size // cfg_attention.num_attention_heads
        # print('self.hidden_per_head', self.hidden_per_head)
        self.register_buffer("norm_factor", torch.tensor(self.hidden_per_head).rsqrt())

        # Strided linear layer.
        
        # # Q = K setting (SOFT paper)
        # ################################
        # # Linear(768, 768*2=1536)
        # self.query_key_value = torch.nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=cfg_attention.qkv_bias)
        # ################################
        
        # Q != K setting
        ################################
        # Linear(768, 2304)
        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=cfg_attention.qkv_bias)
        ################################
        
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

        if cfg_attention.is_train:
            if cfg_attention.train_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "torch-relu":
                self.sequence_op = TorchReLU(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "torch-relu-norm":
                self.sequence_op = TorchReLU_Norm(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "torch-norm":
                self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            # elif cfg_attention.train_sequence_op == "exp_power_app":
            #     self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "exp_poly_app":
                self.sequence_op = exp_poly_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "exp_taylor_app":
                self.sequence_op = exp_taylor_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "poly":
                self.sequence_op = Polynorm(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "none":
                self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "cumsum":
                self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.train_sequence_op == "cumsumexp":
                self.sequence_op = CumsumExp(cfg_attention.seq_op_in_fp32)
            else:
                raise ValueError(f"Invalid sequence operation {cfg_attention.sequence_op} given.")
        else:
            if cfg_attention.eval_sequence_op == "torch-softmax":
                self.sequence_op = TorchSoftmax(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "torch-relu":
                self.sequence_op = TorchReLU(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "torch-relu-norm":
                self.sequence_op = TorchReLU_Norm(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "torch-norm":
                self.sequence_op = TorchNormalize(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp":
                self.sequence_op = Exp(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp_power_app":
                self.sequence_op = exp_power_app(self.num_attention_heads, cfg_attention.exp_power_deg, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp_poly_app":
                self.sequence_op = exp_poly_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "exp_taylor_app":
                self.sequence_op = exp_taylor_app(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "poly":
                self.sequence_op = Polynorm(self.num_attention_heads, cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "none":
                self.sequence_op = ScaledIdentity(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "cumsum":
                self.sequence_op = Cumsum(cfg_attention.seq_op_in_fp32)
            elif cfg_attention.eval_sequence_op == "cumsumexp":
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
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1)) # attention score와 V 행렬곱
        # print('Legacy att context_layer', context_layer.shape)

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # with autocast():
        #     # hidden_states가 self.query_key_value 레이어에 입력되기 전의 데이터 타입을 확인합니다.
        #     print(f"Data type before self.query_key_value: {hidden_states.dtype}")

        #     # self.query_key_value 레이어를 통해 mixed_x_layer 계산
        #     mixed_x_layer = self.query_key_value(hidden_states)

        #     # mixed_x_layer의 데이터 타입을 확인합니다.
        #     print(f"Data type after self.query_key_value: {mixed_x_layer.dtype}")# print('\n================ Start Legacy forward ================')
        # =====================
        # hidden_states: [sq, b, h]
        # Query, Key, and Value
        # =====================
        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.dtype}')
        mixed_x_layer = self.query_key_value(hidden_states) # 128 128 2304
        # log.info('mixed_x_layer: {}'.format(mixed_x_layer))
        # torch._dynamo.config.verbose=True
        # torch._dynamo.config.suppress_errors = True
        # print('\n============== Legacy forward ==============')
        # print('hidden_states', hidden_states.shape) # 128 128 768
        # print('hidden_states', hidden_states.dtype) # 128 128 768
        # mixed_x_layer = mixed_x_layer.type(torch.float32)
        # print('mixed_x_layer', mixed_x_layer.shape)
        # print('mixed_x_layer', mixed_x_layer.dtype)
      
        # Q != K setting
        ################################
        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        # new_tensor_shape = mixed_x_layer.size()[:-1] + (self.num_attention_heads, 3 * self.hidden_per_head)
        mixed_x_layer = mixed_x_layer.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_attention_heads, 3 * self.hidden_per_head
        )
        # print('after view mixed_x_layer', mixed_x_layer.shape)
        # print('after view mixed_x_layer', mixed_x_layer.dtype)
        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = torch.split(mixed_x_layer, [self.hidden_per_head] * 3, dim=3)
        # print('query_layer', query_layer.shape) # 128 128 12 64
        # print('query_layer', query_layer.dtype) # 128 128 12 64
        # print('key_layer', key_layer.shape) 
        # print('value_layer', value_layer.shape) 
        ################################
        
        if self.rotary_emb is not None:
            # print('if self.rotary_emb is not None')
            query_layer, key_layer = self.rotary_emb(query_layer, key_layer)
            # print(f'query_layer: {query_layer.shape}')
        # print('query_layer', query_layer.dtype)
        # ==================================
        # Attention computation
        # ==================================
        # context_layer, matmul_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training)
        
        # Ordinary
        context_layer, matmul_result = self.attention(query_layer, key_layer, value_layer, attention_mask, self.training) # C'
        # print(f'after att context_layer: {context_layer.shape}')
        # print(f'after att context_layer: {context_layer.dtype}')
        
        # print('after att context_layer', context_layer.dtype)
        # context_layer = context_layer.type(torch.float32)
        # print('legacy forward context_layer', context_layer.shape)
        # print('Legacy forward context_layer', context_layer.shape)
        # [b, np, sq, hn] --> [sq, b, np, hn]
        # print('오류 직전')
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous() # C''
        # print(f'after permute context_layer: {context_layer.shape}')
        # print(f'after permute context_layer: {context_layer.dtype}')
        # print('after permute Legacy forward context_layer', context_layer.shape)

        # [sq, b, np, hn] --> [sq, b, hp]
        # new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(context_layer.shape[0], context_layer.shape[1], self.hidden_size) # C'''
        # print("C''' context_layer", context_layer.shape)
        # print('after view context_layer', context_layer.dtype)
        # print('============== End Legacy forward ==============\n')
        
        # Ordinary
        return context_layer, matmul_result
        # # Heatmap
        # return context_layer, matmul_result, heatmap_result

 
class SeqFirstSelfAttention_modified(LegacySeqFirstSelfAttention_modified):
    """Self-attention layer.

    This is the gpt neo-x implementation from:
    https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/transformer.py (which is a megatron variant)

    This is a modified version of the neo-x implementation that I can manage to compile without graph breaks
    """

    __constants__ = ["LAYOUT", "attention_dropout"]
    LAYOUT: str = "[S B H]"
    norm_factor: torch.Tensor

    def attention(self, query_layer, key_layer, value_layer, attention_mask: Optional[torch.Tensor] = None, training: bool = False):
        # print('\n============= Start SeqFirstSelfAttention_modified att =============')
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================
        # [b, np, sq, sk]
        # print('\n============== Seq attention ==============')
        # print('Seq attention query_layer', query_layer.shape)
        # print('Seq attention query_layer', query_layer.dtype)
        # print('Seq attention key_layer', key_layer.shape)
        # print('Seq attention key_layer', key_layer.dtype)
        # print('Seq attention value_layer', value_layer.shape)
        # print('Seq attention value_layer', value_layer.dtype)
        # print(f'attention_mask: {attention_mask}')
        
        # 32, 12, 128, 128
        output_size = (query_layer.shape[1], query_layer.shape[2], query_layer.shape[0], key_layer.shape[0])
        # print('output_size', output_size)
        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        # print('query_layer', query_layer.shape)
        # print('query_layer', query_layer.dtype)
        # print('after view Seq attention query_layer', query_layer.dtype)
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        # print('key_layer', key_layer.shape)
        # print('value_layer', value_layer.shape)
        
        #######################################
        # this better be fused in a clever way:
        # QK^T
        # print('\n============== matmul ==============')
        # print('before transpose query_layer', query_layer.shape)
        # print('after transpose query_layer', query_layer.transpose(0, 1).shape)
        # print('before transpose key_layer', key_layer.shape)
        # print('after transpose 1 key_layer', key_layer.transpose(0, 1).shape)
        # print('after transpose 2 key_layer', key_layer.transpose(0, 1).transpose(1, 2).shape)        
        # gpu 0, self.norm_factor: 0.0361
        # matmul_result = torch.bmm(query_layer.transpose(0, 1), key_layer.transpose(0, 1).transpose(1, 2)) * self.norm_factor
        # print('matmul_result.get_device()', matmul_result.get_device())
        # print('self.norm_factor', self.norm_factor)
        # print('Seq attention matmul_result', matmul_result.shape)
        # print('============== matmul end ==============\n')
        #######################################
        
        # #######################################
        # shape_0 = query_layer.shape[1] # 32
        # shape_1 = query_layer.shape[0] # 128
        # shape_2 = query_layer.shape[0] # 128
        # query_layer_ = query_layer.transpose(0, 1)
        # key_layer_ = key_layer.transpose(0, 1)
        
        # matmul_result = torch.zeros(shape_0, shape_1, shape_2)
        # matmul_result = matmul_result.to('cuda')
        
        # for b in range(shape_0):
        #     for row in range(shape_1):
        #         for col in range(shape_2):
        #             matmul_result[b][row][col] = torch.norm(query_layer_[b][row] - key_layer_[b][col]) ** 2
        
        # matmul_result *= -self.norm_factor * 0.5
        # print('matmul_result', matmul_result.shape)
        #######################################
        
        
        # # GK
        # print('\n============== matmul ==============')
        query_layer_matmul = query_layer.transpose(0, 1)
        # print('query_layer.transpose(0, 1)', query_layer.transpose(0, 1).shape)
        # print('query_layer.transpose(0, 1)', query_layer.transpose(0, 1).dtype)
        key_layer_matmul = key_layer.transpose(0, 1)
        # print('key_layer.transpose(0, 1)', key_layer.transpose(0, 1).shape)
        # print('key_layer.transpose(0, 1)', key_layer.transpose(0, 1).dtype)
        
        # query_layer_matmul = query_layer_matmul.to(dtype=torch.float)
        # key_layer_matmul = key_layer_matmul.to(dtype=torch.float)
                
        matmul_result = subtraction_gaussian_kernel_torch(query_layer_matmul, key_layer_matmul)
        matmul_result *= -self.norm_factor * 0.5
        # print('GK result', matmul_result.shape)
        # print('GK result', matmul_result.dtype)
        # matmul_result를 before_att라는 output으로 빼고 싶음
        # print('query_layer_matmul', query_layer_matmul.shape)
        # print('key_layer_matmul', key_layer_matmul.shape)      
        
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(output_size[0], output_size[1], output_size[2], output_size[3])
        # print('Before exp: attention_scores', attention_scores.shape)
        # print('Before exp: attention_scores', attention_scores.dtype)

        # ===========================
        # Attention probs and dropout
        # ===========================
        # attention scores and attention mask [b, np, sq, sk]
        # sequence: softmax 등 적용
        attention_probs = self.sequence_op(attention_scores, attention_mask)
        # print('attention_probs', torch.norm(attention_probs, p=float('inf')))
        # print('after exp attention_probs', attention_probs.shape)
        # print('after exp attention_probs', attention_probs.dtype)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # [And in great ML tradition I keep this comment in place as it was in megatron and huggingface-bert before :>]
        # attention_probs = self.attention_dropout(attention_probs)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.attention_dropout, training=training)
        # print(f'attention_probs: {attention_probs.shape}')
        # print(f'after dropout attention_probs: {attention_probs.dtype}')
        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.shape[1], value_layer.shape[2], query_layer.shape[0], value_layer.shape[3])
        # print('output_size', output_size)

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0), output_size[0] * output_size[1], -1)
        # print('seq att value_layer', value_layer.dtype)
        # print('after view value_layer', value_layer.shape)
        # print('after view value_layer', value_layer.dtype)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)
        # print('after view Seq attention attention_probs', attention_probs.shape)
        # print('after view Seq attention attention_probs', attention_probs.dtype)
        # matmul: [b * np, sq, hn]
        # att 결과와 V 곱함
        # print(f'value_layer.transpose(0, 1): {value_layer.transpose(0, 1).shape}')
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1)) # C
        # print('context_layer', context_layer.dtype)
        # print('Seq attention context_layer', context_layer.shape)
        # print('Seq attention context_layer', context_layer.dtype)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size) # C'
        # print('after view context_layer', context_layer.shape)
        # print('after view context_layer', context_layer.dtype)
        # print('============== End SeqFirstSelfAttention_modified att ==============\n')
        # print('SeqFirstSelfAttention_modified att context_layer', context_layer.shape)
        # print('SeqFirstSelfAttention_modified att matmul_result', matmul_result.shape)
        
        # Ordinary
        return context_layer, matmul_result
    
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
                
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        probs = torch.softmax(inputs, dim=-1).to(dtype=input_dtype)
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
            inputs = inputs + attention_mask
            
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
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        # 왜 오류?
        if attention_mask is not None:
            inputs[attention_mask != 0] = 0

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
            inputs = inputs + attention_mask
        
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
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        # print(f'Exp input: {inputs.shape}')
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            # print(f'After changing Exp input: {inputs.dtype}')
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)

        # print('============ Start Exp ============')
        if attention_mask is not None:
            # print(f'attention_mask: {attention_mask}')
            inputs = inputs + attention_mask
        activ =  lambda x: torch.exp(x)
        # print('inputs type', inputs.dtype)
        outputs = activ(inputs)
        # print('inputs:', inputs.dtype)
        # print('after exp:', outputs)
        # print('after exp:', outputs.shape)
        # print('after exp:', outputs.dtype)
        # print('============ End Exp ============')
        return outputs
      
class exp_power_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, exp_power_deg=9, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32
        self.exp_power_deg = exp_power_deg

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        # print('inputs', inputs.shape)
        # print('attention_mask', attention_mask.shape), 
        
        # # input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        # self.exp_power_deg = 9
                      
        activ =  lambda x: (1 + x / (2 ** self.exp_power_deg)) ** (2 ** self.exp_power_deg)       
        outputs = activ(inputs)
        
        # print('inputs:', torch.norm(inputs, p=float('inf')))
        # print('outputs:', torch.norm(outputs, p=float('inf')))
        # print('outputs:', outputs)
        # print('inputs shape', inputs.shape, 'outputs shape', outputs.shape)
        
        # print('after app:', outputs.dtype)
        # outputs = (1 + inputs / (2 ** deg))
        # for _ in range(deg):
        #     outputs *= outputs
        
        return outputs
 
class exp_taylor_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        activ =  lambda x: sum([x ** i / math.factorial(i) for i in range(15)])
            # 1 + x + x**2 / 2 + x**3 / 6 + x**4 / 24 + x**5 / 120 + x**6 / 720 + x**7 / 5040
        
        # outputs = activ(inputs)
        outputs = 1
        for i in range(1, 14):
            outputs += (inputs) ** i / math.factorial(i)

        if attention_mask is not None:
            inputs = inputs + attention_mask
            
        return outputs
   
class exp_poly_app(torch.nn.Module):
    seq_op_in_fp32: torch.jit.Final[bool]

    def __init__(self, num_attention_heads=1, seq_op_in_fp32=False):
        """Normalized attention pooling as described in Richter&Wattenhofer, 2020."""
        super().__init__()
        self.seq_op_in_fp32 = seq_op_in_fp32

    def forward(self, inputs, attention_mask: Optional[torch.Tensor] = None):
        # Inputs are [b, np, sq, sk]
        input_dtype = inputs.dtype
        if self.seq_op_in_fp32:
            inputs = inputs.to(dtype=torch.float)
            if attention_mask is not None:
                attention_mask = attention_mask.to(dtype=torch.float)
        
        # x <- x/K            
        inputs = inputs / 300
        
        activ = lambda x: 0.0000195464058*(x**7) + 0.000482184328*(x**6)\
                            + 0.00533666219*(x**5)\
                            + 0.0355159261*(x**4) + 0.159281596*(x**3)\
                            + 0.495328581*(x**2) + 0.99874163*x + 0.999917605
        outputs = activ(inputs) ** 300
        
        if attention_mask is not None:
            inputs = inputs + attention_mask
            
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
