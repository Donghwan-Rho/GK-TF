"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.
OmegaConf
Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoModelForTokenClassification

from typing import Optional
from omegaconf import OmegaConf
from termcolor import colored
import os
from .architectures_grad import ScriptableLMForPreTraining_grad, ScriptableLMForSequenceClassification_grad
from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent_modified,
    PoolingComponent,
    PoolingComponent_lora,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
    Custom_CrossEntropyLoss
)
from .attention_modified import get_attention_mechanism
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)

def construct_crammed_bert_modified(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    # print('construct_crammed_bert')
    # print('cfg_arch\n',cfg_arch)
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes
    # print(f'construct config: {config}')
    # print(f'construct cfg_arch: {cfg_arch}')

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining_modified(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification_modified(config)
    return model

def construct_crammed_bert_grad(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    # print('construct_crammed_bert')
    # print('cfg_arch\n',cfg_arch)
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes
    print(f'construct config: {config}')
    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining_grad(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification_grad(config)
    return model

class AttentionComponent_modified(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        # print(f'self.self_attention: {self.self_attention}')
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT
        # print(f'self.LAYOUT: {self.LAYOUT}')

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # Ordinary
        # print('\n================ Start Att ================')
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.dtype}')
        #######################################
        output, matmul_result = self.self_attention(hidden_states, attention_mask)
        output = self.dense(output)
        # print(f'after self.dense: {output.shape}')
        # print(f'after self.dense: {output.dtype}')
        # print('================ End Att ================\n')
        return output, matmul_result
        #######################################

class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, get_input_range, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
        self.get_input_range = get_input_range

    def forward(self, hidden_states):
        # Ordinary
        # 여기
        # print('\n========= Start FFN =========')
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.dtype}')
        # print(f'self.dense_in(hidden_states): {self.dense_in(hidden_states).shape}')
        # print(f'self.dense_in(hidden_states): {self.dense_in(hidden_states).dtype}')
        # print(f'self.nonlin(self.dense_in(hidden_states)): {self.nonlin(self.dense_in(hidden_states)).shape}')
        # print(f'self.nonlin(self.dense_in(hidden_states)): {self.nonlin(self.dense_in(hidden_states)).dtype}')
        # print(f'self.dense_out(self.nonlin(self.dense_in(hidden_states))): {self.dense_out(self.nonlin(self.dense_in(hidden_states))).shape}')
        # print(f'self.dense_out(self.nonlin(self.dense_in(hidden_states))): {self.dense_out(self.nonlin(self.dense_in(hidden_states))).dtype}')
        #######################################
        hidden_states = self.dense_in(hidden_states)
        
        # Inputs of GELU in GLU setting
        #######################################
        if self.get_input_range:
            # print('ddddddddddd')
            _, nonlin_inputs = hidden_states.chunk(2, dim=-1)
        #######################################
            
        hidden_states = self.nonlin(hidden_states)
        # print(f'hidden_states: {hidden_states.shape}')
        # print(f'hidden_states: {hidden_states.dtype}')
        # print('========= End FFN =========\n')
        if self.get_input_range:
            return self.dense_out(hidden_states), nonlin_inputs
        else:
            return self.dense_out(hidden_states)


class TransformerLayer_modified(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.idx = idx
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        if cfg_arch.norm in ["Approx_LayerNorm"]:
            if idx == 0:
                div_max_1 = 10
                # print(f'tf idx {idx} att div_max: {div_max_1}')
                div_max_2 = 4000
                # print(f'tf idx {idx} ffn div_max: {div_max_2}')
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            elif idx == 1:
                div_max_1 = 5000
                # print(f'tf idx {idx} att div_max: {div_max_1}')
                div_max_2 = 4000
                # print(f'tf idx {idx} ffn div_max: {div_max_2}')
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
            else:
                div_max_1 = 8000
                # print(f'tf idx {idx} att div_max: {div_max_1}')
                div_max_2 = 8000
                # print(f'tf idx {idx} ffn div_max: {div_max_2}')
                self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_1, eps=cfg_arch.norm_eps)
                self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, div_max=div_max_2, eps=cfg_arch.norm_eps)
                
        else:
            self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
            self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent_modified(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.cfg_arch = cfg_arch
        # print(colored(f'TransformerLayer_modified cfg_arch: {cfg_arch}', 'magenta'))
        # print(colored(f'TransformerLayer_modified cfg_arch.get_input_range: {cfg_arch.get_input_range}', 'light_magenta'))
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            self.cfg_arch.get_input_range,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )  
        
    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        # print('\n========== Start Transformer layer ==========')
        # Ordinary
        #######################################
        # print(f'states: {states.shape}')
        # print(f'states: {states.dtype}')
        # print(f'attention_mask: {attention_mask}')
        # print(f'attention_mask: {attention_mask.shape}')
        norm1_inputs = states
        # print(f'idx {self.idx} norm1_inputs max/min: {torch.max(norm1_inputs)}/{torch.min(norm1_inputs)}')
        states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        # print(f'after att states2: {states2.dtype}')
        states = states + self.dropout(states2)
        # print(f'after add & dropout: {states.dtype}')
        norm2_inputs = states
        # print(f'idx {self.idx} norm2_inputs max/min: {torch.max(norm2_inputs)}/{torch.min(norm2_inputs)}')
        # print(f'self.norm2(states): {self.norm2(states).dtype}')
        if self.cfg_arch.get_input_range:
            states2, nonlin_inputs = self.ffn(self.norm2(states))
        else:
            states2 = self.ffn(self.norm2(states))
        states = states + self.dropout(states2)
        # print(f'after norm2, ffn, dropout states: {states.dtype}')
        # print('========== End Transformer layer ==========\n')
        if self.cfg_arch.get_input_range:
            return states, matmul_result, norm1_inputs, norm2_inputs, nonlin_inputs
        else:
            return states, matmul_result

class ScriptableLM_modified(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        # print(colored(f'ScriptableLM_modified self.cfg: {self.cfg}', 'magenta'))
        # print(colored(f'ScriptableLM_modified self.cfg.get_input_range: {self.cfg.get_input_range}', 'light_magenta'))
        self.cfg.embedding.get_emb_input_range = self.cfg.get_input_range
        # print(colored(f'ScriptableLM_modified self.cfg.embedding.get_emb_input_range: {self.cfg.embedding.get_emb_input_range}', 'light_magenta'))

        self.embedding = EmbeddingComponent_modified(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer_modified(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
         
        self.seq_first = True
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            if self.cfg.norm in ["Approx_LayerNorm"]:
                div_max = 8000
                # print(f'final norm div_max: {div_max}')
                self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, div_max=div_max, eps=self.cfg.norm_eps)
            else:
                self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
            # print(f'eps=self.cfg.norm_eps: {self.cfg.norm_eps}') # 1e-12
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        # for name, param in self.named_parameters():
        #     print(f"Parameter name: {name} - dtype: {param.dtype}")
        matmuls = []
        tf_norm1_inputs_list = []
        tf_norm2_inputs_list = []
        nonlin_inputs_list = []
        # print('\n========== ScriptableLM_modified Start ==========')
        # print('input_ids', input_ids.shape)
        # print('input_ids', input_ids.dtype)
        # print('attention_mask', attention_mask.dtype) # attention_mask = None
         
        # print(f'attention_mask: {attention_mask}')
        if attention_mask is not None:
            # print(f'if attention_mask is not None attention_mask: {attention_mask}')
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
            # print(f'extended attention_mask: {attention_mask.dtype}')
       
        # Ordinary
        #######################################
        if self.cfg.embedding.get_emb_input_range:
            hidden_states, emb_norm_inputs = self.embedding(input_ids)
        else:
            hidden_states = self.embedding(input_ids)
        # print(f'after embedding: {hidden_states.dtype}')
        
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            # print('if self.seq_first')
            # print(f'hidden_states = hidden_states.transpose(0, 1).contiguous(): {hidden_states.shape}')
        
        for i, layer_module in enumerate(self.layers):
            # Ordinary
            #######################################
            if self.cfg.get_input_range:
                # print(f'Layer {i}')
                hidden_states, matmul, tf_norm1_inputs, tf_norm2_inputs, nonlin_inputs = layer_module(hidden_states, attention_mask)
                tf_norm1_inputs_list.append(tf_norm1_inputs)
                tf_norm2_inputs_list.append(tf_norm2_inputs)
                nonlin_inputs_list.append(nonlin_inputs)
            else:
                hidden_states, matmul = layer_module(hidden_states, attention_mask)
            # print(f'after transformer hidden_states: {hidden_states.dtype}')
            # print(f'after transformer matmul: {matmul.dtype}')
            
            matmuls.append(matmul)
            
        # print(f'after transformer layers hidden_states: {hidden_states.shape}')
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            # print('if self.seq_first')
            # print(f'hidden_states.transpose(0, 1).contiguous(): {hidden_states.dtype}')

        # Ordinary
        #######################################
        # print('\n========== ScriptableLM_modified End ==========')
        if self.cfg.get_input_range:
            final_norm_inputs = hidden_states
            # print(f'self.final_norm(hidden_states): {self.final_norm(hidden_states).dtype}')
            return self.final_norm(hidden_states), matmuls, emb_norm_inputs, tf_norm1_inputs_list, tf_norm2_inputs_list, final_norm_inputs, nonlin_inputs_list
        else:
            return self.final_norm(hidden_states), matmuls
        
class ScriptableLMForPreTraining_modified(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        
        # print(colored(f'ScriptableLMForPreTraining_modified self.cfg.get_input_range: {self.cfg.get_input_range}', 'light_green'))
        # print(colored(f'ScriptableLMForPreTraining_modified self.cfg.embedding.get_emb_input_range: {self.cfg.embedding.get_emb_input_range}', 'light_green'))

        self.encoder = ScriptableLM_modified(config)

        if not self.cfg.skip_head_transform:
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            # 여기
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        # print('self.decoder', self.decoder)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()
        
        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_graph_interval_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        # self.emb_norm_inputs_var_norms = []
        self.emb_norm_inputs_var_maxs = []
        self.emb_norm_inputs_var_mins = []
        self.emb_norm_inputs_var_ratios = []
        # self.tf_norm1_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.tf_norm2_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.final_norm_inputs_var_norms = []
        self.final_norm_inputs_var_maxs = []
        self.final_norm_inputs_var_mins = []
        self.final_norm_inputs_var_ratios = []
        self.nonlin_inputs_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.nonlin_inputs_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        if self.cfg.get_input_range:
            os.makedirs('norms')
        os.makedirs('loss')
        os.makedirs('after_norm_penalty')
        square_layer = math.floor(math.sqrt(self.cfg.num_transformer_layers))
        if square_layer ** 2 >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer
        elif square_layer * (square_layer+1) >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer + 1
        else:
            self.vertical_num = square_layer + 1
            self.horizontal_num = square_layer + 1
        
    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        # print(f'self.cfg.norm_penalty_coeff: {self.cfg.norm_penalty_coeff}')
        matmul_sup_list = []
        before_att_var_max_list = []
        before_FFN_var_max_list = []
        before_att_var_ratio_list = []
        before_FFN_var_ratio_list = []
        # print('self', self)
        self.count += 1
        self.x_list.append(self.count)
        # print('\n')
        # Ordinary
        #######################################
        if self.cfg.get_input_range:
            outputs, matmuls_from_enc, emb_norm_inputs, tf_norm1_inputs, tf_norm2_inputs, final_norm_inputs, nonlin_inputs = self.encoder(input_ids, attention_mask)
            # print(f'emb_norm_inputs: {emb_norm_inputs.shape}')
            # print(f'tf_norm1_inputs: {tf_norm1_inputs.shape}')
            # print(f'tf_norm2_inputs: {tf_norm2_inputs.shape}')
            # print(f'final_norm_inputs: {final_norm_inputs.shape}')
            # print(f'nonlin_inputs: {nonlin_inputs}')
            
            # print(f'\nCount: {self.count}')
            # print(f'========== Range of Variances of Embedding {self.cfg.norm} Inputs ==========')
            # emb_norm_inputs_norm = torch.norm(emb_norm_inputs, p=float('inf'))
            
            # Embedding Norm Input Variances
            mean = emb_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((emb_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            emb_var_max = torch.max(var)
            emb_var_min = torch.min(var)
            emb_var_ratio = emb_var_max / emb_var_min
            # print(f'Sup Norm of Embedding Layernorm: {emb_norm_inputs_norm.item()}')
            # self.emb_norm_inputs_var_norms.append(emb_norm_inputs_var_norm.item())
            self.emb_norm_inputs_var_maxs.append(emb_var_max.item())
            self.emb_norm_inputs_var_mins.append(emb_var_min.item())
            self.emb_norm_inputs_var_ratios.append(emb_var_ratio.item())
            # print(f'Max of Variances: {max(self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])}')
            # print(f'Min of Variances: {min(self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])}')
            # print(f'========== Range of Variances of {self.cfg.norm} 1 Inputs ==========')
            
            for i in range(self.cfg.num_transformer_layers):
                # Input Variances of Norm Before Attention
                # tf_norm1_inputs_norm = torch.norm(tf_norm1_inputs[i], p=float('inf'))
                mean = tf_norm1_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm1_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var)
                var_min = torch.min(var)
                var_ratio = var_max / var_min
                before_att_var_max_list.append(var_max)
                before_att_var_ratio_list.append(var_ratio)
                self.tf_norm1_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm1_inputs_var_mins[i].append(var_min.item())
                self.tf_norm1_inputs_var_ratios[i].append(var_ratio.item())
                
                # Input Variances of Norm Before FFN
                # print(f'========== Range of Variances of {self.cfg.norm} 2 Inputs ==========')
                # tf_norm2_inputs_norm = torch.norm(tf_norm2_inputs[i], p=float('inf'))
                mean = tf_norm2_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm2_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var)
                var_min = torch.min(var)
                var_ratio = var_max / var_min
                before_FFN_var_max_list.append(var_max)
                before_FFN_var_ratio_list.append(var_ratio)
                # print(f'Sup Norm of Layernorm 2 of Layer {i}: {tf_norm2_inputs_norm.item()}')
                # self.tf_norm2_inputs_norms[i].append(tf_norm2_inputs_norm.item())
                self.tf_norm2_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm2_inputs_var_mins[i].append(var_min.item())
                self.tf_norm2_inputs_var_ratios[i].append(var_ratio.item())
                # print(f'Layer {i}, Max of {self.cfg.norm} Before FFN Variances: {max(self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Layer {i}, Min of {self.cfg.norm} Before FFN Variances: {min(self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])}')
            
                # Inputs of Non-linear function
                # print(f'========== Range of {self.cfg.nonlin} Inputs ==========')
                nonlin_inputs_max = torch.max(nonlin_inputs[i]).detach().cpu()
                nonlin_inputs_min = torch.min(nonlin_inputs[i]).detach().cpu()
                self.nonlin_inputs_maxs[i].append(nonlin_inputs_max.item())
                self.nonlin_inputs_mins[i].append(nonlin_inputs_min.item())
                # print(f'Max of {self.cfg.nonlin} Inputs: {max(self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Min of {self.cfg.nonlin} Inputs: {min(self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])}')
            
                # Inputs of exp
                matmul_max = torch.max(matmuls_from_enc[i]).detach().cpu()
                matmul_min = torch.min(matmuls_from_enc[i]).detach().cpu()
                matmul_sup_list.append(-matmul_min)
                self.matmul_norm_maxs[i].append(matmul_max.item())
                self.matmul_norm_mins[i].append(matmul_min.item())
                # print(f'Max of Inputs of exp of Layer {i}: {max(self.matmul_norm_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Min of Inputs of exp of Layer {i}: {min(self.matmul_norm_mins[i][-self.cfg.graph_interval:])}')
            
            # Inputs of Final Norm        
            # print(f'========== Range of Variances of Final {self.cfg.norm} Inputs ==========')
            # final_norm_inputs_norm = torch.norm(final_norm_inputs, p=float('inf'))
            mean = final_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((final_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            final_var_max = torch.max(var)
            final_var_min = torch.min(var)
            final_var_ratio = final_var_max / final_var_min
            # print(f'Sup Norm of Final Layernorm: {final_norm_inputs_norm.item()}')
            # self.final_norm_inputs_norms.append(final_norm_inputs_norm.item())
            self.final_norm_inputs_var_maxs.append(final_var_max.item())
            self.final_norm_inputs_var_mins.append(final_var_min.item())
            self.final_norm_inputs_var_ratios.append(final_var_ratio.item())
            # print(f'Max of Final {self.cfg.norm} Variances: {max(self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])}')
            # print(f'Min of Final {self.cfg.norm} Variances: {min(self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])}')
           
            if (self.count % self.cfg.graph_interval == 0) or (self.count  == self.cfg.full_steps): 
                # Embedding Variance
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])
                plt.title(f'Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])
                plt.title(f'Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-self.cfg.graph_interval:], self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'norms/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])}\n\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                
                # Before Attention
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'norms/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i][-self.cfg.graph_interval:])}\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
        
                # Before FFN
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-self.cfg.graph_interval:], self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'norms/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])}\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i][-self.cfg.graph_interval:])}\n')
        
                # Final Normalization
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])
                plt.title(f'Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'norms/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])
                plt.title(f'Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'norms/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-self.cfg.graph_interval:], self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])
                # plt.title(f'Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'norms/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'norms/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])}\n\n')
                # with open(f'norms/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios[-self.cfg.graph_interval:])}\n\n')
                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])
                    plt.title(f'Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-self.cfg.graph_interval:], self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])
                    plt.title(f'Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'norms/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'norms/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])}\n')
            
            # Graph after norm-penalty steps
            if self.count  == self.cfg.full_steps and self.cfg.full_steps > self.cfg.penalty_step: 
                last_graph_steps = self.cfg.full_steps - self.cfg.penalty_step
                print(f'last_graph_steps: {last_graph_steps}')
                # Embedding Variance
                plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_maxs[-last_graph_steps:])
                plt.title(f'After Norm-penalty Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_mins[-last_graph_steps:])
                plt.title(f'After Norm-penalty Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-last_graph_steps:], self.emb_norm_inputs_var_ratios[-last_graph_steps:])
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'after_norm_penalty/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs[-last_graph_steps:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins[-last_graph_steps:])}\n\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                
                # Before Attention
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-last_graph_steps:], self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])
                #     plt.title(f'After Norm-penalty Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'after_norm_penalty/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i][-last_graph_steps:])}\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i][-last_graph_steps:])}\n')
        
                # Before FFN
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list[-last_graph_steps:], self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])
                #     plt.title(f'After Norm-penalty Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'after_norm_penalty/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i][-last_graph_steps:])}\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'After Norm-penaltyRatio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i][-last_graph_steps:])}\n')
        
                # Final Normalization
                plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_maxs[-last_graph_steps:])
                plt.title(f'After Norm-penalty Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'after_norm_penalty/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_mins[-last_graph_steps:])
                plt.title(f'After Norm-penalty Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'after_norm_penalty/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list[-last_graph_steps:], self.final_norm_inputs_var_ratios[-last_graph_steps:])
                # plt.title(f'After Norm-penalty Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'after_norm_penalty/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'after_norm_penalty/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs[-last_graph_steps:])}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins[-last_graph_steps:])}\n\n')
                # with open(f'after_norm_penalty/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'After Norm-penalty Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios[-last_graph_steps:])}\n\n')
                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.nonlin_inputs_maxs[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list[-last_graph_steps:], self.nonlin_inputs_mins[i][-last_graph_steps:])
                    plt.title(f'After Norm-penalty Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'after_norm_penalty/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'after_norm_penalty/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'After Norm-penalty Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i][-last_graph_steps:])}\n')
        
        else:
            outputs, matmuls_from_enc = self.encoder(input_ids, attention_mask)
                
        outputs = outputs.view(-1, outputs.shape[-1]) # 여기
        # print(f'after view outputs: {outputs.dtype}')

        if self.sparse_prediction and labels is not None:            
            masked_lm_loss = self._forward_sparse(outputs, labels) ### loss 여기
            original_loss = masked_lm_loss.item()
            # print(f'masked_lm_loss: {masked_lm_loss.dtype}') # tensor 1개
            # print(f'masked_lm_loss: {masked_lm_loss}')
            
            self.loss_list.append(original_loss)
            
            # print(f'Matmul Range Penalty: {self.cfg.matmul_range_penalty}')
            
            # print('========== Supremum of Inputs of Exponential ==========')
            # Loss 추가
            #####################################
            # print(f'self.cfg.penalty_step: {self.cfg.penalty_step}')
            # print(f'self.count: {self.count}')
            if self.count > self.cfg.penalty_step:
                # print('if self.count > self.cfg.penalty_step:')
                if self.cfg.var_ratio_penalty:
                    # print('if self.cfg.var_ratio_penalty:')
                    if emb_var_ratio > self.cfg.var_ratio_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * emb_var_ratio
                    if final_var_ratio > self.cfg.var_ratio_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * final_var_ratio
                    for i in range(self.cfg.num_transformer_layers):
                        if before_att_var_ratio_list[i].item() > self.cfg.var_ratio_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_att_var_ratio_list[i]
                        if before_FFN_var_ratio_list[i].item() > self.cfg.var_ratio_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_FFN_var_ratio_list[i]
                    
                if self.cfg.matmul_range_penalty:
                    for i in range(self.cfg.num_transformer_layers):
                        if matmul_sup_list[i].item() > self.cfg.matmul_norm_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * matmul_sup_list[i]
                
                if self.cfg.max_var_penalty:
                    print('if self.cfg.max_var_penalty:')
                    if emb_var_max.item() > self.cfg.var_max_penalty_scale:
                       masked_lm_loss += self.cfg.norm_penalty_coeff * emb_var_max
                    for i in range(self.cfg.num_transformer_layers):
                        if before_att_var_max_list[i].item() > self.cfg.var_max_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_att_var_max_list[i]
                    for i in range(self.cfg.num_transformer_layers):
                        if before_FFN_var_max_list[i].item() > self.cfg.var_max_penalty_scale:
                            masked_lm_loss += self.cfg.norm_penalty_coeff * before_FFN_var_max_list[i]
                    if final_var_max.item() > self.cfg.var_max_penalty_scale:
                        masked_lm_loss += self.cfg.norm_penalty_coeff * final_var_max
            #####################################
                            
            if self.count < self.cfg.graph_interval:
                last_graph_interval_loss = sum(self.loss_list) / len(self.loss_list)
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            else:
                last_graph_interval_loss = sum(self.loss_list[-self.cfg.graph_interval :]) / len(self.loss_list[-self.cfg.graph_interval :])
                self.last_graph_interval_loss_list.append(last_graph_interval_loss)
                if self.best_loss == 0 or last_graph_interval_loss < self.best_loss:
                    self.best_loss = last_graph_interval_loss
                print(f'Loss: {original_loss}, Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}, Best_{self.cfg.graph_interval}_losses: {self.best_loss}, Layers: {self.cfg.num_transformer_layers}, Count: {self.count}')
            
            # Loss and Inputs of Exponential                
            if (self.count % self.cfg.graph_interval == 0) or (self.count  == self.cfg.full_steps):
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.loss_list[-self.cfg.graph_interval:])
                plt.title('Loss', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig('loss/losses.png')
                plt.clf()
                plt.plot(self.x_list[-self.cfg.graph_interval:], self.last_graph_interval_loss_list[-self.cfg.graph_interval:])
                plt.title(f'Last {self.cfg.graph_interval} losses', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig(f'loss/last_{self.cfg.graph_interval}_losses.png')
                plt.clf()
                if self.cfg.get_input_range:
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-self.cfg.graph_interval:], self.matmul_norm_maxs[i][-self.cfg.graph_interval:])
                        plt.title(f'Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'norms/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-self.cfg.graph_interval:], self.matmul_norm_mins[i][-self.cfg.graph_interval:])
                        plt.title(f'Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'norms/min_of_inputs_of_exp.png')
                    plt.clf()
                    with open(f'norms/inputs_of_exp.txt', 'w') as file:
                        file.write(f'Inputs of exp\n\n')
                        file.write(f'Max\n\n')
                        for i in range(self.cfg.num_transformer_layers):
                            file.write(f'{max(self.matmul_norm_maxs[i][-self.cfg.graph_interval:])}\n')
                        file.write('\n')
                        file.write(f'Min\n\n')
                        for i in range(self.cfg.num_transformer_layers):
                            file.write(f'{min(self.matmul_norm_mins[i][-self.cfg.graph_interval:])}\n')
        
            # Graph after norm-penalty steps                
            if self.count == self.cfg.full_steps and self.cfg.full_steps > self.cfg.penalty_step:
                plt.plot(self.x_list[-last_graph_steps:], self.loss_list[-last_graph_steps:])
                plt.title('After Norm-Penalty Loss', fontsize=10)
                plt.xlabel('Steps', fontsize=10)
                plt.ylabel('Loss', fontsize=10)
                plt.savefig('after_norm_penalty/losses.png')
                plt.clf()
                if self.cfg.get_input_range:
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-last_graph_steps:], self.matmul_norm_maxs[i][-last_graph_steps:])
                        plt.title(f'After Norm-penalty Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'after_norm_penalty/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list[-last_graph_steps:], self.matmul_norm_mins[i][-last_graph_steps:])
                        plt.title(f'After Norm-penalty Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'after_norm_penalty/min_of_inputs_of_exp.png')
                    plt.clf()
                with open(f'after_norm_penalty/inputs_of_exp.txt', 'w') as file:
                    file.write(f'After Norm-penalty Inputs of exp\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.matmul_norm_maxs[i][-last_graph_steps:])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.matmul_norm_mins[i][-last_graph_steps:])}\n')
                with open(f'results.txt', 'w') as file:
                    file.write(f'Loss: {original_loss}\n\n')
                    file.write(f'Last_{self.cfg.graph_interval}_losses: {last_graph_interval_loss}\n\n')
                    file.write(f'Best_{self.cfg.graph_interval}_losses: {self.best_loss}\n\n')
                    file.write(f'Layers: {self.cfg.num_transformer_layers}\n\n')
                    file.write(f'Count: {self.count}\n\n')
                    
        
            
        else:
            outputs = self.decoder(self.prediction_head(outputs))
            if labels is not None:
                masked_lm_loss = self.loss_fn(outputs, labels.view(-1))
            else:
                masked_lm_loss = outputs.new_zeros((1,))

        return {"loss": masked_lm_loss, "outputs": outputs}

    # Sparse prediction usually has an unpredictable number of entries in each batch
    # but the dataloader was modified so that 25% of the batch is ALWAYS masked.
    # This allows for static compilation. If you modify the dataloader, this function will fill your compile cache
    def _forward_sparse(self, outputs: torch.Tensor, labels: Optional[torch.Tensor] = None):
        # print('\n=========== Start _forward_sparse ===========')
        # print(f'outputs: {outputs.shape}')
        # print(f'lalbes: {labels}')
        # print(f'lalbes: {labels.shape}')
        labels = labels.view(-1)
        # print(f'after view labels: {labels.shape}')
        # print(f'self.loss_fn.ignore_index: {self.loss_fn.ignore_index}') # int
        # print(f'self.loss_fn.ignore_index: {self.loss_fn.ignore_index.shape}')
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        # print(f'mask_positions: {mask_positions}')
        # print(f'round(self.sparse_prediction * labels.shape[0]): round({self.sparse_prediction} * {labels.shape[0]})')
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        # print(f'num_masks_guaranteed: {num_masks_guaranteed}')
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh
        # print(f'torch.argsort(mask_positions.int()): {torch.argsort(mask_positions.int())}')
        # print(f'indices: {indices}')
        # print(f'indices: {indices.shape}')
        '''
        torch.argsort(): 작은 것들의 idx부터 큰 것들의 idx 순으로 씀
        a = torch.randn(4, 4)
        a
        tensor([[ 0.0785,  1.5267, -0.8521,  0.4065],
                [ 0.1598,  0.0788, -0.0745, -1.2700],
                [ 1.2208,  1.0722, -0.7064,  1.2564],
                [ 0.0669, -0.2318, -0.8229, -0.9280]])

        torch.argsort(a, dim=1)
        tensor([[2, 0, 3, 1],
                [3, 2, 1, 0],
                [2, 1, 0, 3],
                [3, 2, 1, 0]])
        '''
        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        # print(f'outputs: {outputs.shape}')
        labels = labels[indices]
        # print(f'labels: {labels.shape}')
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs)) # Prediction_head: Identity()
        # print(f'outputs: {outputs.shape}')
        # print('493 outputs', outputs.shape)
        # print('493 labels', labels.shape)
        masked_lm_loss = self.loss_fn(outputs, labels)
        # print(f'masked_lm_loss: {masked_lm_loss.shape}')
        # print('=========== End _forward_sparse ===========\n')
        return masked_lm_loss

class ScriptableLMForSequenceClassification_modified(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels
        
        # print('dddddddddddd self.cfg', self.cfg)
        # print(f'self.cfg.get_input_range: {self.cfg.get_input_range}')
        self.cfg.classification_head.get_input_range = self.cfg.get_input_range
        if not self.cfg.get_grad == None:
            self.cfg.classification_head.get_grad = self.cfg.get_grad

        self.encoder = ScriptableLM_modified(config)
        # self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.pooler = PoolingComponent_lora(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_graph_interval_loss_list = []
        # self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.matmul_norm_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        self.emb_norm_inputs_var_maxs = []
        self.emb_norm_inputs_var_mins = []
        self.emb_norm_inputs_var_ratios = []
        # self.tf_norm1_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.tf_norm2_inputs_var_norms = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm1_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_mins = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.tf_norm2_inputs_var_ratios = [[] for _ in range(self.cfg.num_transformer_layers)]
        # self.final_norm_inputs_var_norms = []
        self.final_norm_inputs_var_maxs = []
        self.final_norm_inputs_var_mins = []
        self.final_norm_inputs_var_ratios = []
        self.nonlin_inputs_maxs = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.nonlin_inputs_mins = [[] for _ in range(self.cfg.num_transformer_layers)]       
        self.act_ftn_inputs_maxs = []
        self.act_ftn_inputs_mins = []
        
        os.makedirs(self.cfg.task_name, exist_ok=True)
        self.act_ftn_path = os.path.join(self.cfg.task_name, 'activation_ftn')
        os.makedirs(self.act_ftn_path, exist_ok=True)
        if self.cfg.get_input_range:
            self.norm_path = os.path.join(self.cfg.task_name, 'norms')
            os.makedirs(self.norm_path, exist_ok=True)
        self.loss_path = os.path.join(self.cfg.task_name, 'loss')
        os.makedirs(self.loss_path, exist_ok=True)
        
        # print(f'self.cfg.num_transformer_layers: {self.cfg.num_transformer_layers}')
        square_layer = math.floor(math.sqrt(self.cfg.num_transformer_layers))        
        if square_layer ** 2 >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer
        elif square_layer * (square_layer+1) >= self.cfg.num_transformer_layers:
            self.vertical_num = square_layer
            self.horizontal_num = square_layer + 1
        else:
            self.vertical_num = square_layer + 1
            self.horizontal_num = square_layer + 1

    def _init_weights(self, module=None):
        modules = self.modules() if module is None else [module]
        for module in modules:
            _init_module(
                module,
                self.cfg.init.type,
                self.cfg.init.std,
                self.cfg.hidden_size,
                self.cfg.num_transformer_layers,
            )

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, **kwargs):
        matmul_sup_list = []
        before_att_var_max_list = []
        before_FFN_var_max_list = []
        before_att_var_ratio_list = []
        before_FFN_var_ratio_list = []
              
        self.count += 1
        # print(f'Count: {self.count}')
        self.x_list.append(self.count)  
        
        # print(colored('ScriptableLMForSequenceClassification_modified', 'yellow'))
        if self.cfg.get_input_range:
            encoder_output,  matmuls_from_enc, emb_norm_inputs, tf_norm1_inputs, tf_norm2_inputs, final_norm_inputs, nonlin_inputs = self.encoder(input_ids, attention_mask)
        else:
            encoder_output,  matmuls_from_enc = self.encoder(input_ids, attention_mask)
        # print(f'encoder_output: {encoder_output.shape}') # b, seq_len, 768
        # print(f'encoder_output: {encoder_output.dtype}') # b, seq_len, 768
        # print('encoder_output', encoder_output)
        if self.cfg.get_input_range:
            pooler_output, act_ftn_inputs = self.pooler(encoder_output)
        elif self.cfg.get_grad:
            pooler_output, before_zero_indexing_hidden_states, first_token_tensor = self.pooler(encoder_output)
        else:
            pooler_output = self.pooler(encoder_output)
        # print(f'pooler_output: {pooler_output.dtype}')
        logits = self.head(pooler_output)
        # print(f'logits: {logits.dtype}')
        
        if self.cfg.get_input_range:
            # print(f'emb_norm_inputs: {emb_norm_inputs.shape}')
            # print(f'tf_norm1_inputs: {tf_norm1_inputs.shape}')
            # print(f'tf_norm2_inputs: {tf_norm2_inputs.shape}')
            # print(f'final_norm_inputs: {final_norm_inputs.shape}')
            # print(f'nonlin_inputs: {nonlin_inputs}')
            
            # print(f'\nCount: {self.count}')
            # print(f'========== Range of Variances of Embedding {self.cfg.norm} Inputs ==========')
            # emb_norm_inputs_norm = torch.norm(emb_norm_inputs, p=float('inf'))
            
            # Embedding Norm Input Variances
            mean = emb_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((emb_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            emb_var_max = torch.max(var).detach().cpu()
            emb_var_min = torch.min(var).detach().cpu()
            emb_var_ratio = emb_var_max / emb_var_min
            # print(f'Sup Norm of Embedding Layernorm: {emb_norm_inputs_norm.item()}')
            # self.emb_norm_inputs_var_norms.append(emb_norm_inputs_var_norm.item())
            self.emb_norm_inputs_var_maxs.append(emb_var_max.item())
            self.emb_norm_inputs_var_mins.append(emb_var_min.item())
            self.emb_norm_inputs_var_ratios.append(emb_var_ratio.item())
            # print(f'Max of Variances: {max(self.emb_norm_inputs_var_maxs[-self.cfg.graph_interval:])}')
            # print(f'Min of Variances: {min(self.emb_norm_inputs_var_mins[-self.cfg.graph_interval:])}')
            # print(f'========== Range of Variances of {self.cfg.norm} 1 Inputs ==========')
            
            for i in range(self.cfg.num_transformer_layers):
                # Input Variances of Norm Before Attention
                # tf_norm1_inputs_norm = torch.norm(tf_norm1_inputs[i], p=float('inf'))
                mean = tf_norm1_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm1_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var).detach().cpu()
                var_min = torch.min(var).detach().cpu()
                var_ratio = var_max / var_min
                before_att_var_max_list.append(var_max)
                before_att_var_ratio_list.append(var_ratio)
                self.tf_norm1_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm1_inputs_var_mins[i].append(var_min.item())
                self.tf_norm1_inputs_var_ratios[i].append(var_ratio.item())
                
                # Input Variances of Norm Before FFN
                # print(f'========== Range of Variances of {self.cfg.norm} 2 Inputs ==========')
                # tf_norm2_inputs_norm = torch.norm(tf_norm2_inputs[i], p=float('inf'))
                mean = tf_norm2_inputs[i].mean(dim=-1, keepdim=True)
                var = ((tf_norm2_inputs[i] - mean) ** 2).mean(dim=-1, keepdim=True)
                var_max = torch.max(var).detach().cpu()
                var_min = torch.min(var).detach().cpu()
                var_ratio = var_max / var_min
                before_FFN_var_max_list.append(var_max)
                before_FFN_var_ratio_list.append(var_ratio)
                # print(f'Sup Norm of Layernorm 2 of Layer {i}: {tf_norm2_inputs_norm.item()}')
                # self.tf_norm2_inputs_norms[i].append(tf_norm2_inputs_norm.item())
                self.tf_norm2_inputs_var_maxs[i].append(var_max.item())
                self.tf_norm2_inputs_var_mins[i].append(var_min.item())
                self.tf_norm2_inputs_var_ratios[i].append(var_ratio.item())
                # print(f'Layer {i}, Max of {self.cfg.norm} Before FFN Variances: {max(self.tf_norm2_inputs_var_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Layer {i}, Min of {self.cfg.norm} Before FFN Variances: {min(self.tf_norm2_inputs_var_mins[i][-self.cfg.graph_interval:])}')
            
                # Inputs of Non-linear function
                # print(f'========== Range of {self.cfg.nonlin} Inputs ==========')
                nonlin_inputs_max = torch.max(nonlin_inputs[i]).detach().cpu()
                nonlin_inputs_min = torch.min(nonlin_inputs[i]).detach().cpu()
                self.nonlin_inputs_maxs[i].append(nonlin_inputs_max.item())
                self.nonlin_inputs_mins[i].append(nonlin_inputs_min.item())
                # print(f'Max of {self.cfg.nonlin} Inputs: {max(self.nonlin_inputs_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Min of {self.cfg.nonlin} Inputs: {min(self.nonlin_inputs_mins[i][-self.cfg.graph_interval:])}')
            
                # Inputs of exp
                matmul_max = torch.max(matmuls_from_enc[i]).detach().cpu()
                matmul_min = torch.min(matmuls_from_enc[i])
                matmul_sup_list.append(-matmul_min)
                self.matmul_norm_maxs[i].append(matmul_max.item())
                self.matmul_norm_mins[i].append(matmul_min.item())
                # print(f'Max of Inputs of exp of Layer {i}: {max(self.matmul_norm_maxs[i][-self.cfg.graph_interval:])}')
                # print(f'Min of Inputs of exp of Layer {i}: {min(self.matmul_norm_mins[i][-self.cfg.graph_interval:])}')
            
            # Inputs of Final Norm        
            # print(f'========== Range of Variances of Final {self.cfg.norm} Inputs ==========')
            # final_norm_inputs_norm = torch.norm(final_norm_inputs, p=float('inf'))
            mean = final_norm_inputs.mean(dim=-1, keepdim=True)
            var = ((final_norm_inputs - mean) ** 2).mean(dim=-1, keepdim=True)
            final_var_max = torch.max(var).detach().cpu()
            final_var_min = torch.min(var).detach().cpu()
            final_var_ratio = final_var_max / final_var_min
            # print(f'Sup Norm of Final Layernorm: {final_norm_inputs_norm.item()}')
            # self.final_norm_inputs_norms.append(final_norm_inputs_norm.item())
            self.final_norm_inputs_var_maxs.append(final_var_max.item())
            self.final_norm_inputs_var_mins.append(final_var_min.item())
            self.final_norm_inputs_var_ratios.append(final_var_ratio.item())
            # print(f'Max of Final {self.cfg.norm} Variances: {max(self.final_norm_inputs_var_maxs[-self.cfg.graph_interval:])}')
            # print(f'Min of Final {self.cfg.norm} Variances: {min(self.final_norm_inputs_var_mins[-self.cfg.graph_interval:])}')
            
            if self.count % self.cfg.eval_graph_interval == 0:
                plt.plot(self.x_list, self.emb_norm_inputs_var_maxs)
                plt.title(f'Max of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list, self.emb_norm_inputs_var_mins)
                plt.title(f'Min of Variances of Embedding {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_emb_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list, self.emb_norm_inputs_var_ratios)
                # plt.title(f'Ratio of Min/Max Variances of Embedding {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Max/Min')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'{self.norm_path}/ratio_of_variances_of_emb_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.emb_norm_inputs_var_maxs)}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.emb_norm_inputs_var_mins)}\n\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_emb_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} in Embedding\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.emb_norm_inputs_var_ratios)}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.emb_norm_inputs_var_ratios)}\n\n')
                
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm1_inputs_var_maxs[i])
                    plt.title(f'Max of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm1_inputs_var_mins[i])
                    plt.title(f'Min of Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_{self.cfg.norm}_before_attention.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list, self.tf_norm1_inputs_var_ratios[i])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before Att of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'{self.norm_path}/ratio_of_of_variances_of_{self.cfg.norm}_before_attention.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm1_inputs_var_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm1_inputs_var_mins[i])}\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_attention.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before Attention\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm1_inputs_var_ratios[i])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm1_inputs_var_ratios[i])}\n')
        
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm2_inputs_var_maxs[i])
                    plt.title(f'Max of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.tf_norm2_inputs_var_mins[i])
                    plt.title(f'Min of Variances of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min Variance', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_{self.cfg.norm}_Before FFN.png')
                plt.clf()
                # for i in range(self.cfg.num_transformer_layers):
                #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                #     plt.plot(self.x_list, self.tf_norm2_inputs_var_ratios[i])
                #     plt.title(f'Ratio of Max/Min Vars of {self.cfg.norm} Before FFN of Layer {i}', fontsize=5)
                #     plt.xlabel('Steps', fontsize=5)
                #     plt.ylabel('Max/Min', fontsize=5)
                #     plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                #     plt.tick_params(axis='both', which='major', labelsize=5)
                # plt.savefig(f'{self.norm_path}/ratio_of_of_variances_of_{self.cfg.norm}_before_ffn.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                    file.write(f'Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.tf_norm2_inputs_var_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.tf_norm2_inputs_var_mins[i])}\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_{self.cfg.norm}_before_ffn.txt', 'w') as file:
                #     file.write(f'Ratio of Max/Min Variances of Inputs of {self.cfg.norm} Before FFN\n\n')
                #     file.write(f'Max of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{max(self.tf_norm2_inputs_var_ratios[i])}\n')
                #     file.write(f'\nMin of Max/Min\n\n')
                #     for i in range(self.cfg.num_transformer_layers):
                #         file.write(f'{min(self.tf_norm2_inputs_var_ratios[i])}\n')
        
                # Final Normalization
                plt.plot(self.x_list, self.final_norm_inputs_var_maxs)
                plt.title(f'Max of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Max Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.savefig(f'{self.norm_path}/max_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                plt.plot(self.x_list, self.final_norm_inputs_var_mins)
                plt.title(f'Min of Variances of Final {self.cfg.norm}')
                plt.xlabel('Steps')
                plt.ylabel('Min Variance')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.norm_path}/min_of_variances_of_final_{self.cfg.norm}.png')
                plt.clf()
                # plt.plot(self.x_list, self.final_norm_inputs_var_ratios)
                # plt.title(f'Ratio of Max/Min Variances of Final {self.cfg.norm}')
                # plt.xlabel('Steps')
                # plt.ylabel('Min/Max')
                # plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                # plt.tick_params(axis='both', which='major', labelsize=10)
                # plt.savefig(f'{self.norm_path}/ratio_of_variances_of_final_{self.cfg.norm}.png')
                # plt.clf()
                with open(f'{self.norm_path}/variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                    file.write(f'Variances of Inputs of Final {self.cfg.norm}\n\n')
                    file.write(f'Max\n\n')
                    file.write(f'{max(self.final_norm_inputs_var_maxs)}\n\n')
                    file.write(f'Min\n\n')
                    file.write(f'{min(self.final_norm_inputs_var_mins)}\n\n')
                # with open(f'{self.norm_path}/ratio_of_variances_of_inputs_of_final_{self.cfg.norm}.txt', 'w') as file:
                #     file.write(f'Ratio of Min/Max Variances of Inputs of Final {self.cfg.norm}\n\n')
                #     file.write(f'Max of Max/Min\n\n') 
                #     file.write(f'{max(self.final_norm_inputs_var_ratios)}\n\n')
                #     file.write(f'Min of Max/Min\n\n') 
                #     file.write(f'{min(self.final_norm_inputs_var_ratios)}\n\n')
                                
                # Non-lin Inputs
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.nonlin_inputs_maxs[i])
                    plt.title(f'Max of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Max', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/max_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                for i in range(self.cfg.num_transformer_layers):
                    plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
                    plt.plot(self.x_list, self.nonlin_inputs_mins[i])
                    plt.title(f'Min of Inputs of {self.cfg.nonlin}', fontsize=5)
                    plt.xlabel('Steps', fontsize=5)
                    plt.ylabel('Min', fontsize=5)
                    plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                    plt.tick_params(axis='both', which='major', labelsize=5)
                plt.savefig(f'{self.norm_path}/min_of_inputs_of_{self.cfg.nonlin}.png')
                plt.clf()
                with open(f'{self.norm_path}/inputs_of_{self.cfg.nonlin}.txt', 'w') as file:
                    file.write(f'Inputs of {self.cfg.nonlin}\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.nonlin_inputs_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.nonlin_inputs_mins[i])}\n')
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list, self.matmul_norm_maxs[i])
                        plt.title(f'Max of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Max', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'{self.norm_path}/max_of_inputs_of_exp.png')
                    plt.clf()
                    for i in range(self.cfg.num_transformer_layers):
                        plt.subplot(self.vertical_num, self.horizontal_num, i+1)
                        plt.plot(self.x_list, self.matmul_norm_mins[i])
                        plt.title(f'Min of Inputs of exp of Layer {i}', fontsize=5)
                        plt.xlabel('Steps', fontsize=5)
                        plt.ylabel('Min', fontsize=5)
                        plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                        plt.tick_params(axis='both', which='major', labelsize=5)
                    plt.savefig(f'{self.norm_path}/min_of_inputs_of_exp.png')
                    plt.clf()
                with open(f'{self.norm_path}/inputs_of_exp.txt', 'w') as file:
                    file.write(f'Inputs of exp\n\n')
                    file.write(f'Max\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{max(self.matmul_norm_maxs[i])}\n')
                    file.write('\n')
                    file.write(f'Min\n\n')
                    for i in range(self.cfg.num_transformer_layers):
                        file.write(f'{min(self.matmul_norm_mins[i])}\n')
        
        
        
        
        # print(colored('logits=self.head(self.pooler(encoder_output)): {}'.format(logits.shape), 'yellow'))
        if labels is not None:
            # 여기
            # print(f'if labels is not None')
            if self.problem_type is None:  # very much from huggingface
                # 여기
                # print('if self.problem_type is None')
                if self.num_labels == 1:
                    self.problem_type = "regression"
                    # print('self.problem_type = "regression"')
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    # 여기
                    # print('elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int)')
                    self.problem_type = "single_label_classification"
                    # print('self.problem_type = "single_label_classification"')
                else:
                    self.problem_type = "multi_label_classification"
                    # print('self.problem_type = "multi_label_classification"')
            # print(colored('self.problem_type: {}'.format(self.problem_type), 'yellow'))
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                # print('loss_fct = torch.nn.MSELoss()')
                if self.num_labels == 1:
                    # print('if self.num_labels == 1')
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    # print('not if self.num_labels == 1')
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                # 여기
                # print('elif self.problem_type == "single_label_classification"')
                # loss_fct = torch.nn.CrossEntropyLoss()
                loss_fct = Custom_CrossEntropyLoss()
                # print('loss_fct = torch.nn.CrossEntropyLoss()')
                # print(f'loss_fct: {loss_fct.shape}')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                # print(f'labels: {labels}') # b
                # print(f'logits: {logits.shape}') # b 2
                
            elif self.problem_type == "multi_label_classification":
                # print('elif self.problem_type == "multi_label_classification"')
                loss_fct = torch.nn.BCEWithLogitsLoss()
                # print('loss_fct = torch.nn.BCEWithLogitsLoss()')
                loss = loss_fct(logits, labels)
        else:
            # print('not if labels is not None')
            loss = logits.new_zeros((1,))

        
        self.loss_list.append(loss.item())
        # print(f'loss: {loss}')
        
        if self.count < self.cfg.eval_graph_interval:
            last_graph_interval_loss = sum(self.loss_list) / len(self.loss_list)
            self.last_graph_interval_loss_list.append(last_graph_interval_loss)
            # print('\nLoss: {}, last_graph_interval_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_graph_interval_loss, self.cfg.num_transformer_layers, self.count))
        else:
            last_graph_interval_loss = sum(self.loss_list[-self.cfg.eval_graph_interval:]) / len(self.loss_list[-self.cfg.eval_graph_interval:])
            self.last_graph_interval_loss_list.append(last_graph_interval_loss)
            if self.best_loss == 0 or last_graph_interval_loss < self.best_loss:
                self.best_loss = last_graph_interval_loss
            # print('\nLoss: {}, last_graph_interval_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_graph_interval_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
        # for i in range(self.cfg.num_transformer_layers):
        #     norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
        #     # print('Matmul_{}: {}'.format(i, norm_i.item()))
        #     self.matmul_results[i].append(norm_i.item())
            
            # # Loss 추가
            # if norm_i > 450:
            #     loss += 0.1 * norm_i
            
        if self.count % self.cfg.eval_graph_interval == 0:
            plt.plot(self.x_list, self.loss_list)
            plt.title('Loss')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'loss', 'losses.png'))
            plt.clf()
            plt.plot(self.x_list, self.last_graph_interval_loss_list)
            plt.title(f'Last {self.cfg.eval_graph_interval} losses')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'loss', f'last_{self.cfg.eval_graph_interval}_losses.png'))
            plt.clf()
            # for i in range(self.cfg.num_transformer_layers):
            #     plt.subplot(self.vertical_num, self.horizontal_num, i + 1)
            #     plt.plot(self.x_list, self.matmul_results[i])
            #     plt.title('Matmul_{}'.format(i))
            #     plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'matmuls.png'))
            # plt.clf()
        if self.cfg.get_input_range:
            act_ftn_inputs_max = torch.max(act_ftn_inputs).detach().cpu()
            act_ftn_inputs_min = torch.min(act_ftn_inputs).detach().cpu()
            self.act_ftn_inputs_maxs.append(act_ftn_inputs_max)
            self.act_ftn_inputs_mins.append(act_ftn_inputs_min)
            # print(f'self.x_list: {len(self.x_list)}')
            # print(f'self.act_ftn_inputs_maxs: {len(self.act_ftn_inputs_maxs)}')
            # print(f'Max of Inputs of {self.cfg.classification_head.nonlin}: {max(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:])}')
            # print(f'Min of Inputs of {self.cfg.classification_head.nonlin}: {min(self.act_ftn_inputs_mins[-self.cfg.eval_graph_interval:])}')
            if self.count % self.cfg.eval_graph_interval == 0:
                plt.plot(self.x_list, self.act_ftn_inputs_maxs)
                # print(f'len(self.x_list[-self.cfg.eval_graph_interval:]): {len(self.x_list[-self.cfg.eval_graph_interval:])}')
                # print(f'len(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:]): {len(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:])}')
                plt.title(f'Max of Inputs of {self.cfg.classification_head.nonlin}')
                plt.xlabel('Steps')
                plt.ylabel('Max')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.act_ftn_path}/max_of_inputs_of_{self.cfg.classification_head.nonlin}.png')
                plt.clf()
                plt.plot(self.x_list, self.act_ftn_inputs_mins)
                plt.title(f'Min of Inputs of {self.cfg.classification_head.nonlin}')
                plt.xlabel('Steps')
                plt.ylabel('Min')
                plt.gca().yaxis.set_major_locator(MaxNLocator(10))
                plt.tick_params(axis='both', which='major', labelsize=10)
                plt.savefig(f'{self.act_ftn_path}/min_of_inputs_of_{self.cfg.classification_head.nonlin}.png')
                plt.clf()
                with open(f'{self.act_ftn_path}/{self.cfg.classification_head.nonlin}.txt', 'w') as file:
                    file.write(f'{self.cfg.classification_head.nonlin}\n')
                    file.write(f'Max: {max(self.act_ftn_inputs_maxs[-self.cfg.eval_graph_interval:])}\n')
                    file.write(f'Min: {min(self.act_ftn_inputs_mins[-self.cfg.eval_graph_interval:])}\n')
        if self.cfg.get_grad:
            return dict(logits=logits, loss=loss), before_zero_indexing_hidden_states, first_token_tensor
        else:
            return dict(logits=logits, loss=loss)
