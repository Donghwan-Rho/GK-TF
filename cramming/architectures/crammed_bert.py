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

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    GLU,
    get_extended_attention_mask,
    _init_module,
)
from .attention import get_attention_mechanism
import matplotlib.pyplot as plt
from termcolor import colored
import os

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes=None):
    """See the config file for details on what is possible."""
    print('construct_crammed_bert')
    print('cfg_arch\n',cfg_arch)
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    if downstream_classes is None:
        if config.arch["objective_layout"] == "MLM":
            model = ScriptableLMForPreTraining(config)
        elif config.arch["objective_layout"] == "SCRIPT":
            model = ScriptableLMForSCRIPTTraining(config)
        else:
            raise ValueError(f"Invalid layout {config.arch['objective_layout']} of training objective given.")
    else:
        model = ScriptableLMForSequenceClassification(config)
    return model

class AttentionComponent(torch.nn.Module):
    def __init__(self, idx, hidden_size, cfg_attention, use_bias=True):
        super().__init__()
        self.self_attention = get_attention_mechanism(idx, hidden_size, cfg_attention)
        if cfg_attention.skip_output_projection:
            self.dense = torch.nn.Identity()
        else:
            self.dense = torch.nn.Linear(self.self_attention.output_dim, hidden_size, bias=use_bias)

        self.LAYOUT = self.self_attention.LAYOUT

    def forward(self, hidden_states, attention_mask: Optional[torch.Tensor] = None):
        # Ordinary
        #######################################
        output = self.self_attention(hidden_states, attention_mask)
        # print('att output', output.shape)
        output = self.dense(output)
        return output
        #######################################
        # # Ordinary-matmul
        # #######################################
        # output, matmul_result = self.self_attention(hidden_states, attention_mask)
        # # print('att output', output.shape)
        # output = self.dense(output)
        # return output, matmul_result
        # #######################################
        # # Heatmap
        # #######################################
        # output, matmul_result, heatmap_result = self.self_attention(hidden_states, attention_mask)
        # output = self.dense(output)
        # return output, matmul_result, heatmap_result
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # # output, matmul_result, norm_sum = self.self_attention(hidden_states, attention_mask)
        # # output = self.dense(output)
        # # return output, matmul_result, norm_sum
        # output, matmul_result, query_norm, key_norm = self.self_attention(hidden_states, attention_mask)
        # output = self.dense(output)
        # return output, matmul_result, query_norm, key_norm
        # #######################################


class FFNComponent(torch.nn.Module):
    """Note: The FF layer is not auto-scaled when using a GLU type activation.
    It actually turned out better not to scale it, so here the block is effectively smaller than may be expected.

    The neox suggestion for approx. equal parameter count is int(4 * 2 / 3 * hidden_size) * 2 [this is ~5.33]
    """

    def __init__(self, hidden_size, intermed_size, nonlin_fn=torch.nn.GELU, use_bias=True):
        super().__init__()
        self.dense_in = torch.nn.Linear(hidden_size, intermed_size, bias=use_bias)
        self.nonlin = nonlin_fn()
        # print('self.nonlin', self.nonlin)
        if isinstance(self.nonlin, GLU):
            intermed_output_size = intermed_size // 2
        else:
            intermed_output_size = intermed_size
        self.dense_out = torch.nn.Linear(intermed_output_size, hidden_size, bias=use_bias)
        
    def forward(self, hidden_states):
        # Ordinary
        #######################################
        if torch.isnan(self.dense_out(self.nonlin(self.dense_in(hidden_states)))).any():
            print('FFN nan')
            return
        return self.dense_out(self.nonlin(self.dense_in(hidden_states)))
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # dense_in_output = self.dense_in(hidden_states)
        # before_FFN_GELU_norm = torch.norm(dense_in_output, p=float('inf')).item()
        # # print('before_FFN_GELU_norm', before_FFN_GELU_norm.item())
        # return self.dense_out(self.nonlin(dense_in_output)), before_FFN_GELU_norm
        # #######################################


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        # print('idx', idx) # 0, 1
        # print('cfg_arch.hidden_size', cfg_arch.hidden_size)
        # print('cfg_arch.attention', cfg_arch.attention)
        # print('cfg_arch.use_bias', cfg_arch.use_bias)
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        # GELU activation 조정
        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            # cfg_arch.nonlin: GELUglu
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        # Ordinary
        #######################################
        states2 = self.attn(self.norm1(states), attention_mask)
        states = states + self.dropout(states2)
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states
        #######################################
        # # Ordinary-matmul
        # #######################################
        # states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result
        # #######################################
        # Before normalization / GELU checking
        # #######################################
        # before_layernorm_1 = torch.norm(states, p=float('inf')).item()
        # # print('before_layernorm_1', before_layernorm_1)
                
        # states2, matmul_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        
        # before_layernorm_2 = torch.norm(states, p=float('inf')).item()
        # # print('before_layernorm_2', before_layernorm_2)
        
        # ffn_output, before_FFN_GELU_norm = self.ffn(self.norm2(states))
        # states = states + self.dropout(ffn_output)
        # return states, matmul_result, before_layernorm_1, before_layernorm_2, before_FFN_GELU_norm
        # #######################################
        # # Heatmap
        #######################################
        # states2, matmul_result, heatmap_result = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result, heatmap_result
        #######################################
        # # # Q, K L2 normalization scheme
        # #######################################
        # # states2, matmul_result, norm_sum = self.attn(self.norm1(states), attention_mask)
        # # states = states + self.dropout(states2)
        # # states = states + self.dropout(self.ffn(self.norm2(states)))
        # # return states, matmul_result, norm_sum
        # states2, matmul_result, query_norm, key_norm = self.attn(self.norm1(states), attention_mask)
        # states = states + self.dropout(states2)
        # states = states + self.dropout(self.ffn(self.norm2(states)))
        # return states, matmul_result, query_norm, key_norm
        # #######################################



class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        # print('self.seq_first', self.seq_first)
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

        # Before normalization / GELU checking
        #######################################
        self.layernorms_1 = []
        self.layernorms_2 = []
        self.gelus = []
        #######################################
        
    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        matmuls = []
        # Before normalization / GELU checking
        #######################################
        self.layernorms_1 = []
        self.layernorms_2 = []
        self.gelus = []
        #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # norms = []
        # #######################################
    
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        # print('ScriptableLM input_ids', input_ids.dtype)
        
        # Ordinary
        #######################################
        hidden_states = self.embedding(input_ids)
        # print('after embedding', hidden_states.shape)
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # hidden_states, before_emb_layernorm = self.embedding(input_ids)
        # # print('self.emb_layernorms', self.emb_layernorms)
        # #######################################
        
        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        # # Shape: (num_layers, num_heads)
        # q_minus_list = [[6.5453, 7.3048, 6.0966, 6.4464, 6.8230, 5.6180, 5.7366, 7.8409, 5.5652,
        # 6.8137, 6.7894, 7.3565],
        #                 [9.0119,  9.9930,  7.1479,  7.3478,  9.3569, 10.9437,  7.3196,  9.9101,
        # 10.4148,  9.2822, 13.4686, 10.2901],
        #                 [10.4658, 10.5926, 16.6584,  6.1592,  3.7752,  7.3358,  6.8034, 11.8727,
        #  6.2677,  7.5784,  4.8773,  6.0751],
        #                 [ 9.0363, 11.0546,  9.7901,  9.2901, 10.9956,  9.3972, 12.0702, 11.8807,
        # 10.6749, 10.6923, 14.3970, 10.2228],
        #                 [ 9.1939, 10.0599,  9.1099, 10.0178,  9.2778,  8.9418,  5.9639, 10.2251,
        #  7.5559, 11.3001,  9.3974,  7.5936],
        #                 [ 9.8671,  9.3087,  8.0332,  9.6969,  8.6711,  8.3204, 11.0814,  9.0200,
        #  7.0178,  9.2484,  9.1459,  9.6874],
        #                 [ 9.5752, 10.8706, 10.4627, 11.1658,  8.0520, 11.3200, 11.2173, 10.7415,
        # 11.0518, 11.3651, 11.5943, 10.4282],
        #                 [ 9.1617,  9.9473,  9.5779,  8.1709, 10.8587, 10.0076,  9.9062, 10.1503,
        #  9.9007,  8.8131,  9.9549, 10.9858],
        #                 [5.7378, 8.3163, 8.0467, 8.5597, 5.5818, 8.8003, 9.4949, 9.0927, 8.0327,
        # 8.5704, 8.8784, 8.9042]]
        # k_minus_list = [[6.5972, 9.4494, 6.9003, 6.8696, 8.5952, 5.8139, 6.0617, 9.3341, 6.7450,
        # 8.0636, 8.4179, 8.0233],
        #                 [10.9063, 11.4491,  7.8387,  7.5795, 11.3935, 12.6177,  9.1905, 11.3523,
        # 11.8661, 10.5776, 14.3794, 11.9101],
        #                 [11.7445, 10.9189, 16.5787,  6.3045,  4.6811,  7.7153,  7.4379, 12.5885,
        #  6.7564,  8.0339,  5.3770,  6.0490], 
        #                 [10.4362, 11.5250, 10.5673, 10.4981, 11.8140, 10.7983, 12.3806, 12.9659,
        # 11.7629, 11.8772, 14.7792, 11.2239],
        #                 [ 9.1048,  9.5516,  8.8884, 10.2973,  9.4112,  9.2654,  5.7005, 10.1705,
        #  7.8035, 10.9841,  8.7755,  7.7430],
        #                 [ 9.2948,  9.0743,  7.7677,  9.3053,  8.8886,  8.1638, 10.7158,  8.8252,
        #  6.4585,  8.7158,  8.7986,  8.6417],
        #                 [ 9.4486, 10.0466,  9.8508, 10.3644,  8.0525, 10.6902, 10.4850, 10.0586,
        # 10.3643, 10.7107, 11.2756, 10.1299],
        #                 [ 9.0736, 10.0668,  9.7290,  8.5708, 10.5558, 10.2731, 10.1509, 10.0850,
        # 10.0175,  8.9946, 10.1294, 10.7520],
        #                 [5.6573, 8.7088, 8.5017, 9.1730, 5.2062, 9.2441, 9.8183, 9.5207, 8.5354,
        # 9.1494, 9.5691, 9.4102]]
            
        for i, layer_module in enumerate(self.layers):
            # Ordinary
            #######################################
            hidden_states = layer_module(hidden_states, attention_mask)
            #######################################
            # # Ordinary-matmul
            # #######################################
            # hidden_states, matmul = layer_module(hidden_states, attention_mask)
            # #######################################
            # # Before normalization / GELU checking
            # #######################################
            # hidden_states, matmul, before_layernorm_1, before_layernorm_2, before_FFN_GELU_norm = layer_module(hidden_states, attention_mask)
            # self.layernorms_1.append(before_layernorm_1)
            # self.layernorms_2.append(before_layernorm_2)
            # self.gelus.append(before_FFN_GELU_norm)
            # #######################################
            # # Heatmap
            # #######################################
            # hidden_states, matmul, heatmap_result = layer_module(hidden_states, attention_mask)
            # heatmaps.append(heatmap_result)
            # #######################################
            
            # # Q, K L2 normalization scheme
            # #######################################
            # hidden_states, matmul, norm_sum = layer_module(hidden_states, attention_mask)
            # # norms.append(norm_sum)
            
            # # query_norm, key_norm: (bh, m)
            # # hidden_states, matmul, query_norm, key_norm = layer_module(hidden_states, attention_mask)
            
            # # print('{} query norm: {}'.format(i, query_norm.shape))
            # # print('{} key norm: {}'.format(i, key_norm.shape))
            
            # # b=128, h=12, m=128
            # # query_norm_bhm = query_norm.view(128, 12, 128) # b, h, m
            # # key_norm_bhm = key_norm.view(128, 12, 128)
            # # # print('query_norm b h m', query_norm.shape)
            # # query_norm_hm = torch.mean(query_norm_bhm, dim=0) # h, m
            # # key_norm_hm = torch.mean(key_norm_bhm, dim=0)
            # # # print('query_norm h m', query_norm.shape)
            # # query_norm_h = torch.mean(query_norm_hm, dim=1) # h
            # # key_norm_h = torch.mean(key_norm_hm, dim=1)
            
            # # print('{} query norm: {}'.format(i, query_norm_h))
            # # print('{} key norm: {}'.format(i, key_norm_h))
            
            # # for h in range(12): # h=12
            # #     query_norm_hm[h] -= q_minus_list[i][h]
            # #     key_norm_hm[h] -= k_minus_list[i][h]
            
            # # query_norm_hm = torch.square(query_norm_hm)
            # # key_norm_hm = torch.square(key_norm_hm)
            
            # # query_norm_sum = torch.sum(query_norm_hm)
            # # key_norm_sum = torch.sum(key_norm_hm)
            # # norm_sum = query_norm_sum + key_norm_sum
            # # norms.append(norm_sum)
            # # #######################################
            
            # # Ordinary-matmul
            # #######################################
            # matmuls.append(matmul)
            # #######################################
            
            # # Heatmap
            # ################################
            # if input_ids.shape[1] == 24:
            #     fig, ax0 = plt.subplots(1, 1)
            #     c = ax0.pcolor(heatmap_result.detach().cpu().numpy())
            #     ax0.set_title('Softmax')
            #     ax0.set_xlabel('Words')
            #     ax0.set_ylabel('Words')
            #     # ax0.set_xticklabels(lst)
            #     plt.colorbar(c, ax=ax0)
            #     plt.gca().invert_yaxis()
            #     ax0.xaxis.tick_top()
            #     plt.savefig('heatmap_{}.png'.format(i))
            # ################################
                
            # print('ScriptableLM tf hidden_states', hidden_states.dtype)
            # print('layer {} after attention: {}'.format(i, hidden_states.shape))

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()
        # hidden_states = hidden_states.type(torch.float16)
        # print('ScriptableLM tf hidden_states', hidden_states.dtype)
        # print('ScriptableLM self.final_norm(hidden_states)', self.final_norm(hidden_states).dtype)
        
        # Ordinary
        #######################################
        return self.final_norm(hidden_states)
        #######################################
        # # Ordinary-matmul
        # #######################################
        # return self.final_norm(hidden_states), matmuls
        # #######################################
        # # Before normalization / GELU checking
        # #######################################
        # before_layernorm_final = torch.norm(hidden_states, p=float('inf')).item()
        # # print('before_layernorm_final', before_layernorm_final.item())
        # return self.final_norm(hidden_states), matmuls, before_emb_layernorm, self.layernorms_1, self.layernorms_2, self.gelus, before_layernorm_final
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # # print(norms)
        # # print('matmuls', matmuls)
        # return self.final_norm(hidden_states), matmuls, norms
        # #######################################
    


class ScriptableLMForPreTraining(PreTrainedModel):
    """Pretraining version with optional prediction head and variant for sparse prediction."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)

        if not self.cfg.skip_head_transform:
            # print('not self.cfg.skip_head_transform:')
            self.prediction_head = PredictionHeadComponent(self.cfg)
        else:
            # print('not not self.cfg.skip_head_transform:')
            # 여기
            self.prediction_head = torch.nn.Identity()  # from linear in old version

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction

        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_100_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        
        # Before normalization / GELU checking
        #######################################
        self.emb_layer_list = []
        self.last_100_emb_layer_list = []
        self.layernorm_list_1 = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.layernorm_list_2 = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.last_100_layernorm_list_1 = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.last_100_layernorm_list_2 = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.gelu_list = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.last_100_gelu_list = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.final_layernorm_list = []
        self.last_100_final_layernorm_list = []
        #######################################
        
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
        # print('input_ids:', input_ids.shape)
        self.count += 1
        self.x_list.append(self.count)
        
        # Ordinary
        #######################################
        outputs = self.encoder(input_ids, attention_mask)
        #######################################
        # # Ordinary-matmul
        # #######################################
        # outputs, matmuls_from_enc = self.encoder(input_ids, attention_mask)
        # #######################################
        # # Before normalization / GELU checking
        # #######################################
        # outputs, matmuls_from_enc, before_emb_layernorm, layernorms_1, layernorms_2, before_gelus, before_layernorm_final = self.encoder(input_ids, attention_mask)
        # # print('layernorms_1', layernorms_1)
        # self.emb_layer_list.append(before_emb_layernorm)
        # self.final_layernorm_list.append(before_layernorm_final)
        # for i in range(self.cfg.num_transformer_layers):
        #     self.layernorm_list_1[i].append(layernorms_1[i])
        #     self.layernorm_list_2[i].append(layernorms_2[i])
        #     self.gelu_list[i].append(before_gelus[i])
            
        # if self.count < 100:
        #     last_100_emb_layernorm = sum(self.emb_layer_list) / len(self.emb_layer_list)
        #     self.last_100_emb_layer_list.append(last_100_emb_layernorm)
        #     last_100_final_layernorm = sum(self.final_layernorm_list) / len(self.final_layernorm_list)
        #     self.last_100_final_layernorm_list.append(last_100_final_layernorm)
        #     for i in range(self.cfg.num_transformer_layers):
        #         last_100_layernorm_1 = sum(self.layernorm_list_1[i]) / len(self.layernorm_list_1[i])
        #         self.last_100_layernorm_list_1[i].append(last_100_layernorm_1)
        #         # print('self.last_100_layernorm_list_1[i]', self.last_100_layernorm_list_1[i])
        #         last_100_layernorm_2 = sum(self.layernorm_list_2[i]) / len(self.layernorm_list_2[i])
        #         self.last_100_layernorm_list_2[i].append(last_100_layernorm_2)
        #         # print('self.last_100_layernorm_list_2[i]', self.last_100_layernorm_list_2[i])
        #         last_100_gelu = sum(self.gelu_list[i]) / len(self.gelu_list[i])
        #         self.last_100_gelu_list[i].append(last_100_gelu)
        # else:
        #     last_100_emb_layernorm = sum(self.emb_layer_list[-100 :]) / len(self.emb_layer_list[-100 :])
        #     self.last_100_emb_layer_list.append(last_100_emb_layernorm)
        #     last_100_final_layernorm = sum(self.final_layernorm_list[-100 :]) / len(self.final_layernorm_list[-100 :])
        #     self.last_100_final_layernorm_list.append(last_100_final_layernorm)
        #     for i in range(self.cfg.num_transformer_layers):
        #         last_100_layernorm_1 = sum(self.layernorm_list_1[i][-100 :]) / len(self.layernorm_list_1[i][-100 :])
        #         self.last_100_layernorm_list_1[i].append(last_100_layernorm_1)
        #         # print('self.last_100_layernorm_list_1[i]', self.last_100_layernorm_list_1[i])
        #         last_100_layernorm_2 = sum(self.layernorm_list_2[i][-100 :]) / len(self.layernorm_list_2[i][-100 :])
        #         self.last_100_layernorm_list_2[i].append(last_100_layernorm_2)
        #         # print('self.last_100_layernorm_list_2[i]', self.last_100_layernorm_list_2[i])
        #         last_100_gelu = sum(self.gelu_list[i][-100 :]) / len(self.gelu_list[i][-100 :])
        #         self.last_100_gelu_list[i].append(last_100_gelu)
            
        # if self.count % 100 == 0:
        #     plt.plot(self.x_list, self.last_100_emb_layer_list)
        #     plt.title('Embedding layernorm')
        #     plt.xlabel('Steps')
        #     plt.savefig('last_100_emb_layernorms.png')
        #     plt.clf()
        #     plt.plot(self.x_list, self.last_100_final_layernorm_list)
        #     plt.title('Final layernorm')
        #     plt.xlabel('Steps')
        #     plt.savefig('last_100_final_layernorms.png')
        #     plt.clf()
        #     for i in range(self.cfg.num_transformer_layers):
        #         plt.subplot(5, 4, i + 1)
        #         plt.plot(self.x_list, self.last_100_layernorm_list_1[i])
        #         plt.title('{}th LN 1'.format(str(i)))
        #         plt.xlabel('Steps')
        #     plt.savefig('last_100_layernorm_1.png')
        #     plt.clf()
        #     for i in range(self.cfg.num_transformer_layers):
        #         plt.subplot(5, 4, i + 1)
        #         plt.plot(self.x_list, self.last_100_layernorm_list_2[i])
        #         plt.title('{}th LN 2'.format(str(i)))
        #         plt.xlabel('Steps')
        #     plt.savefig('last_100_layernorm_2.png')
        #     plt.clf()
        #     for i in range(self.cfg.num_transformer_layers):
        #         plt.subplot(5, 4, i + 1)
        #         plt.plot(self.x_list, self.last_100_gelu_list[i])
        #         plt.title('{}th before GELU'.format(str(i)))
        #         plt.xlabel('Steps')
        #     plt.savefig('last_100_gelu.png')
        #     plt.clf()
        # #######################################
        # # Q, K L2 normalization scheme
        # #######################################
        # outputs, matmuls_from_enc, norms_from_enc = self.encoder(input_ids, attention_mask)
        # #######################################
        
        outputs = outputs.view(-1, outputs.shape[-1])
        # print('outputs:', outputs.shape)
        
        # # Ordinary-matmul
        # #######################################
        # len_matmuls = len(matmuls_from_enc)
        # #######################################

        if self.sparse_prediction and labels is not None:
            # print('self.sparse_prediction and labels is not None')
            # 여기
            # Loss 계산되는 부분
            masked_lm_loss = self._forward_sparse(outputs, labels)
            
            self.loss_list.append(masked_lm_loss.item())
            
            if self.count < 100:
                last_100_loss = sum(self.loss_list) / len(self.loss_list)
                self.last_100_loss_list.append(last_100_loss)
                print('\nLoss: {}, Last_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.cfg.num_transformer_layers, self.count))
            else:
                last_100_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
                self.last_100_loss_list.append(last_100_loss)
                if self.best_loss == 0 or last_100_loss < self.best_loss:
                    self.best_loss = last_100_loss
                print('\nLoss: {}, Last_100_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
            # # Ordinary-matmul
            # #######################################
            # for i in range(len_matmuls):
            #     norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
            #     # print('Matmul_{}: {}'.format(i, norm_i.item()))
            #     self.matmul_results[i].append(norm_i.item())
            # #######################################
                
            # ######################################
            #     # Loss 추가
            #     if norm_i > 450:
            #         masked_lm_loss += 10 * norm_i
            # print('Norm Penalty: O')
            # ######################################
            
                # # Q, K L2 normalization scheme
                # #######################################
                # if self.count < 1000000:
                #     # masked_lm_loss += 0.1 * (0.99 ** self.count) * norms_from_enc[i]
                #     masked_lm_loss += 0.01 * norms_from_enc[i]
                # #######################################
          
            if self.count % 100 == 0:
                plt.plot(self.x_list, self.loss_list)
                plt.title('Loss')
                plt.xlabel('Steps')
                plt.savefig('losses.png')
                plt.clf()
                plt.plot(self.x_list, self.last_100_loss_list)
                plt.title('Last 100 losses')
                plt.xlabel('Steps')
                plt.savefig('last_100_losses.png')
                plt.clf()
                # # Ordinary-matmul
                # #######################################
                # for i in range(len_matmuls):
                #     plt.subplot(5, 4, i + 1)
                #     plt.plot(self.x_list, self.matmul_results[i])
                #     plt.title('Matmul_{}'.format(i))
                #     plt.xlabel('Steps')
                # plt.savefig('matmuls.png')
                # plt.clf()                
                # #######################################
            
        else:
            # print('not self.sparse_prediction and labels is not None')
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

        labels = labels.view(-1)
        mask_positions = labels.view(-1) != self.loss_fn.ignore_index
        num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
        # outputs = outputs[mask_positions]  # not allowed as dynamic shape op
        # labels = labels[mask_positions]
        # torch.masked_select(labels, mask_positions)  # not allowed as a dynamic shape operator

        # indices = torch.arange(mask_positions.shape[0], device=outputs.device)[mask_positions] # not allowed
        indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]  # ugh

        outputs = outputs[indices]  # not allowed as dynamic shape op, but ok with indices
        labels = labels[indices]
        # alternative:
        # outputs = torch.take_along_dim(outputs, indices.view(-1, 1), 0)
        # labels = torch.take(labels, indices)

        outputs = self.decoder(self.prediction_head(outputs))
        # print('outputs', outputs.shape)
        masked_lm_loss = self.loss_fn(outputs, labels)
        return masked_lm_loss
 
class ScriptableLMForSequenceClassification(PreTrainedModel):
    """Classification head and pooler."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        # print('dddd self.cfg\n', self.cfg)
        # print('self.cfg.attention.seq_op_in_fp32', self.cfg.attention.seq_op_in_fp32)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.pooler = PoolingComponent(self.cfg.classification_head, self.cfg.hidden_size)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

        self.count = 0
        self.x_list = []
        self.loss_list = []
        self.last_100_loss_list = []
        self.matmul_results = [[] for _ in range(self.cfg.num_transformer_layers)]
        self.best_loss = 0
        self.before_Tanh_norm_list = []
        self.last_100_before_Tanh_norm_list = []
        os.makedirs(self.cfg.task_name)

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
        # Ordinary
        ######################################
        encoder_output = self.encoder(input_ids, attention_mask)
        ######################################
        # # Ordinary-matmul
        #######################################
        # encoder_output,  matmuls_from_enc = self.encoder(input_ids, attention_mask)
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # encoder_output,  matmuls_from_enc, before_emb_layernorm, layernorms_1, layernorms_2, gelus, before_layernorm_final = self.encoder(input_ids, attention_mask)
        # #######################################
        
        # if not self.cfg.attention.seq_op_in_fp32:
        #     encoder_output = encoder_output.type(torch.float16)
        # print('encoder_output', encoder_output.dtype)
        
        # Ordinary / Ordinary-matmul
        #######################################
        logits = self.head(self.pooler(encoder_output))
        #######################################
        # # Before normalization / GELU checking
        # #######################################
        # pooler_output , before_Tanh_norm = self.pooler(encoder_output)
        # logits = self.head(pooler_output)
        # #######################################
        
        # print(colored('logits: {}'.format(logits.shape), 'green'))
        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"
            # print(colored('self.problem_type: {}'.format(self.problem_type), 'green'))
            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        else:
            loss = logits.new_zeros((1,))

        # # Ordinary-matmul
        # #######################################
        # len_matmuls = len(matmuls_from_enc)
        # #######################################
        
        self.count += 1
        self.x_list.append(self.count)
        self.loss_list.append(loss.item())
        # # Before normalization / GELU checking
        # self.before_Tanh_norm_list.append(before_Tanh_norm)
        
        if self.count < 100:
            last_100_loss = sum(self.loss_list) / len(self.loss_list)
            self.last_100_loss_list.append(last_100_loss)
            last_100_before_Tanh_norm = sum(self.before_Tanh_norm_list) / len(self.before_Tanh_norm_list)
            self.last_100_before_Tanh_norm_list.append(last_100_before_Tanh_norm)
            # print('\nLoss: {}, Last_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.cfg.num_transformer_layers, self.count))
        else:
            last_100_loss = sum(self.loss_list[-100 :]) / len(self.loss_list[-100 :])
            self.last_100_loss_list.append(last_100_loss)
            if self.best_loss == 0 or last_100_loss < self.best_loss:
                self.best_loss = last_100_loss
            last_100_before_Tanh_norm = sum(self.before_Tanh_norm_list[-100 :]) / len(self.before_Tanh_norm_list[-100 :])
            self.last_100_before_Tanh_norm_list.append(last_100_before_Tanh_norm)
            # print('\nLoss: {}, Last_100_losses: {}, Best_100_losses: {}, Layers: {}, Count: {}'.format(masked_lm_loss.item(), last_100_loss, self.best_loss, self.cfg.num_transformer_layers, self.count))
            
        # # Ordinary-matmul
        # #######################################
        # for i in range(len_matmuls):
        #     norm_i = torch.norm(matmuls_from_enc[i], p=float('inf'))
        #     # print('Matmul_{}: {}'.format(i, norm_i.item()))
        #     self.matmul_results[i].append(norm_i.item())
            
        #     # # Loss 추가
        #     # if norm_i > 450:
        #     #     loss += 0.1 * norm_i
        # #######################################
            
        if self.count % 100 == 0:
            plt.plot(self.x_list, self.loss_list)
            plt.title('Loss')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'losses.png'))
            plt.clf()
            plt.plot(self.x_list, self.last_100_loss_list)
            plt.title('Last 100 losses')
            plt.xlabel('Steps')
            plt.savefig(os.path.join(self.cfg.task_name, 'last_100_losses.png'))
            plt.clf()
            # # Ordinary-matmul
            # #######################################
            # for i in range(len_matmuls):
            #     plt.subplot(5, 4, i + 1)
            #     plt.plot(self.x_list, self.matmul_results[i])
            #     plt.title('Matmul_{}'.format(i))
            #     plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'matmuls.png'))
            # plt.clf()
            # #######################################
            # # Before normalization / GELU checking
            # #######################################
            # plt.plot(self.x_list, self.before_Tanh_norm_list)
            # plt.title('Before Tanh norm')
            # plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'before_Tanh_norm.png'))
            # plt.clf()
            # plt.plot(self.x_list, self.last_100_before_Tanh_norm_list)
            # plt.title('Last 100 before Tanh norm')
            # plt.xlabel('Steps')
            # plt.savefig(os.path.join(self.cfg.task_name, 'last_100_before_Tanh_norm.png'))
            # plt.clf()
            # #######################################
                     
        
        return dict(logits=logits, loss=loss)


class ScriptableLMForSCRIPTTraining(PreTrainedModel):
    """Pretraining machinery using SCRIPT from Nijkamp et al., 2021. Always running sparse prediction."""

    config_class = crammedBertConfig
    ALPHA = 1.0  # SCRIPT constant

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)
        self.num_labels = self.cfg.num_labels

        self.encoder = ScriptableLM(config)
        self.prediction_head = PredictionHeadComponent(self.cfg)

        self.decoder = torch.nn.Linear(self.cfg.embedding.embedding_dim, self.cfg.embedding.vocab_size, bias=self.cfg.decoder_bias)
        self.decoder.weight = self.encoder.embedding.word_embedding.weight

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sparse_prediction = self.cfg.sparse_prediction
        assert self.sparse_prediction

        self._init_weights()

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        loss = torch.tensor(0.0, dtype=torch.float, device=input_ids.device)

        outputs = self.encoder(input_ids, attention_mask)
        outputs = outputs.view(-1, outputs.shape[-1])

        if labels is not None:
            # ## Generation pass ##
            labels = labels.view(-1)
            mask_positions = labels.view(-1) != self.loss_fn.ignore_index
            num_masks_guaranteed = round(self.sparse_prediction * labels.shape[0])
            indices = torch.argsort(mask_positions.int())[-num_masks_guaranteed:]

            # sparse outputs for prediction
            outputs = outputs[indices]
            labels = labels[indices]

            logits = self.decoder(self.prediction_head(outputs))  # sparse logits
            loss += self.loss_fn(logits, labels)

            # ## Discrimination pass ##
            resampled_token_ids = self._gumbel_sample(logits.detach())
            discriminator_input_ids = input_ids.clone().view(-1)
            discriminator_input_ids[indices] = resampled_token_ids

            critic_labels = (input_ids.view(-1) != discriminator_input_ids).to(outputs.dtype)

            outputs = self.encoder(discriminator_input_ids.view_as(input_ids), attention_mask).view(-1, outputs.shape[-1])
            disc_logits = self.decoder(self.prediction_head(outputs))  # full logits
            binary_logits = self._get_binary_logits(disc_logits)

            # ELECTRA-type discriminator:
            loss += self.ALPHA * torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, critic_labels)

        else:
            logits = self.decoder(self.prediction_head(outputs))
            loss += outputs.new_zeros((1,))

        return {"loss": loss, "logits": logits}

    def _get_binary_logits(self, logits):
        # Convert to binary decision as described in SCRIPT
        # exp_logitsum = torch.exp(disc_logits).sum(dim=-1)  # autocast ok?
        # binary_logits = torch.stack([1 / (exp_logitsum + 1), exp_logitsum / (exp_logitsum + 1)], dim=-1)  # stack minus and plus
        # instead, we can also compute logit[binary_logits], which is

        # let y = sum(exp(logits)) / ( sum(exp(logits))+1 ), 1-y = 1 / ( sum(exp(logits))+1 )
        # log(y / (1-y)) = log( sum(exp(logits)) / ( sum(exp(logits))+1 ) * ( sum(exp(logits))+1 ) / 1)
        #                = log(sum(exp(logits))
        # Then, we can use BCEWithLogitsLoss, to safely compute logit probs via sigmoids
        return torch.logsumexp(logits, dim=-1)

    def _gumbel_sample(self, logits, temperature=1.0):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        return ((logits / temperature) + self._gumbel_noise(logits)).argmax(dim=-1)

    def _gumbel_noise(self, inputs, eps=1e-9):
        """via https://github.com/lucidrains/electra-pytorch/blob/master/electra_pytorch/electra_pytorch.py"""
        noise = torch.zeros_like(inputs).uniform_(0, 1)
        return -torch.log(-torch.log(noise + eps) + eps)


class ScriptableLMForTokenClassification(PreTrainedModel):
    """Classification head without pooling."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.encoder = ScriptableLM(config)
        self.head = torch.nn.Linear(self.cfg.classification_head.head_dim, self.num_labels)

        self.problem_type = None
        self._init_weights()

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

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        logits = self.head(self.encoder(input_ids, attention_mask))

        if labels is not None:
            if self.problem_type is None:  # very much from huggingface
                if self.num_labels == 1:
                    self.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.problem_type = "single_label_classification"
                else:
                    self.problem_type = "multi_label_classification"

            if self.problem_type == "regression":
                loss_fct = torch.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.problem_type == "single_label_classification":
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.problem_type == "multi_label_classification":
                loss_fct = torch.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            else:
                raise ValueError("Wrong problem type!")
        else:
            loss = logits.new_zeros((1,))

        return dict(logits=logits, loss=loss)


# ###### HF registry here ############### #

AutoConfig.register("crammedBERT", crammedBertConfig)
AutoModel.register(crammedBertConfig, ScriptableLM)
AutoModelForMaskedLM.register(crammedBertConfig, ScriptableLMForPreTraining)
AutoModelForSequenceClassification.register(crammedBertConfig, ScriptableLMForSequenceClassification)
AutoModelForTokenClassification.register(crammedBertConfig, ScriptableLMForTokenClassification)
