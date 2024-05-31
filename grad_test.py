"""Script to evaluate a pretrained model."""

import os  

import torch
import torch.nn as nn
import hydra
from torchviz import make_dot


import time
import datetime
import logging
from collections import defaultdict
import transformers
from omegaconf import OmegaConf, open_dict

import cramming
import evaluate
from termcolor import colored
from safetensors.torch import load_file, save_file
from cramming.data.downstream_task_preparation import prepare_task_dataloaders_modified
import json
from cramming.architectures.crammed_bert_modified import crammedBertConfig
from cramming.architectures.architectures_grad import ScriptableLMForPreTraining_grad

log = logging.getLogger(__name__)

def convert_to_serializable(outputs):
    return {
        'logits': outputs['logits'].detach().cpu().tolist(),
        'loss': outputs['loss'].item()  # loss 값은 스칼라이므로 item()을 사용해 Python 숫자로 변환
    }
    
def main_downstream_process(cfg, setup):
    """This function controls the central routine."""
    local_time = time.time()
    
    ##########################################
    # Choose 2 or 10 layers
    cfg.name = "layers2_12hrs_layernorm_relu"
    # cfg.name = "10lys_ln_relu_matX_var100_ratioX_249900_250000_penaltycoerr100"
    
    # Choose whether use Pooling_lora
    Pooling_lora = False
    ##########################################
    
    local_checkpoint_folder = os.path.join(cfg.base_dir, cfg.name, "checkpoints")
    all_checkpoints = [f for f in os.listdir(local_checkpoint_folder)]
    checkpoint_paths = [os.path.join(local_checkpoint_folder, c) for c in all_checkpoints]
    checkpoint_name = checkpoint_paths[cfg.eval.ckpt_num]
    tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_name)
    with open(os.path.join(checkpoint_name, "model_config.json"), "r") as file:
        cfg_arch = OmegaConf.create(json.load(file))  # Could have done pure hydra here, but wanted interop
    cfg_arch.architectures = ["ScriptableCrammedBERT-grad"]
    
    if Pooling_lora:
        safetensors_name = f'{cfg.name}_eval_rte_128padding_poolinglora.safetensors'
        print('poolinglora in name')
        cfg_arch.poolinglora = True
    else:
        safetensors_name = f'{cfg.name}_eval_rte_128padding.safetensors'
        print('poolinglora NOT in name')
        cfg_arch.poolinglora = False
    
    tasks, train_dataset, eval_datasets = prepare_task_dataloaders_modified(tokenizer, cfg.eval, cfg.impl)
    print(f'tasks: {tasks}')
    metrics = dict()
    stats = defaultdict(list)
    
    for task_name, task in tasks.items():
        print(colored('task_name: {}'.format(task_name), 'red'))
        print(colored('task: {}'.format(task), 'red'))
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        log.info(f"Sentense test eval with padding size 128")
        log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        
        cfg_arch.task_name = task_name
        
        # 여기서 제가 드린 safetensor 파일을 불러와주세요.
        # load_file에서 safetensors 파일을 불러올 경로를 수정해야 합니다.
        #############################
        cfg_arch.nonlin = "Approx_ReLUglu"
        cfg_arch.norm = "Approx_LayerNorm"
        model = cramming.construct_model(cfg_arch, tokenizer.vocab_size, downstream_classes=task["num_classes"])
        print(f'model: {model}')
        current_path = os.getcwd()
        # print(f'current_path: {current_path}')
        model_state_dict_loaded = load_file(os.path.join(current_path, '..', '..', '..', '..', '..', 'safetensors', safetensors_name))
        model.load_state_dict(model_state_dict_loaded)
        #############################
        
        model_engine, _, _, _ = cramming.load_backend(model, None, tokenizer, cfg.eval, cfg.impl, setup=setup)
        model.eval()
        eval_dataset = eval_datasets[f'{task_name}']
        
        idx = 4
        
        # padding 있는 버전
        labels = eval_dataset[idx]["labels"].unsqueeze(0).to('cuda')
        decoded_sentence = tokenizer.decode(eval_dataset[idx]["input_ids"])
        encoded_inputs = tokenizer([decoded_sentence], padding="max_length", max_length=128, truncation=True, return_tensors="pt")
        data_modified = {'input_ids': encoded_inputs["input_ids"].to('cuda'), 'attention_mask': encoded_inputs["attention_mask"].to('cuda'), 'labels': labels}
         
        outputs, emb_outputs, attentions_outputs, tf_outputs = model(**data_modified)
        outputs['loss'].backward()
        
        # data_modified: the input of the model
        print(f'data_modified: {data_modified}')
        
        # Embedding layer
        print('================= Embedding layer =================')
        print(f'after_word_embedding_grad: {emb_outputs["after_word_emb"].grad.shape}')
        print(f'after_pos_embedding_grad: {emb_outputs["after_pos_emb"].grad.shape}')
        print(f'after_emb_ln_grad: {emb_outputs["after_emb_ln"].grad.shape}')
        print(f'tf_inputs_transpose: {emb_outputs["tf_inputs_transpose"].grad.shape}')
        
        # Transformer layers
        num_layers = len(model.encoder.layers)
        for i in range(num_layers):
            print(f'================= Transformer layer {i} =================')
            print('Transformer intermediate results') 
            print(f'layer {i} transformer input: {model.encoder.layers[i].tf_inputs.grad.shape}')
            print(f'layer {i} after LN: {model.encoder.layers[i].after_ln_before_att.grad.shape}')
            print(f'layer {i} after attention: {model.encoder.layers[i].after_att.grad.shape}')
            print(f'layer {i} after dropout: {model.encoder.layers[i].after_dropout_after_att.grad.shape}')
            print(f'layer {i} after first residual connecton: {model.encoder.layers[i].after_res_conn_after_att.grad.shape}')
            print(f'layer {i} after LN: {model.encoder.layers[i].after_ln_before_ffn.grad.shape}')
            print(f'layer {i} after FFN: {model.encoder.layers[i].after_ffn.grad.shape}')
            print(f'layer {i} after dropout: {model.encoder.layers[i].after_dropout_after_ffn.grad.shape}')
            print(f'layer {i} transformer output: {model.encoder.layers[i].tf_output.grad.shape}')
            print('Attention results')
            print(f'layer {i} input of seqfirst_attention: {model.encoder.layers[i].attn.attention_input.grad.shape}')
            print(f'layer {i} output of seqfirst_attention: {model.encoder.layers[i].attn.attention_output.grad.shape}')
            print(f'layer {i} output of attention (after dense layer): {model.encoder.layers[i].attn.after_dense.grad.shape}')
            print('Seqfirst attention') 
            print(f'layer {i} q: {model.encoder.layers[i].attn.self_attention.q.grad.shape}')
            print(f'layer {i} k: {model.encoder.layers[i].attn.self_attention.k.grad.shape}')
            print(f'layer {i} GK result: {model.encoder.layers[i].attn.self_attention.GK_result.grad.shape}')
            print(f'layer {i} after exp: {model.encoder.layers[i].attn.self_attention.after_exp.grad.shape}')
            print(f'layer {i} after dropout: {model.encoder.layers[i].attn.self_attention.after_dropout.grad.shape}')
            print(f'layer {i} transpose of dropout: {model.encoder.layers[i].attn.self_attention.transpose_of_dropout.grad.shape}')
            print(f'layer {i} v: {model.encoder.layers[i].attn.self_attention.v.grad.shape}')
            print(f'layer {i} GK_v: {model.encoder.layers[i].attn.self_attention.GK_v.grad.shape}')
            print('GK intermediate results')
            print(f'layer {i} GK q: {model.encoder.layers[i].attn.self_attention.GK_q.grad.shape}')
            print(f'layer {i} GK k: {model.encoder.layers[i].attn.self_attention.GK_k.grad.shape}')
            print(f'layer {i} GK matQ_square: {model.encoder.layers[i].attn.self_attention.matQ_square.grad.shape}')
            print(f'layer {i} GK matK_square: {model.encoder.layers[i].attn.self_attention.matK_square.grad.shape}')
        # Final layernorm
        print('Final layernorm')
        print(f'final layernorm: {model.encoder.output.grad.shape}') 
        
        # Pooling layer
        print(f'model.pooler.output.grad: {model.pooler.output.grad.shape}')
        print(f'model.pooler.pooled_output.grad: {model.pooler.pooled_output.grad.shape}')
        print(f'model.pooler.dense_out.grad: {model.pooler.dense_out.grad.shape}')
        print(f'model.pooler.first_token_tensor.grad: {model.pooler.first_token_tensor.grad.shape}')
        print(f'model.pooler.pooler_input.grad: {model.pooler.pooler_input.grad.shape}')
        
        # Last output (logits)
        print(f'model.logits.grad: {model.logits.grad.shape}')
        
        
    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml

@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")

if __name__ == "__main__":
    launch()
