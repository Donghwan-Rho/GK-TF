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
    # print(f'tasks: {tasks}')
    metrics = dict()
    stats = defaultdict(list)
    
    for task_name, task in tasks.items():
        # print(colored('task_name: {}'.format(task_name), 'red'))
        # print(colored('task: {}'.format(task), 'red'))
        cfg.eval.steps = len(task["trainloader"]) * cfg.eval.epochs
        # log.info(f"Sentense test eval with padding size 128")
        # log.info(f"Finetuning task {task_name} with {task['num_classes']} classes for {cfg.eval.steps} steps.")
        
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
        
        # 2 layers 모델의 weight를 받은 후 첫 layer의 attention만 가져옴
        first_attention = model.encoder.layers[0].attn
        print(f'first_attention: {first_attention}')
        
        # Input: shape (128, 1, 768)이 되게 하면 됩니다.
        # 예시에서는 random tensor로 설정했는데 원하는 tensor로 설정하시면 됩니다.
        input = torch.randn(128, 1, 768).to('cuda').requires_grad_(True)
        print(f'input: {input.shape}')
        attention_mask = torch.randn(1, 1, 1, 128).to('cuda')
        
        output, _, _ = first_attention(input, attention_mask)
        print(f'output: {output.shape}')
        
        # loss.backward()를 진행해야 gradient가 계산되기 때문에 임의로 간단한 loss를 설정함
        loss = output.sum()
        loss.backward()
        
        print('====== Start Attention layer ======')
        print(f'inputs of seqfirst_attention: {first_attention.attention_input.grad.shape}')
        print('====== Start Seqfirstattention ======')
        self_attention = first_attention.self_attention
        print(f'Seqfirstselfattention Q: {self_attention.q.grad.shape}')
        print(f'Seqfirstselfattention K: {self_attention.k.grad.shape}')
        print(f'Seqfirstselfattention matQ_square: {self_attention.matQ_square.grad.shape}')
        print(f'Seqfirstselfattention matK_square: {self_attention.matK_square.grad.shape}')
        print(f'Seqfirstselfattention GK result: {self_attention.GK_result.grad.shape}')
        print(f'Seqfirstselfattention after exp: {self_attention.after_exp.grad.shape}')
        print(f'Seqfirstselfattention after dropout: {self_attention.after_dropout.grad.shape}')
        print(f'Seqfirstselfattention reshaping of dropout: {self_attention.transpose_of_dropout.grad.shape}')
        print(f'Seqfirstselfattention V: {self_attention.v.grad.shape}')
        print(f'Seqfirstselfattention GK x V (output of attention): {self_attention.GK_v.grad.shape}')
        print('====== Start Seqfirstattention ======')
        print(f'attention output (reshape of GK x V): {first_attention.attention_output.grad.shape}')
        print(f'final output (after dense): {first_attention.after_dense.grad.shape}')
        print('====== End Attention layer ======')
        
    # Save to summary:
    if cramming.utils.is_main_process():
        cramming.utils.save_summary("downstream", cfg, stats, time.time() - local_time, setup)
    return metrics  # will be dumped into yaml

@hydra.main(config_path="cramming/config", config_name="cfg_eval", version_base="1.1")
def launch(cfg):
    cramming.utils.main_launcher(cfg, main_downstream_process, job_name="downstream finetuning")

if __name__ == "__main__":
    launch()
