# from llava.model.utils import convert_state_dict
# import torch 
# import os 
# model_name_or_path = "/remote-home/share/models/vicuna-7b-v1.5"
# ckpt = {}
# for each_ckpt in os.listdir(model_name_or_path):
#     if each_ckpt.endswith(".bin"):
#         ckpt.update(torch.load(os.path.join(model_name_or_path, each_ckpt), map_location='cpu'))

# for key, value in ckpt.items():
#     # print(key)
#     # print(value.shape)
#     # print(value.dtype)
#     # print("===")
#     print(key, value.shape, value.dtype)
import torch
import torch.nn.functional as F  
import torch.nn as nn 
import os 
import logging 
import transformers 
from llava.model import MoELlavaConfig, MoELlavaLlamaForCausalLM
from peft import get_peft_model, LoraConfig
from llava.model.language_model.moe_llava_llama import get_mixoflora_model, MoELLamaDecoderLayer

torch.set_default_device("cuda")
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

model_state_dict_path = "/remote-home/yushengliao/syjiang/checkpoints/llava-v1.5-7b-moe-molora-4x2-trial-nolbl-lora/non_lora_trainables.bin"
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k or ("lora_" in k and "experts" in k)}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

# state_dict = torch.load(os.path.expanduser(model_state_dict_path), map_location='cpu')
# for k, v in state_dict.items():
#     print(k)
model_path = "/remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5"
config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
moe_config = MoELlavaConfig(**config.to_dict())
moe_config.moe_layer_index = -1
moe_config.num_experts = 4
moe_config.num_experts_per_token = 2

moe_config.architectures = ["MoELLamaForCausalLM"]


ckpt = {}
# for each_ckpt in os.listdir(model_args.model_name_or_path):
#     if each_ckpt.endswith(".bin"):
#         ckpt.update(torch.load(os.path.join(model_args.model_name_or_path, each_ckpt), map_location='cpu'))
# # please obtain all submodule (recursively) of MoELlavaLlamaForCausalLM
# model_state_dict = set(list(MoELlavaLlamaForCausalLM(moe_config).state_dict().keys()))

# if model_args.is_eff_moe:
#     new_state_dict = convert_eff_state_dict(model_state_dict, ckpt, model_args.num_experts)
# else:
#     new_state_dict = convert_state_dict(model_state_dict, ckpt, model_args.num_experts)
# model = MoELlavaLlamaForCausalLM.from_pretrained(
#     model_args.model_name_or_path,
#     config=moe_config,
#     cache_dir=training_args.cache_dir,
#     state_dict=new_state_dict,
#     **bnb_model_from_pretrained_args
# )
model = MoELlavaLlamaForCausalLM.from_pretrained(
    model_path,
    config=moe_config,
)

def find_all_linear_names(model, wrap_projector=False):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'switch']
    if wrap_projector:
        multimodal_keywords.remove("mm_projector")
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(model.get_submodule(".".join(name.split(".")[:-2])), MoELLamaDecoderLayer) and isinstance(module, cls) and "mlp" in name:
            continue
        # if isinstance(module, cls) and "experts" in name:
        #     continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=128,
    lora_alpha=256,
    target_modules=find_all_linear_names(model, False),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

        # for name, param in model.named_parameters():
        #     if "lora" not in name:
        #         param.data = param.data.to(torch.bfloat16)
        #     else:
        #         param.data = param.data.to(torch.float32)
            
model.to(torch.bfloat16)


model = get_peft_model(model, lora_config)

model = get_mixoflora_model(model, 4, 2, lora_config=lora_config, inference_mode=False)

state_dict = get_peft_state_maybe_zero_3(
    model.named_parameters(), "none"
)
non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    model.named_parameters()
)

print("lora-params:")
for k, v in state_dict.items():
    print(k)
    
print("Non-lora-trainables:")
for k, v in non_lora_state_dict.items():
    print(k)