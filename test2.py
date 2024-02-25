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

model_state_dict_path = "~/syjiang/checkpoints/llava-v1.5-7b-moe-molora-4x2-trial-nolbl-lora/checkpoint-100/adapter_model.bin"

state_dict = torch.load(os.path.expanduser(model_state_dict_path), map_location='cpu')
for k, v in state_dict.items():
    print(k)