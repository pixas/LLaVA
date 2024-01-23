<<<<<<< HEAD
from llava.model.language_model.moe_llava_llama import MoELlamaModel, MoELlamaConfig
from transformers import LlamaConfig, LlamaModel
from llava.model import MoELlavaLlamaForCausalLM, MoELlavaConfig
import transformers
import os 
# new_config = MoELlamaConfig(moe_layer_index=4, num_hidden_layers=8, hidden_size=32, intermediate_size=92, num_attention_heads=8)

# model = MoELlamaModel(new_config)

# # for name, param in model.named_parameters():
# #     print(name, param.shape)

# llama_config = LlamaConfig(num_hidden_layers=8, hidden_size=32, intermediate_size=92, num_attention_heads=8)
import torch
# llama_model = LlamaModel(llama_config)

# model.load_state_dict(llama_model.state_dict())
path = "/remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5"

for each in os.listdir(path):


    print(each)
    try:
        with open(os.path.join(path, each), 'r') as f:
            print(f.read())
    except:
        continue
    print("=" * 100)

        
=======
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
 
input = torch.tensor([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
counts = torch.bincount(input)
 
print(counts)  # 输出: tensor([0, 1, 2, 3, 4])
print(torch.bincount(input, minlength=2))
>>>>>>> a64d62495c5a9e8f95ffd40d657997c45f294f53
