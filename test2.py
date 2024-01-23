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

        