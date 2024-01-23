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

config = transformers.AutoConfig.from_pretrained(path, trust_remote_code=True)
moe_config = MoELlavaConfig(**config.to_dict())
moe_config.moe_layer_index = 1
moe_config.num_experts = 2 
moe_config.num_experts_per_token = 1
moe_config.is_sparse = True
moe_config.architectures = ["MoELLamaForCausalLM"]

model = MoELlavaLlamaForCausalLM.from_pretrained(
    path,
    config=moe_config,
    cache_dir=None,
)

# model.from_pretrained()
ckpt = {}
for each_ckpt in os.listdir(path):
    if each_ckpt.endswith(".bin"):
        print(each_ckpt)
        ckpt.update(torch.load(os.path.join(path, each_ckpt)))
# for name, param in ckpt.items():
#     if "experts" in name:
#         print(name, param)
def convert_state_dict(current_state_dict, state_dict, num_experts, prefix='model.'):
    new_state_dict = {}
    print("Begin to load state dict")
    print([key for key in current_state_dict if "experts" in key])
    # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    # layers.{index}.mlp.experts.{expert_index}.param if it exists
    # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    # please supplement the code below
    for name, param in state_dict.items():
        # Check if the parameter belongs to an MoE layer
        if name.startswith(prefix + "layers.") and ".mlp." in name:
            layer_index = int(name.split('.')[2])
            param_name = name.split('.mlp.')[1]

            # Find the corresponding MoE parameter

            moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
            if prefix + moe_param_name in current_state_dict:
                for expert_index in range(num_experts):
                    moe_param_name = prefix + f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"

                    new_state_dict[moe_param_name] = param

            else:
                new_state_dict[name] = param
        else:
            new_state_dict[name] = param
    return new_state_dict
new_state_dict = convert_state_dict(model.state_dict(), ckpt, 2)

err_msg = model.load_state_dict(new_state_dict, False)
language_model = model.get_model()
language_model.save_pretrained("/remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5-moe")
language_model.tokenizer.save_pretrained("/remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5-moe")
# print(err_msg)
# assert torch.equal(model.state_dict()["model.layers.0.mlp.experts.1.gate_proj.weight"], ckpt["model.layers.0.mlp.gate_proj.weight"])
# assert torch.equal(model.state_dict()["model.layers.0.mlp.experts.0.gate_proj.weight"], ckpt["model.layers.0.mlp.gate_proj.weight"])
# print(list(model.state_dict().keys()))

# for name, param in model.named_parameters():
#     if "experts" in name:
#         print(name, param)