import torch 
model_path = "/remote-home/yushengliao/syjiang/checkpoints/llava-v1.5-7b-moe-162-243-324-2-nolbl-lora/non_lora_trainables.bin"

state_dict = torch.load(model_path)

for k, v in state_dict.items():
    print(k)