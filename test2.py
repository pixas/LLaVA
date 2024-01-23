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