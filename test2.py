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
 
cls = list 

class MyList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.append(1)
        self.append(2)
        self.experts = 3
    
    
a = MyList([1, 2, 3])
print(isinstance(a, cls) and not hasattr(a, "experts"))
