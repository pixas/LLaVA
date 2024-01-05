# from llava.model.builder import load_pretrained_model
# import os 
# model_path="~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b"
# model_base=None
# model_name="llava"
# model_path = os.path.expanduser(model_path)

# tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda")

# for each in model.named_parameters():
#     if "vision_tower" in each[0]:
#         continue
#     print(each[0], each[1].shape)
import json 
import os 
from tqdm import tqdm

path = os.path.expanduser("/remote-home/syjiang/datasets/share_gpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json")
with open(path, 'r') as f:
    data = json.load(f)

image_folder = "/remote-home/syjiang/datasets/share_gpt4v"
image_prefix = set()
for i in tqdm(data):
    # print(i)
    if 'image' in i:
        image_path = os.path.join(image_folder, i['image'])
        # try to open the image with PIL
        try:
            with open(image_path, 'rb') as f:
                pass
        except:
            print(image_path)
            exit()

# print(image_prefix)