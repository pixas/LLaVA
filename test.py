import torch
from transformers import BertConfig
def foo(x=0):
    li = [1, 2, 3]
    return 0, *li 

print(foo())