import torch 

import torch.nn as nn 
import torch.nn.functional as F 

num_experts = 4
x = torch.randn(24, 32) + torch.stack([torch.ones(32) * i for i in range(24)])
route_weight = nn.Linear(32, num_experts)
route_prob = torch.softmax(route_weight(x), dim=-1)
experts = [nn.Linear(32, 32) for _ in range(num_experts)]
# [N, 2]
top_max_route_prob, routes = torch.topk(route_prob, 2, dim=-1)
indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0]  for i in range(num_experts)]
another_list = [torch.eq(routes.reshape(-1), i).nonzero(as_tuple=True)[0] // 2 for i in range(num_experts)]
print(indexes_list)
print(another_list)