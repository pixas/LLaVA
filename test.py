import torch 

import torch.nn as nn 
import torch.nn.functional as F 

num_experts = 4
# x = torch.randn(24, 32) + torch.stack([torch.ones(32) * i for i in range(24)])
# route_weight = nn.Linear(32, num_experts)
# route_prob = torch.softmax(route_weight(x), dim=-1)
# experts = [nn.Linear(32, 32) for _ in range(num_experts)]
# # [N, 2]
# top_max_route_prob, routes = torch.topk(route_prob, 2, dim=-1)
# indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0]  for i in range(num_experts)]
# another_list = [torch.eq(routes.reshape(-1), i).nonzero(as_tuple=True)[0] // 2 for i in range(num_experts)]
# print(indexes_list)
# print(another_list)
class LLaMAFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4, *args, **kwargs) -> None:
        super(LLaMAFeedForward, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.SiLU()
        self.fn1 = nn.Linear(self.in_channels, int(self.in_channels * scale_factor))
        self.fn2 = nn.Linear(int(self.in_channels * scale_factor), self.out_channels)
        self.fn3 = nn.Linear(self.in_channels, int(self.in_channels * scale_factor))
        
        
        # self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x = self.fn1(x)
        # x = self.act(x)
        # x = self.dropout(x)
        
        x = self.fn2(self.act(self.fn1(x)) * self.fn3(x))
        
        return x    
class SwitchLinear(nn.Module):
    def __init__(self, mm_hidden_size, channels, n_experts,
                 num_experts_per_token=1, use_balancing_loss=True):
        super(SwitchLinear, self).__init__()
        # self.pre_norm = nn.LayerNorm(mm_hidden_size)
        # self.pre_linear = nn.Linear(mm_hidden_size, channels)
        # self.gelu = nn.GELU()
        self.mm_hidden_size = mm_hidden_size
        self.n_experts = n_experts

        self.experts = nn.ModuleList([LLaMAFeedForward(self.mm_hidden_size, channels) for i in range(self.n_experts)])
        self.switch = nn.Linear(self.mm_hidden_size, n_experts)
        self.channels = channels
        self.softmax = nn.Softmax(-1)
        self.num_experts_per_token = num_experts_per_token
        
        self.use_balancing_loss = use_balancing_loss
    
    def forward(self, x: torch.Tensor):
        # x = self.gelu(self.pre_linear(self.pre_norm(x)))
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)         
        gate_logits = self.switch(x)  # [bs * N, expert]
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        weights = self.softmax(weights)  # [bs * N, expert]
        results = torch.zeros((batch_size * N, self.channels)) # bs*N, d
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i) 
            # batch_idx: [bs * N, 1]
            # nth_expert: [bs * N, 1]
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                x[batch_idx]
            )
        
        results = results.contiguous().view(batch_size, N, self.channels)
        return results

if __name__ == "__main__":
    x = torch.randn(2, 3, 4) # [bs, N, d]
    linear = SwitchLinear(4, 5, n_experts=num_experts, num_experts_per_token=2)
    print(linear(x).shape)