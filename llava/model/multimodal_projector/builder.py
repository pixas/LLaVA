import torch
import torch.nn as nn
import re

from llava.model.Qformer_utils import BertConfig, BertLMHeadModel

from transformers import BertTokenizer

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class GatedLinear(nn.Module):
    def __init__(self, mm_hidden_size, channels, n_gates):
        super(GatedLinear, self).__init__()
        self.pre_norm = nn.LayerNorm(mm_hidden_size)
        self.mm_hidden_size = mm_hidden_size
        self.n_gates = n_gates 
        self.gate_weights = nn.Linear(24 * 24 * mm_hidden_size, self.n_gates)
        self.projs = nn.ModuleList([nn.Linear(mm_hidden_size, channels) for i in range(self.n_gates)])
    
    def forward(self, x):
        # print(x.shape)
        x = self.pre_norm(x)
        weight_x = x.reshape(-1, 24 * 24 * x.shape[-1])  # [bs, 24 * 24 * dim]
        x = x.reshape(-1, 24 * 24, self.mm_hidden_size)  # [bs, 24 * 24, dim]
        
        weights = self.gate_weights(weight_x).unsqueeze(-1).unsqueeze(-1)  # [b, C, 1, 1]
        each_feat = [self.projs[i](x).unsqueeze(1) for i in range(self.n_gates)]
        total_feat = torch.cat(each_feat, dim=1)
        averaged_feat = (total_feat * weights).sum(dim=1)
        return averaged_feat

class FeedForward(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=4, *args, **kwargs) -> None:
        super(FeedForward, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.GELU()
        self.fn1 = nn.Linear(self.in_channels, self.out_channels * scale_factor)
        self.fn2 = nn.Linear(self.out_channels * scale_factor, self.out_channels)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fn1(x)
        x = self.act(x)
        x = self.dropout(x)
        return self.fn2(x)

class SwitchLinear(nn.Module):
    def __init__(self, mm_hidden_size, channels, n_experts, capacity_factor=1.5, drop_tokens=True, is_scale_prob=True):
        super(SwitchLinear, self).__init__()
        self.pre_norm = nn.LayerNorm(mm_hidden_size)
        self.pre_linear = nn.Linear(mm_hidden_size, channels)
        self.gelu = nn.GELU()
        self.mm_hidden_size = mm_hidden_size
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens
        self.capacity_factor=capacity_factor
        self.experts = nn.ModuleList([FeedForward(channels, channels) for i in range(self.n_experts)])
        self.switch = nn.Linear(channels, n_experts)
        self.channels = channels
        self.softmax = nn.Softmax(-1)
        
        self.is_scale_prob = is_scale_prob
    
    def forward(self, x: torch.Tensor):
        x = self.gelu(self.pre_linear(self.pre_norm(x)))
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)         
        route_prob = self.softmax(self.switch(x))  # [bs * N, expert]
        
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        final_output = x.new_zeros(x.shape)
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        
        # how many tokens are routed to ith expert
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])
        
        dropped = []
        
        if self.drop_tokens:
            for i in range(self.n_experts):
                if len(indexes_list[i]) < capacity:
                    continue
                
                # drop tokens 
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]
        
        
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]
        
        for i in range(self.n_experts):
            final_output[indexes_list[i], :] = expert_output[i]
        
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]
        
        if self.is_scale_prob:
            final_output = final_output * route_prob_max.contiguous().view(-1, 1)
        else:
            final_output = final_output * (route_prob_max / route_prob_max.detach()).contiguous().view(-1, 1)
        
        final_output = final_output.contiguous().view(batch_size, N, d)
        
        return final_output, counts, route_prob.sum(0), len(dropped), route_prob_max

class ECSwitchLinear(SwitchLinear):
    def __init__(self, mm_hidden_size, channels, n_experts, capacity_factor=2, drop_tokens=True, is_scale_prob=True):
        super().__init__(mm_hidden_size, channels, n_experts, capacity_factor, drop_tokens, is_scale_prob)
        self.switch = nn.Linear(mm_hidden_size, n_experts)
        self.pre_linear = None
        self.gelu = None
        self.mm_hidden_size = mm_hidden_size

        self.experts = nn.ModuleList([FeedForward(mm_hidden_size, channels) for i in range(self.n_experts)])
        # self.switch = nn.Linear(channels, n_experts)
        # self.softmax = nn.Softmax(-1)
        
        self.is_scale_prob = is_scale_prob
    def forward(self, x: torch.Tensor):
        # x = self.gelu(self.pre_linear(self.pre_norm(x)))
        x = self.pre_norm(x)
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)
        n = x.shape[0]
        k = int(self.capacity_factor * n / self.n_experts)
        route_prob = self.softmax(self.switch(x))  # [bs * N, expert]
        # e: number of experts
        # n: number of tokens
        # k: number of selected tokens for each expert 
        G, I  = torch.topk(route_prob.transpose(-1, -2), k)
        # G: [e, k] G[i, j] means ith expert choose kth token as G[i, j] weight
        # I: [e, k] means the ith expert chooses kth token 
        # P = torch
        final_output = x.new_zeros((n, self.channels))
        # print(G.shape, I.shape)
        # print(final_output.shape, x.shape, route_prob.shape)
        X_in = x[I]  # [e, k, d]
        expert_output = []
        for i in range(self.n_experts):
            expert_output.append(self.experts[i](X_in[i]))
        
        expert_output = torch.stack(expert_output, 0)
        expert_output = expert_output * G.unsqueeze(-1)
        final_output.index_add_(0, I.contiguous().view(-1), expert_output.contiguous().view(-1, self.channels))
        final_output = final_output.contiguous().view(batch_size, N, -1)
        return final_output

class Qformer(nn.Module):
    def __init__(self, in_channels, out_channels, num_query_token=32, cross_attention_freq=2, qformer_text_input=True,
                 max_txt_len=128, qformer_use_pretrained=False):
        super(Qformer, self).__init__()
        bert_channel = 768
        self.in_proj = nn.Linear(
            in_channels, bert_channel
        )
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = bert_channel
        encoder_config.n_experts = 0
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        if qformer_use_pretrained:
            self.Qformer = BertLMHeadModel.from_pretrained(
                "bert-base-uncased", config=encoder_config
            )
        else:
            self.Qformer = BertLMHeadModel(config=encoder_config)
        # self.Qformer.bert.embeddings.position_ids=None
        # for name, param in self.Qformer.named_parameters():
        #     print(name, param.shape, param)
        # print(encoder_config)
        self.query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        self.query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        self.qformer_text_input = qformer_text_input
        if not qformer_text_input:
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
        else:
            self.Qformer.resize_token_embeddings(len(self.tokenizer))
        self.Qformer.cls = None 
        
        self.out_channels = out_channels
        self.llm_proj = nn.Linear(
            self.Qformer.config.hidden_size, out_channels
        )
        self.max_txt_len = max_txt_len
        
    def forward(self, image_features, text=None):
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)
        image_features = self.in_proj(image_features)
        query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_features.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_features.device)
            # query_atts = query_atts.repeat([image_features.shape[0], 1])
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        query_output = query_output['bert_output']
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:self.query_tokens.size(1),:])
        return inputs_llm

class MoEQformer(Qformer):
    def __init__(self, in_channels, out_channels, num_query_token=32, cross_attention_freq=2, qformer_text_input=True, max_txt_len=128, n_experts=4):
        super().__init__(in_channels, out_channels, num_query_token, cross_attention_freq, qformer_text_input, max_txt_len)
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        bert_channel = 768
        encoder_config.encoder_width = bert_channel
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = cross_attention_freq
        encoder_config.query_length = num_query_token
        encoder_config.n_experts = n_experts
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", truncation_side="left")
        self.tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        # self.Qformer = BertLMHeadModel.from_pretrained(
        #     "bert-base-uncased", config=encoder_config
        # )
        self.Qformer = BertLMHeadModel(config=encoder_config)
        
    def forward(self, image_features, text=None):
        image_atts = torch.ones(image_features.size()[:-1], dtype=torch.long).to(image_features.device)
        image_features = self.in_proj(image_features)
        query_tokens = self.query_tokens.expand(image_features.shape[0], -1, -1)
        if self.qformer_text_input:
            text_Qformer = self.tokenizer(
                text,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(image_features.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image_features.device)
            # query_atts = query_atts.repeat([image_features.shape[0], 1])
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

            query_output, moe_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output, moe_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        inputs_llm = self.llm_proj(query_output.last_hidden_state[:,:self.query_tokens.size(1),:])
        return inputs_llm, *moe_output

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == 'gated_linear':
        return GatedLinear(config.mm_hidden_size, config.hidden_size, config.mm_projector_gates)

    if projector_type == 'moe':
        return SwitchLinear(config.mm_hidden_size, config.hidden_size, config.mm_projector_experts)

    if projector_type == 'ec_moe':
        return ECSwitchLinear(config.mm_hidden_size, config.hidden_size, config.mm_projector_experts)
    
    if projector_type == 'qformer':
        return Qformer(config.mm_hidden_size, config.hidden_size, qformer_text_input=config.qformer_text_input, qformer_use_pretrained=config.qformer_use_pretrained)
    
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

if __name__ == "__main__":
    mm_projector = Qformer(1024, 4096, qformer_text_input=True)
    for name, param in mm_projector.named_parameters():
        print(name, param.shape)