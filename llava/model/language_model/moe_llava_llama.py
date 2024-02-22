#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import Any, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import time 
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput, dataclass
from copy import deepcopy
from ..moe_llava_arch import MoELlavaMetaModel, MoELlavaMetaForCausalLM
from transformers.utils import logging
import math 

logger = logging.get_logger(__name__)

@dataclass
class MoEBaseModelOutputWithPast(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    lbl_loss: list = None



class MoELlamaConfig(LlamaConfig):
    model_type = "moe_llama"
    def __init__(self, vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_scaling=None, 
                 num_experts=4, num_experts_per_token=2, is_sparse=True, moe_layer_index=-1, is_eff_moe=False, use_lbl_loss=False, 
                 mix_lora_r=128, mix_lora_alpha=256, lora_dropout=0.0, merge_weights=False,**kwargs):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.is_sparse = is_sparse
        self.moe_layer_index = moe_layer_index
        self.is_eff_moe = is_eff_moe
        self.use_lbl_loss = use_lbl_loss
        self.mix_lora_r = mix_lora_r 
        self.mix_lora_alpha = mix_lora_alpha
        self.lora_dropout = lora_dropout
        self.merge_weights = merge_weights
        super().__init__(vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps, use_cache, pad_token_id, bos_token_id, eos_token_id, pretraining_tp, tie_word_embeddings, rope_scaling, **kwargs)
        

class MoELlavaConfig(MoELlamaConfig):
    model_type = "moe_llava"

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRAModule(nn.Module):
    def __init__(self, in_features, out_features, r):
        super(LoRAModule, self).__init__()
        self.lora_a = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_b = nn.Parameter(torch.zeros((out_features, r)))
        self.reset_parameters()

    def forward(self):
        return self.lora_a.transpose(0, 1) @ self.lora_b.transpose(0, 1)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

class MoLoRALinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        num_experts: int = 4,
        num_experts_per_token: int = 2,
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        use_lbl_loss: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # moe parameters
        self.num_experts = num_experts 
        self.num_experts_per_token = num_experts_per_token
        if num_experts > 1:
            self.switch = nn.Linear(in_features, num_experts)
        self.use_lbl_loss = use_lbl_loss    
        
        # Actual trainable parameters
        if r > 0:
            self.experts = nn.ModuleList([
                nn.ModuleDict({"lora_A_{}".format(i): nn.Linear(in_features, r, False, dtype=torch.float32),
                               "lora_B_{}".format(i): nn.Linear(r, out_features, False, dtype=torch.float32)})
            for i in range(num_experts)])

            # self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        
        if hasattr(self, 'experts'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            for idx, expert in enumerate(self.experts):
                nn.init.kaiming_uniform_(expert[f'lora_A_{idx}'].weight, a=math.sqrt(5))
                nn.init.zeros_(expert[f'lora_B_{idx}'].weight)
            # nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        # if mode:
        #     if self.merge_weights and self.merged:
        #         # Make sure that the weights are not merged
        #         if self.r > 0:
        #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = False
        # else:
        #     if self.merge_weights and not self.merged:
        #         # Merge the weights and mark it
        #         if self.r > 0:
        #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
        #         self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            # sta = time.time()
            result = F.linear(x, T(self.weight), bias=self.bias)
            # sta = time.time()
            if self.use_lbl_loss:
                moe_result, lbl_loss = self.molora_helpder(x)
            else:
                moe_result = self.molora_helpder(x)
            result += moe_result
            # result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)
    
    def molora_helpder(self, x: torch.Tensor):
        if self.num_experts <= 1:
            expert_output = self.experts[0]['lora_B_0'](
                self.experts[0]['lora_A_0'](self.lora_dropout(x))
            ) * self.scaling
            return expert_output
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)       
        gate_logits = self.switch(x)  # [bs * N, expert]
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.zeros((batch_size * N, self.out_features)).to(x) # bs*N, d
        load_balancing_loss = 0
        # compute load balancing loss
        # first compute total token number 
        # then compute the fraction of tokens routed to each expert
        # then compute the mean routing probability
        # then compute the load balancing loss
        if self.training or N > 1:
            for i, expert in enumerate(self.experts):

                batch_idx, nth_expert = torch.where(selected_experts == i) 
                # batch_idx: [bs * N, 1]
                # nth_expert: [bs * N, 1]
                expert_output = expert['lora_B_{}'.format(i)](
                    expert['lora_A_{}'.format(i)](self.lora_dropout(x[batch_idx]))
                ) * self.scaling
                # expert_output = expert(x[batch_idx])
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
                # begin to compute load balancing loss 
                # compute the number of tokens routed to each expert
                # compute the fraction of tokens routed to each expert
                # 选择第i个expert的token数量
                num_per_expert = len(batch_idx)
                # 选择第i个expert的token 比例，对应公式中的f_i
                fraction_per_expert = num_per_expert / (batch_size * N)
                # 选择第i个expert的所有token的概率的均值，对应公式中的P_i
                prob_per_expert = weights[batch_idx, nth_expert, None].mean()
                load_balancing_loss += fraction_per_expert * prob_per_expert
            load_balancing_loss = load_balancing_loss * self.num_experts / (self.num_experts_per_token * self.num_experts_per_token)
        else:
            assert selected_experts.shape[0] == 1
            
            selected_experts = selected_experts.flatten()
            weights = weights.flatten()
            for idx, expert_idx in enumerate(selected_experts):
                results += weights[idx] * expert['lora_B_{}'.format(i)](
                    expert['lora_A_{}'.format(i)](self.lora_dropout(x))
                ) * self.scaling
        
        results = results.contiguous().view(batch_size, N, self.out_features)
        if self.use_lbl_loss:
            return results, load_balancing_loss
        else:
            return results

class MoLoRALlamaMLP(LlamaMLP):
    def __init__(self, config) -> None:
        super(MoLoRALlamaMLP, self).__init__(config)
        self.mix_lora_r = config.mix_lora_r
        self.mix_lora_alpha = config.mix_lora_alpha
        self.lora_dropout = config.lora_dropout
        self.merge_weights = config.merge_weights

        self.gate_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False, r=self.mix_lora_r,
                                    lora_alpha=self.mix_lora_alpha, lora_dropout=self.lora_dropout, merge_weights=self.merge_weights, use_lbl_loss=config.use_lbl_loss, num_experts=config.num_experts, num_experts_per_token=config.num_experts_per_token)
        self.up_proj = MoLoRALinear(self.hidden_size, self.intermediate_size, bias=False, r=self.mix_lora_r,
                                    lora_alpha=self.mix_lora_alpha, lora_dropout=self.lora_dropout, merge_weights=self.merge_weights, use_lbl_loss=config.use_lbl_loss, num_experts=config.num_experts, num_experts_per_token=config.num_experts_per_token)
        self.down_proj = MoLoRALinear(self.intermediate_size, self.hidden_size, bias=False, r=self.mix_lora_r,
                                    lora_alpha=self.mix_lora_alpha, lora_dropout=self.lora_dropout, merge_weights=self.merge_weights, use_lbl_loss=config.use_lbl_loss, num_experts=config.num_experts, num_experts_per_token=config.num_experts_per_token)

        


    

class MoELLamaMLP(nn.Module):
    def __init__(self, config) -> None:
        super(MoELLamaMLP, self).__init__()
        self.use_lbl_loss = config.use_lbl_loss
        self.hidden_size = config.hidden_size 
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts 
        self.num_experts_per_token = config.num_experts_per_token 
        self.is_sparse = config.is_sparse 
        self.is_eff_moe = config.is_eff_moe
        self.experts = nn.ModuleList([LlamaMLP(config) for _ in range(self.num_experts)])
        # if self.is_sparse:
        self.switch = nn.Linear(config.hidden_size, self.num_experts)
        self.score_scale_factor = self.num_experts / self.num_experts_per_token
        
        # self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x = self.fn1(x)
        # x = self.act(x)
        # x = self.dropout(x)
        if self.is_sparse:
            # return self.fast_forward_sparse(x)
            return self.forward_sparse(x)
        else:
            return self.forward_dense(x)
        
    def fast_forward_sparse(self, x: torch.Tensor):
        bsz, N, d = x.shape 
        batch_size = bsz * N 
        x = x.contiguous().view(-1, d)
        gate_logits = self.switch(x)  # [bs * N, expert]
        topK_weights, topK_indices = torch.topk(gate_logits, self.num_experts_per_token)
        topK_scores = F.softmax(topK_weights, dim=-1)  # [bs * N, expert]
        num_selects = topK_indices.shape[1]
        topK_indices = topK_indices.flatten()
        topK_scores = topK_scores.flatten()  # [bs * N * expert]
        # the batch indexes for those chosen experts
        batch_indices = torch.arange(batch_size).to(x.device).repeat_interleave(num_selects)  # [bs * N * select]
        # generate the expert indexes based on the expert order in ascending order
        _, index_sorted_topK_indices = topK_indices.sort(0)
        
        
        sorted_topK_scores = topK_scores.index_select(0, index_sorted_topK_indices)
        sorted_batch_indices = batch_indices.index_select(0, index_sorted_topK_indices)
        expert_batch_size = topK_indices.bincount(minlength=self.num_experts).tolist()
        
        # combine batch for each expert
        sorted_x = x.index_select(0, sorted_batch_indices)
        split_x = torch.split(sorted_x, expert_batch_size, dim=0)
        
        """各专家分别正向传播"""  # 此处应该有并行优化的空间 (如果单次forward不足以占满显卡利用率)
        # args = [(split_x[i], i) for i in range(self.num_experts) if split_x[i].shape[0] > 0]
        # expert_outputs = self.experts_vmap(args)
        expert_outputs = [self.experts[i](split_x[i]) for i in range(self.num_experts) if split_x[i].shape[0] > 0]

        """重组各个专家的输出，并进行加权"""
        # (bsz*seq_len*num_selects, hidden_size)
        cat_expert_outputs = torch.cat(expert_outputs, 0)  # 拼接专家输出
        output_dim = cat_expert_outputs.size(1)
        cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor)  # 乘权重
        zeros = torch.zeros(batch_size, output_dim).to(x)
        y = zeros.index_add(0, sorted_batch_indices, cat_expert_outputs)
        y = y.reshape(bsz, N, output_dim)
        return y, 0
        # if self.multiply_gate_scores:
        #     if self.mlp_norm is None:
        #         cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * self.score_scale_factor)  # 乘权重
        #         # cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1) * 1.0)  # 乘权重
        #     else:
        #         cat_expert_outputs = torch.mul(cat_expert_outputs, sorted_topK_scores.reshape(-1, 1))  # 乘权重
        #         cat_expert_outputs = self.mlp_norm(cat_expert_outputs)

    def forward_sparse(self, x: torch.Tensor):
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)         
        gate_logits = self.switch(x)  # [bs * N, expert]
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.zeros((batch_size * N, self.hidden_size)).to(x) # bs*N, d
        load_balancing_loss = 0
        # compute load balancing loss
        # first compute total token number 
        # then compute the fraction of tokens routed to each expert
        # then compute the mean routing probability
        # then compute the load balancing loss
        if self.training or N > 1:
            for i, expert in enumerate(self.experts):

                batch_idx, nth_expert = torch.where(selected_experts == i) 
                # batch_idx: [bs * N, 1]
                # nth_expert: [bs * N, 1]
                expert_output = expert(x[batch_idx])
                if self.is_eff_moe:
                    expert_output *= (self.num_experts / self.num_experts_per_token)
                results[batch_idx] += weights[batch_idx, nth_expert, None] * expert_output
                # begin to compute load balancing loss 
                # compute the number of tokens routed to each expert
                # compute the fraction of tokens routed to each expert
                # 选择第i个expert的token数量
                num_per_expert = len(batch_idx)
                # 选择第i个expert的token 比例，对应公式中的f_i
                fraction_per_expert = num_per_expert / (batch_size * N)
                # 选择第i个expert的所有token的概率的均值，对应公式中的P_i
                prob_per_expert = weights[batch_idx, nth_expert, None].mean()
                load_balancing_loss += fraction_per_expert * prob_per_expert
            load_balancing_loss = load_balancing_loss * self.num_experts / (self.num_experts_per_token * self.num_experts_per_token)
        else:
            assert selected_experts.shape[0] == 1
            
            selected_experts = selected_experts.flatten()
            weights = weights.flatten()
            for idx, expert_idx in enumerate(selected_experts):
                results += weights[idx] * self.experts[expert_idx](x)
        
        results = results.contiguous().view(batch_size, N, self.hidden_size)
        if self.use_lbl_loss:
            return results, load_balancing_loss
        else:
            return results

    
    def forward_dense(self, x: torch.Tensor):
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)         
        gate_logits = self.switch(x)  # [bs * N, expert]
        weights = F.softmax(gate_logits, dim=-1)  # [bs * N, expert]
        results = torch.zeros((batch_size * N, self.hidden_size)).to(x) # bs*N, d
        for i, expert in enumerate(self.experts):
            results += weights[:, i, None] * expert(x)
            
        
        results = results.contiguous().view(batch_size, N, self.hidden_size)
        return results, 0

class MoELLamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.mlp = MoLoRALlamaMLP(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_value: Tuple[torch.Tensor] | None = None, output_attentions: bool | None = False, use_cache: bool | None = False) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        # sta = time.time()
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # sta = time.time()
        if self.config.use_lbl_loss:
            hidden_states, lbl_loss = self.mlp(hidden_states)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)
        if self.config.use_lbl_loss:
            return outputs, lbl_loss
        else:
            return outputs
        


class MoELlamaModel(LlamaModel):
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        self.config = config
        # only first some layers are processed with MoE
        self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
        # if config.moe_layer_index == -1:
        #     self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
        # else:
        #     moe_layer_index = config.moe_layer_index 
        #     half_layer_index = moe_layer_index // 2
        #     self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) if i < moe_layer_index else LlamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
        #     self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) if i < half_layer_index or i > config.num_hidden_layers - half_layer_index else LlamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
            # self.layers = nn.ModuleList([])
            # for i in range(config.num_hidden_layers):
            #     cur_config = deepcopy(config)
            #     if i < 16:
            #         cur_config.num_experts = 2
            #     elif i < 24:
            #         cur_config.num_experts = 3
            #     else:
            #         cur_config.num_experts = 4
                
            #     self.layers.append(MoELLamaDecoderLayer(cur_config))
                
        self.post_init()
    
    def forward(self, input_ids: torch.LongTensor = None, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_values: List[torch.FloatTensor] | None = None, inputs_embeds: torch.FloatTensor | None = None, use_cache: bool | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None) -> Tuple | MoEBaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        lbl_loss_total = [0 for i in range(self.config.num_hidden_layers)]
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None
            lbl_loss = 0
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward
                if isinstance(decoder_layer, MoELLamaDecoderLayer) and self.config.use_lbl_loss:
                    layer_outputs, lbl_loss = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
                else:
                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(decoder_layer),
                        hidden_states,
                        attention_mask,
                        position_ids,
                        None,
                    )
            else:
                if isinstance(decoder_layer, MoELLamaDecoderLayer) and self.config.use_lbl_loss:
                    layer_outputs, lbl_loss = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                    )
                

            hidden_states = layer_outputs[0]
            lbl_loss_total[idx] = lbl_loss
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
       

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, lbl_loss_total] if v is not None)
        return MoEBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            lbl_loss=lbl_loss_total
        )
        
class MoELlavaLlamaModel(MoELlavaMetaModel, MoELlamaModel):
    config_class = MoELlavaConfig
    
    def __init__(self, config: MoELlamaConfig):
        super(MoELlavaLlamaModel, self).__init__(config)

class MoELlamaForCausalLM(LlamaForCausalLM):
    config_class = MoELlamaConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = MoELlamaModel(config)
        # self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # self.load_balancing_loss_ceof = 0.01
        # Initialize weights and apply final processing
        self.post_init()

class MoELlavaLlamaForCausalLM(MoELlamaForCausalLM, MoELlavaMetaForCausalLM):
    config_class = MoELlavaConfig
    
    def __init__(self, config):
        super(MoELlamaForCausalLM, self).__init__(config)
        self.model = MoELlavaLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.load_balancing_loss_ceof = 0.01
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, expert_info = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # sta = time.time()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        # print("language model:", time.time() - sta)

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if getattr(outputs, "lbl_loss", None) is not None:
                if outputs.lbl_loss [0] != 0:
                    lbl_loss = outputs.lbl_loss
                    loss = loss + sum(lbl_loss) * self.load_balancing_loss_ceof 
                
            if expert_info is not None:
                counts, route_prob, n_dropped, route_prob_max = list(expert_info.values())
                total = counts.sum(dim=-1, keepdims=True)
                n_experts = counts.shape[0]
                # Fraction of tokens routed to each expert
                # $$f_i = \frac{1}{T} \sum_{x \in \mathscr{B}} \mathbf{1} \{ \mathop{argmax} p(x), i \}$$
                # $f_i$ is the count of tokens where the argmax of $p(x)$ is equal to $i$.
                route_frac = counts / total
                # Mean routing probability
                # $$P_i = \frac{1}{T} \sum_{x \in \mathscr{B}} p_i (x)$$
                route_prob = route_prob / total
                # Load balancing loss
                # $$\mathscr{L} = N \sum_{i=1}^N f_i \cdot P_i$$
                # $\mathscr{L}$ is the loss for a single layer and here we are
                # taking the sum of losses across all layers.
                load_balancing_loss = n_experts * (route_frac * route_prob).sum()
                loss = loss + self.load_balancing_loss_ceof * load_balancing_loss
                
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs



AutoConfig.register("moe_llama", MoELlamaConfig)
AutoConfig.register("moe_llava", MoELlavaConfig)
AutoModelForCausalLM.register(MoELlamaConfig, MoELlamaForCausalLM)
AutoModelForCausalLM.register(MoELlavaConfig, MoELlavaLlamaForCausalLM)
