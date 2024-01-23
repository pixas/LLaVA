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

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaMLP

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput, dataclass

from ..moe_llava_arch import MoELlavaMetaModel, MoELlavaMetaForCausalLM
from transformers.utils import logging

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
                 num_experts=4, num_experts_per_token=2, is_sparse=True, moe_layer_index=-1, is_eff_moe=False, **kwargs):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.is_sparse = is_sparse
        self.moe_layer_index = moe_layer_index
        self.is_eff_moe = is_eff_moe
        super().__init__(vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads, num_key_value_heads, hidden_act, max_position_embeddings, initializer_range, rms_norm_eps, use_cache, pad_token_id, bos_token_id, eos_token_id, pretraining_tp, tie_word_embeddings, rope_scaling, **kwargs)
        

class MoELlavaConfig(MoELlamaConfig):
    model_type = "moe_llava"

class MoELLamaMLP(nn.Module):
    def __init__(self, config) -> None:
        super(MoELLamaMLP, self).__init__()
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
            return self.fast_forward_sparse(x)
            # return self.forward_sparse(x)
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
        
        # compute load balancing loss
        # first compute total token number 
        # then compute the fraction of tokens routed to each expert
        # then compute the mean routing probability
        # then compute the load balancing loss
        load_balancing_loss = 0
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
            num_per_expert = len(batch_idx)
            fraction_per_expert = num_per_expert / (batch_size * N)
            prob_per_expert = weights[batch_idx, nth_expert, None].mean()
            load_balancing_loss += fraction_per_expert * prob_per_expert

        
        results = results.contiguous().view(batch_size, N, self.hidden_size)
        return results, load_balancing_loss
    
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
        
        self.mlp = MoELLamaMLP(config)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, position_ids: torch.LongTensor | None = None, past_key_value: Tuple[torch.Tensor] | None = None, output_attentions: bool | None = False, use_cache: bool | None = False) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
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
        hidden_states, lbl_loss = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, lbl_loss

class MoELlamaModel(LlamaModel):
    def __init__(self, config: MoELlamaConfig):
        super().__init__(config)
        
        # only first some layers are processed with MoE
        if config.moe_layer_index == -1:
            self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
        else:
            moe_layer_index = config.moe_layer_index 
            self.layers = nn.ModuleList([MoELLamaDecoderLayer(config) if i < moe_layer_index else LlamaDecoderLayer(config) for i in range(config.num_hidden_layers)])
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

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, None)

                    return custom_forward

                layer_outputs, lbl_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs, lbl_loss = decoder_layer(
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
            # if getattr(outputs, "lbl_loss", None) is not None:
            #     lbl_loss = outputs.lbl_loss
            #     loss = loss + sum(lbl_loss) * self.load_balancing_loss_ceof 
                
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
