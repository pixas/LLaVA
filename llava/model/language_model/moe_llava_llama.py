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

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..moe_llava_arch import MoELlavaMetaModel, MoELlavaMetaForCausalLM


class MoELlamaConfig(LlamaConfig):
    model_type = "moe_llama"
    def __init__(self, vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=None, hidden_act="silu", max_position_embeddings=2048, initializer_range=0.02, rms_norm_eps=0.000001, use_cache=True, pad_token_id=0, bos_token_id=1, eos_token_id=2, pretraining_tp=1, tie_word_embeddings=False, rope_scaling=None, 
                 num_experts=4, num_experts_per_token=2, is_sparse=True, moe_layer_index=-1, **kwargs):
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.is_sparse = is_sparse
        self.moe_layer_index = moe_layer_index
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
        
        self.experts = nn.ModuleList([LlamaMLP(config) for _ in range(self.num_experts)])
        # if self.is_sparse:
        self.switch = nn.Linear(config.hidden_size, self.num_experts)

        
        # self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x = self.fn1(x)
        # x = self.act(x)
        # x = self.dropout(x)
        if self.is_sparse:
            return self.forward_sparse(x)
        else:
            return self.forward_dense(x)
        


    def forward_sparse(self, x: torch.Tensor):
        batch_size, N, d = x.shape 
        x = x.contiguous().view(-1, d)         
        gate_logits = self.switch(x)  # [bs * N, expert]
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        weights = F.softmax(weights, dim=-1)  # [bs * N, expert]
        results = torch.zeros((batch_size * N, self.hidden_size)).to(x) # bs*N, d
        for i, expert in enumerate(self.experts):

            batch_idx, nth_expert = torch.where(selected_experts == i) 
            # batch_idx: [bs * N, 1]
            # nth_expert: [bs * N, 1]

            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                x[batch_idx]
            )
        
        results = results.contiguous().view(batch_size, N, self.hidden_size)
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
        return results

class MoELLamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        
        self.mlp = MoELLamaMLP(config)

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
    
    # def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
    #     new_state_dict = {}
    #     current_state_dict = self.state_dict()
    #     print("Begin to load state dict")
    #     # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    #     # layers.{index}.mlp.experts.{expert_index}.param if it exists
    #     # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    #     # please supplement the code below
    #     for name, param in state_dict.items():
    #         # Check if the parameter belongs to an MoE layer
    #         if name.startswith("layers.") and ".mlp." in name:
    #             layer_index = int(name.split('.')[1])
    #             param_name = name.split('.mlp.')[1]

    #             # Find the corresponding MoE parameter

    #             moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
    #             if moe_param_name in current_state_dict:
    #                 for expert_index in range(self.config.num_experts):
    #                     moe_param_name = f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"
    #                     new_state_dict[moe_param_name] = param

    #             else:
    #                 new_state_dict[name] = param
    #         else:
    #             new_state_dict[name] = param
    #     # for each in new_state_dict.keys():
    #     #     print(each)
    #     # print("=" * 100)
    #     # for each in current_state_dict.keys():
    #     #     print(each)
    #     assert new_state_dict.keys() == current_state_dict.keys()
    #     print("Finish loading state dict")
    #     # Load the state dict using the super class method
    #     return super().load_state_dict(new_state_dict, strict=strict)
    
    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    #     new_state_dict = {}

    #     print("Begin to load state dict")

    #     # print(prefix)
    #     # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    #     # layers.{index}.mlp.experts.{expert_index}.param if it exists
    #     # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    #     # please supplement the code below
    #     for name, param in state_dict.items():
    #         # Check if the parameter belongs to an MoE layer
    #         if name.startswith(prefix + "layers.") and ".mlp." in name:
    #             layer_index = int(name.split('.')[2])
    #             param_name = name.split('.mlp.')[1]

    #             # Find the corresponding MoE parameter

    #             moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
    #             if self.config.moe_layer_index == -1 or layer_index < self.config.moe_layer_index:
    #                 for expert_index in range(self.config.num_experts):
    #                     moe_param_name = prefix + f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"
    #                     new_state_dict[moe_param_name] = param

    #             else:
    #                 new_state_dict[name] = param
    #         else:
    #             new_state_dict[name] = param
    #     # if "model.layers.0.mlp.experts.1.gate_proj.weight" in new_state_dict:
    #     #     assert torch.equal(new_state_dict["model.layers.0.mlp.experts.1.gate_proj.weight"],
    #     #                     state_dict["model.layers.0.mlp.gate_proj.weight"])
    #     # print(list(state_dict.keys()))
    #     # print(list(new_state_dict.keys()))
    #     # print("missing keys", set(new_state_dict.keys()) - set(current_state_dict.keys()))
    #     # print("unexpected keys", set(current_state_dict.keys()) - set(new_state_dict.keys()))
    #     msg = super()._load_from_state_dict(new_state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
    #     print(msg)
    #     return msg 
        
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
