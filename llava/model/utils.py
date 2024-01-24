from transformers import AutoConfig
import torch.distributed as dist
import os 
import timm.models.hub as timm_hub
import re 
import logging 
import torch

def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if 'llava' in config and 'llava' not in cfg.model_type:
        assert cfg.model_type == 'llama'
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = 'LlavaLlamaForCausalLM'
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


def is_url(input_url):
    """
    Check if an input string is a url. look for http(s):// and ignoring the case
    """
    is_url = re.match(r"^(?:http)s?://", input_url, re.IGNORECASE) is not None
    return is_url

def download_cached_file(url, check_hash=True, progress=False):
    """
    Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
    If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
    """

    def get_cached_file_path():
        # a hack to sync the file path across processes
        parts = torch.hub.urlparse(url)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

        return cached_file

    if is_main_process():
        timm_hub.download_cached_file(url, check_hash, progress)

    if is_dist_avail_and_initialized():
        dist.barrier()

    return get_cached_file_path()

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def convert_state_dict(current_state_dict, state_dict, num_experts, prefix='model.'):
    new_state_dict = {}
    print("Begin to load state dict")
    

    # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    # layers.{index}.mlp.experts.{expert_index}.param if it exists
    # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    # please supplement the code below
    for name, param in state_dict.items():
        # Check if the parameter belongs to an MoE layer
        if name.startswith(prefix + "layers.") and ".mlp." in name:
            layer_index = int(name.split('.')[2])
            param_name = name.split('.mlp.')[1]

            # Find the corresponding MoE parameter

            moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
            if prefix + moe_param_name in current_state_dict:
                for expert_index in range(num_experts):
                    moe_param_name = prefix + f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"

                    new_state_dict[moe_param_name] = param

            else:
                new_state_dict[name] = param
        else:
            new_state_dict[name] = param
    return new_state_dict


def convert_eff_state_dict(current_state_dict, state_dict, num_experts, prefix='model.'):
    new_state_dict = {}
    print("Begin to load state dict")
    

    # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    # layers.{index}.mlp.experts.{expert_index}.param if it exists
    # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    # please supplement the code below
    assert num_experts == 4 or num_experts == 8
    
    for name, param in state_dict.items():
        # Check if the parameter belongs to an MoE layer
        if name.startswith(prefix + "layers.") and ".mlp." in name:
            layer_index = int(name.split('.')[2])
            param_name = name.split('.mlp.')[1]

            # Find the corresponding MoE parameter

            moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
            if prefix + moe_param_name in current_state_dict:

                for expert_index in range(num_experts):
                    moe_param_name = prefix + f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"
                    if "down_proj" in param_name:
                        each_dim = param.shape[1] // num_experts
                        new_param = param[:, each_dim * expert_index: each_dim * (expert_index + 1)]
                    else:
                        each_dim = param.shape[0] // num_experts
                        new_param = param[each_dim * expert_index: each_dim * (expert_index + 1)]
                    new_state_dict[moe_param_name] = new_param

            else:
                new_state_dict[name] = param
        else:
            new_state_dict[name] = param
    return new_state_dict


def convert_uni_state_dict(current_state_dict, state_dict, num_experts, prefix='model.'):
    new_state_dict = {}
    print("Begin to load state dict")
    

    # for any param in `state_dict` that begins with layers.{index}.mlp, initialize it to 
    # layers.{index}.mlp.experts.{expert_index}.param if it exists
    # for example, if layers.0.mlp.experts.0.{param_name} exists, then layers.0.mlp.{param_name} is initialized to layers.0.mlp.experts.0.{param_name}
    # please supplement the code below
    assert num_experts == 4 or num_experts == 8
    
    for name, param in state_dict.items():
        # Check if the parameter belongs to an MoE layer
        if name.startswith(prefix + "layers.") and ".mlp." in name:
            layer_index = int(name.split('.')[2])
            param_name = name.split('.mlp.')[1]

            # Find the corresponding MoE parameter
            shared_param_name = f"model.layers.{layer_index}.mlp.shared_expert.{param_name}"
            new_state_dict[shared_param_name] = param
            moe_param_name = f"layers.{layer_index}.mlp.experts.0.{param_name}"
            if prefix + moe_param_name in current_state_dict:

                for expert_index in range(num_experts):
                    moe_param_name = prefix + f"layers.{layer_index}.mlp.experts.{expert_index}.{param_name}"
                    if "down_proj" in param_name:
                        each_dim = param.shape[1] // num_experts
                        new_param = param[:, each_dim * expert_index: each_dim * (expert_index + 1)]
                    else:
                        each_dim = param.shape[0] // num_experts
                        new_param = param[each_dim * expert_index: each_dim * (expert_index + 1)]
                    new_state_dict[moe_param_name] = new_param

            else:
                new_state_dict[name] = param
        else:
            new_state_dict[name] = param
    return new_state_dict