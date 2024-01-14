import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.tune_vision_tower = getattr(args, "tune_vision_tower", False)
        # try:
        #     self.tune_vision_tower = args.tune_vision_tower 
        # except:
        #     self.tune_vision_tower = False
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.tuned_vision_path = getattr(args, "tuned_vision_path", None)
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        
        if self.tuned_vision_path is not None:
            vision_config = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel(vision_config)
            vision_weights = torch.load(self.tuned_vision_path, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[-1]: v for k, v in weights.items() if keyword in k}
            self.vision_tower.load_state_dict(get_w(vision_weights, 'vision_tower'), strict=True)
            print("load tuned vision tower")
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        if not self.tune_vision_tower:
            self.vision_tower.requires_grad_(False)
        else:
            print("unfreeze vision tower")
            self.vision_tower.requires_grad_(True)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
