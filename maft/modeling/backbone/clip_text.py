"""
Copyright (2023) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License. 
"""

import torch
import torch.nn.functional as F
import math
from detectron2.utils import comm

import open_clip

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

class CLIP_text(Backbone):
    def __init__(self, cfg):
        super().__init__()

        if cfg == 'vith':
            model_name, pretrained = ('ViT-H-14', 'laion2b_s32b_b79k') 
        elif cfg == 'vitg':
            model_name, pretrained = ('ViT-bigG-14', 'laion2b_s39b_b160k')
        elif cfg == 'convl':
            model_name, pretrained = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
        else:
            raise NotImplementedError(
                "Prompt learner {} is not supported".format(model_name)
            )
    
        # download on local rank 0 first
        if comm.get_local_rank() == 0:
            open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        comm.synchronize()

        self.model_name = model_name

        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.attn_mask = clip_model.attn_mask


        model_name = model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]
        
        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]


    def get_text_classifier_unfix(self, text_list, device):
        text_tokens = self.text_tokenizer(text_list)
        text_tokens = text_tokens.to(device)
        text_features = self.encode_text(text_tokens, normalize=False)
        return text_features

    def encode_text(self, text, normalize: bool = False):
        with torch.no_grad():
            cast_dtype = self.transformer.get_cast_dtype()

            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

            x = x + self.positional_embedding.to(cast_dtype)
            x = x.permute(1, 0, 2)  # NLD -> LND
            self.attn_mask = self.attn_mask.to(x.device)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return F.normalize(x, dim=-1) if normalize else x

    def forward(self, x):
        return self.get_text_classifier_unfix(x, 'cuda')