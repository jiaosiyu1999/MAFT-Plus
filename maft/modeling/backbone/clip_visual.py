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

class CLIP_visual(Backbone):
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
        self.pretrained = pretrained

        clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.clip_model = clip_model.visual

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

        # self.freeze_everything()

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False


    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
        }[self.model_type](x)
    
    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.trunk.norm_pre(x)

        # x = self.clip_model.head.norm(x)
        # x = self.clip_model.head.mlp(x)
        out['clip_vis_dense'] = x.contiguous()
        return out
    
    def extract_features_resnet(self, x):
        out = {}
        x = self.clip_model.act1(self.clip_model.bn1(self.clip_model.conv1(x)))
        x = self.clip_model.act2(self.clip_model.bn2(self.clip_model.conv2(x)))
        x = self.clip_model.act3(self.clip_model.bn3(self.clip_model.conv3(x)))
        out['stem'] = x.contiguous() # os2
        x = self.clip_model.avgpool(x)
        x = self.clip_model.layer1(x)
        out['res2'] = x.contiguous() # os4
        x = self.clip_model.layer2(x)
        out['res3'] = x.contiguous() # os8
        x = self.clip_model.layer3(x)
        out['res4'] = x.contiguous() # os16
        x = self.clip_model.layer4(x)
        out['res5'] = x.contiguous() # os32
        out['clip_vis_dense'] = x
        return out

    def forward(self, x):
        return self.extract_features(x)



    def visual_prediction_forward(self, x, masks=None):
        return {
            'convnext': self.visual_prediction_forward_convnext,
            'resnet': self.visual_prediction_forward_resnet,
        }[self.model_type](x, masks)


    def visual_prediction_forward_convnext(self, x, masks):
        batch, num_query, channel = x.shape
        x = x.reshape(batch*num_query, channel, 1, 1) # fake 2D input
        x = self.clip_model.trunk.head(x)
        x = self.clip_model.head(x)
        return x.view(batch, num_query, x.shape[-1]) # B x num_queries x 640

    def visual_prediction_forward_resnet(self, x, masks):
        batch, channel, height, width = x.shape
        if masks.shape[-2] != height or masks.shape[-1] != width:
            masks = F.inteprolate(masks, size=(height, width), mode='bilinear', align_corners=False)
        num_masks = masks.shape[1]

        positional_embedding = self.clip_model.attnpool.positional_embedding.to(x.dtype)
        spatial_pos_embed = positional_embedding[1:, None, :] # HW x 1 x C
        orig_size = int(math.sqrt(spatial_pos_embed.shape[0]))
        spatial_pos_embed = spatial_pos_embed.permute(1, 2, 0).reshape(1, channel, orig_size, orig_size)
        spatial_pos_embed = F.interpolate(spatial_pos_embed, size=(height, width), mode='bilinear', align_corners=False) # 1 x C x H x W
        spatial_pos_embed = spatial_pos_embed.permute(2, 3, 0, 1).reshape(height*width, 1, channel)
        x = x.reshape(batch, channel, height * width).permute(2, 0, 1)  # BCHW -> (HW)BC
        key_value = x + spatial_pos_embed
        
        masks = masks.reshape(batch, num_masks, height * width)
        masks = (masks > 0).to(masks.dtype)
        query = x.mean(0, keepdim=True) + positional_embedding[:1, None, :]
        query = query.repeat_interleave(num_masks, dim=0)

        attn_mask = masks < 0.5
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.clip_model.attnpool.num_heads, -1, -1)
        attn_mask = attn_mask.reshape(batch * self.clip_model.attnpool.num_heads,
                                    query.shape[0], key_value.shape[0])

        x = F.multi_head_attention_forward(
            query=query, key=key_value, value=key_value,
            embed_dim_to_check=key_value.shape[-1],
            num_heads=self.clip_model.attnpool.num_heads,
            q_proj_weight=self.clip_model.attnpool.q_proj.weight,
            k_proj_weight=self.clip_model.attnpool.k_proj.weight,
            v_proj_weight=self.clip_model.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip_model.attnpool.q_proj.bias,
                                    self.clip_model.attnpool.k_proj.bias,
                                    self.clip_model.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.clip_model.attnpool.c_proj.weight,
            out_proj_bias=self.clip_model.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip_model.attnpool.training,
            need_weights=False,
            attn_mask=attn_mask
        )[0].permute(1, 0, 2) # B x N x C

        return x
