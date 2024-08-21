import torch
from torch import nn, Tensor
from torch.nn import functional as F
from typing import Optional


from maft.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine





class ShortCut_CrossAttention(nn.Module):

    def __init__(self, d_model, nhead, panoptic_on = False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.0)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu

        self._reset_parameters()

        self.MLP = nn.Linear(d_model, d_model)
        self.panoptic_on = panoptic_on
        if panoptic_on:
            nn.init.constant(self.MLP.weight, 0.0)
            nn.init.constant(self.MLP.bias, 0.0)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        if self.panoptic_on:
            tgt = tgt + self.norm(self.MLP(tgt2))
        else:
            tgt = self.norm(tgt + self.MLP(tgt2))

        return tgt
    


class ContentDependentTransfer(nn.Module):

    def __init__(self, d_model, nhead, panoptic_on):
        super().__init__()
        self.pe_layer = PositionEmbeddingSine(d_model//2, normalize=True)
        self.cross_atten = ShortCut_CrossAttention(d_model = d_model, nhead = nhead, panoptic_on = panoptic_on)

    def visual_prediction_forward_convnext(self, x):
        batch, channel, h, w = x.shape
        x = x.reshape(batch*h*w, channel).unsqueeze(-1).unsqueeze(-1) # fake 2D input
        x = self.truck_head(x)
        x = self.head(x)
        return x.reshape(batch, h, w, x.shape[-1]).permute(0,3,1,2) # B x num_queries x 640
    

    def forward(self, img_feat, text_classifier, ):
        text_classifier = text_classifier.unsqueeze(0).repeat(img_feat.shape[0],1,1)

        pos = self.pe_layer(img_feat, None).flatten(2).permute(2, 0, 1)  # hw * b * c
        img_feat = img_feat.flatten(2).permute(2, 0, 1)  # hw * b * c

        bias = self.cross_atten(text_classifier.permute(1, 0, 2), img_feat, memory_mask=None, memory_key_padding_mask=None, pos=pos, query_pos=None)

        return bias.permute(1, 0, 2) 

