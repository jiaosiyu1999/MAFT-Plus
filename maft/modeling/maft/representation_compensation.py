import torch
from torch import nn
from torch.nn import functional as F

def _global_pod(x, spp_scales=[2, 4, 8], normalize=False):
    b = x.shape[0]
    w = x.shape[-1]
    if x.shape[-1] != x.shape[-2]:
        oh = x.shape[-1] if x.shape[-1] > x.shape[-2] else x.shape[-2]
        x = F.interpolate(x, size=(oh, oh), mode='bilinear', align_corners=False)

    emb = []
    for scale in spp_scales:
        try:
            tensor = F.avg_pool2d(x, kernel_size=w // scale)
        except:
            print('x.shape, w, scale', x.shape, w, scale)
            raise NotImplementedError(f"Unknown difference_function={x.shape, w, scale}")
        horizontal_pool = tensor.sum(dim=2).view(b, -1)
        vertical_pool = tensor.sum(dim=3).view(b, -1)
        
        if normalize:
            horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
            vertical_pool = F.normalize(vertical_pool, dim=1, p=2)
        tensor_pool = torch.cat([horizontal_pool, vertical_pool], dim=-1)
        emb.append(tensor_pool)

    return torch.cat(emb, dim=1)




class Representation_Compensation(nn.Module):

    def __init__(self, difference_function = "sl1", prepro = "relu", spp_scales = [1,2,4]):
        super().__init__()
        self.difference_function = difference_function
        self.prepro = prepro
        self.spp_scales = spp_scales
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, list_a, list_b, ):
        layer_losses = []
        if not isinstance(list_a, list):
            list_a = [list_a]
            list_b = [list_b]

        for i, (a, b) in enumerate(zip(list_a, list_b)):
            assert a.shape == b.shape, (a.shape, b.shape)

            if self.prepro == "pow":
                a = torch.pow(a, 2)
                b = torch.pow(b, 2)
            elif self.prepro == "none":
                pass
            elif self.prepro == "abs":
                a = torch.abs(a, 2)
                b = torch.abs(b, 2)
            elif self.prepro == "relu":
                a = torch.clamp(a, min=0.)
                b = torch.clamp(b, min=0.)
            else:
                raise ValueError("Unknown method to collapse: {}".format(self.prepro))
            
            a = _global_pod(a, self.spp_scales, normalize=False)
            b = _global_pod(b, self.spp_scales, normalize=False)

            if self.difference_function == "frobenius":
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
            elif self.difference_function == "l1":
                layer_loss = torch.norm(a - b, p=1, dim=-1)
            elif self.difference_function == 'sl1':
                layer_loss = self.sl1(a, b)
            elif self.difference_function == "kl":
                d1, d2, d3 = a.shape
                a = (a.view(d1 * d2, d3) + 1e-8).log()
                b = b.view(d1 * d2, d3) + 1e-8
                layer_loss = F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
            else:
                raise NotImplementedError(f"Unknown difference_function={self.difference_function}")


            assert torch.isfinite(layer_loss).all(), layer_loss
            assert (layer_loss >= 0.).all(), layer_loss

            layer_loss = torch.mean(layer_loss)

            layer_losses.append(layer_loss.unsqueeze(0))
        
        layer_losses = torch.cat(layer_losses).mean()

        return layer_losses

