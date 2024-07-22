import torch
from torch import nn, Tensor
from torch.nn import functional as F

class MA_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, out_vocab_cls_results, mask_results, targets):
        '''
        input:  cls_score (out_vocab_cls_results)      bs * 100 * 172; 
                mask proposals (mask_results)          bs * 100 * h * w
                groundtruth (targets)                  {'labels': 1 * k; 'masks': k * h * w}
        
        output: ma_loss
        '''

        logits_per_image = F.softmax(out_vocab_cls_results[...,:-1], dim=-1)  # 16*100*171

        logits_per_instance = [] # bn * 100
        labels_per_instance = [] # bn * h*w
        masks_per_instance = []  # bn * 100 * h*w
        assert len(targets)>0, len(targets)
        mask_results = mask_results.sigmoid()
        for b in range(len(targets)):
            maski = mask_results[b].unsqueeze(0)
            for i in range(targets[b]['masks'].shape[0]):
                logiti = logits_per_image[b,:,targets[b]['labels'][i]].unsqueeze(0)
                labeli = targets[b]['masks'][i].unsqueeze(0)
                logits_per_instance.append(logiti)
                labels_per_instance.append(labeli)
                masks_per_instance.append(maski)
        
        masks_per_instance = torch.cat(masks_per_instance, dim = 0)
        labels_per_instance = torch.cat(labels_per_instance, dim = 0)
        logits_per_instance = torch.cat(logits_per_instance, dim = 0)

        ious = self.get_iou(masks_per_instance, labels_per_instance).detach()  # bs*100
        ious = self.mynorm(ious)
        
        ma_loss = self.sl1(logits_per_instance, ious)
        return ma_loss


    def get_iou(self, pred, target):
        # pred = pred.sigmoid() 
        b, c, h, w = pred.shape
        if len(target.shape)!=len(pred.shape):
            target = target.unsqueeze(1)
        # print(pred.shape, target.shape)
        # assert pred.shape == target.shape
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(
            pred,
            size=(target.shape[-2], target.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )


        pred = pred.reshape(b, c,-1)
        target = target.reshape(b, 1, -1)
        
        #compute the IoU of the foreground
        Iand1 = torch.sum(target*pred, dim = -1)
        Ior1 = torch.sum(target, dim = -1) + torch.sum(pred, dim = -1)-Iand1 + 0.0000001
        IoU1 = Iand1/Ior1

        return IoU1

    def mynorm(self, embeding):
        assert len(embeding.shape) == 2, embeding.shape
        min_em, _ = torch.min(embeding, dim = -1)
        max_em, _ = torch.max(embeding, dim = -1)
        embeding = (embeding-min_em.unsqueeze(-1))/((max_em-min_em+0.00000001).unsqueeze(-1))
        return embeding


