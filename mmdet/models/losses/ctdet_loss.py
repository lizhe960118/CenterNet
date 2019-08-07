import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..registry import LOSSES

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
  diameter = 2 * radius + 1
  gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

  x, y = int(center[0]), int(center[1])

  height, width = heatmap.shape[0:2]

  left, right = min(x, radius), min(width - x, radius + 1)
  top, bottom = min(y, radius), min(height - y, radius + 1)

  masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
  masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
  if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
  return heatmap

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # print(target)
        # import pdb; pdb.set_trace()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

@LOSSES.register_module
class CtdetLoss(torch.nn.Module):
    def __init__(self):
        super(CtdetLoss, self).__init__()
        # self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        # self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
        #           RegLoss() if opt.reg_loss == 'sl1' else None
        # self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
        #           NormRegL1Loss() if opt.norm_wh else \
        #           RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        # self.crit_wh = self.crit_reg
        # self.opt = opt
        # opts
        self.num_stacks = 1
        self.wh_weight = 0.1
        self.off_weight = 1
        self.hm_weight = 1

    def forward(self, outputs, **kwargs):
        batch = kwargs
        hm_loss, wh_loss, off_loss = 0, 0, 0
        for s in range(self.num_stacks):
            output = outputs[s]
            # for key, value in output.items():
            #     print(key, value.shape)
            # if not opt.mse_loss:
            output['hm'] = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)

            # if opt.eval_oracle_hm:
            #   output['hm'] = batch['hm']
            # if opt.eval_oracle_wh:
            #   output['wh'] = torch.from_numpy(gen_oracle_map(
            #     batch['wh'].detach().cpu().numpy(),
            #     batch['ind'].detach().cpu().numpy(),
            #     output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            # if opt.eval_oracle_offset:
            #   output['reg'] = torch.from_numpy(gen_oracle_map(
            #     batch['reg'].detach().cpu().numpy(),
            #     batch['ind'].detach().cpu().numpy(),
            #     output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / self.num_stacks
            if self.wh_weight > 0:
                wh_loss += self.crit_reg(
                    output['wh'], batch['reg_mask'],
                    batch['ind'], batch['wh']) / self.num_stacks

            if self.off_weight > 0:
              off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                    batch['ind'], batch['reg']) / self.num_stacks

        # loss = self.hm_weight * hm_loss + self.wh_weight * wh_loss + \
        #        self.off_weight * off_loss
        losses = {'hm_loss': self.hm_weight * hm_loss,
                    'wh_loss': self.wh_weight * wh_loss, 'off_loss': self.off_weight * off_loss}
        # loss_stats = {'loss': loss, 'hm_loss': hm_loss,
        #               'wh_loss': wh_loss, 'off_loss': off_loss}
        # return loss, loss_stats
        return losses