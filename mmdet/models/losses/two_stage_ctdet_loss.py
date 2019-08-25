import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from ..registry import LOSSES

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
#     print(pred) # 几乎全部是0
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

class RegMSELoss(nn.Module):
    def __init__(self):
        super(RegMSELoss, self).__init__()

    def forward(self, output_stage_one, output_stage_two, mask, ind, target):
        pred_1 = _tranpose_and_gather_feat(output_stage_one, ind)
        pred_2 = _tranpose_and_gather_feat(output_stage_two, ind)
        mask = mask.unsqueeze(2).expand_as(pred_1).float()
        # updata_target
        target = target - pred_1
        # print(target)
        # import pdb; pdb.set_trace()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        #loss = F.l1_loss(pred2 * mask, target * mask, reduction='sum')
        #loss = loss / (mask.sum() + 1e-4)
        loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
        loss = loss_fn(pred_2 * mask, target * mask) 
        loss = loss / (mask.sum() + 1e-4)
        return loss

@LOSSES.register_module
class TwoStageCtdetLoss(torch.nn.Module):
    def __init__(self):
        super(TwoStageCtdetLoss, self).__init__()
        # self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        # self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
        #           RegLoss() if opt.reg_loss == 'sl1' else None
        # self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
        #           NormRegL1Loss() if opt.norm_wh else \
        #           RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit = FocalLoss()
        self.crit_reg = RegL1Loss()
        self.stage_two_crit_reg = RegMSELoss()
        # self.crit_wh = self.crit_reg
        # self.opt = opt
        # opts
#         self.num_stacks = 1
#         self.wh_weight = 0.1
        self.wh_weight = 0.1
        self.off_weight = 1
        self.hm_weight = 1

    def forward(self, outputs, **kwargs):
        batch = kwargs
        #hm_loss, wh_loss, off_loss = 0, 0, 0
        stage_two_hm_loss, stage_two_wh_loss, stage_two_off_loss = 0, 0, 0
        ####  stage one ########

        stage_one_output = outputs[0]

#        stage_one_output['hm'] = torch.clamp(stage_one_output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)

#         stage_one_hm_loss += self.crit(stage_one_output['hm'], batch['hm'])
#         if self.wh_weight > 0:
#             stage_one_wh_loss += self.crit_reg(
#                 stage_one_output['wh'], batch['reg_mask'],
#                 batch['ind'], batch['wh'])

#         if self.off_weight > 0:
#             stage_one_off_loss += self.crit_reg(stage_one_output['reg'], batch['reg_mask'],
#                                 batch['ind'], batch['reg'])
        #### stage two ####
        stage_two_output = outputs[1]

        stage_two_output['hm2'] = torch.clamp(stage_two_output['hm2'].sigmoid_(), min=1e-4, max=1-1e-4)

        stage_two_hm_loss += self.crit(stage_two_output['hm2'], batch['hm'])
        
        if self.wh_weight > 0:
            stage_two_wh_loss += self.stage_two_crit_reg(
                stage_one_output['wh'], stage_two_output['delta_wh'], 
                batch['reg_mask'], batch['ind'], batch['wh'])

        if self.off_weight > 0:
            stage_two_off_loss += self.stage_two_crit_reg(
                stage_one_output['reg'], stage_two_output['delta_reg'],
                batch['reg_mask'], batch['ind'], batch['reg'])
            
        losses = {  
#                  'stage_one_hm_loss': self.hm_weight * stage_one_hm_loss,
#                  'stage_one_wh_loss': self.wh_weight * stage_one_wh_loss, 
#                  'stage_one_off_loss': self.off_weight * stage_one_off_loss,
                 'stage_two_hm_loss': self.hm_weight * stage_two_hm_loss,
                 'stage_two_wh_loss': self.wh_weight * stage_two_wh_loss, 
                 'stage_two_off_loss': self.off_weight * stage_two_off_loss
               }
        
        return losses