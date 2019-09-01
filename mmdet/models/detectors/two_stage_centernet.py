from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from ..losses import TwoStageCtdetLoss
import torch
from .ctdet_decetor import ctdet_decode, post_process, merge_outputs
# from mmdet.core import bbox2result


@DETECTORS.register_module
class TwoStageCenterNet(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head=None,
                 bbox_roi_extractor=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(TwoStageCenterNet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.loss = TwoStageCtdetLoss()

    def forward_train(self, img, img_meta, **kwargs):
        # print('in forward train')
        output = self.backbone(img.type(torch.cuda.FloatTensor))
        # print(kwargs)
        # loss, loss_stats = self.loss(output, **kwargs)
        losses = self.loss(output, **kwargs)

        # import pdb; pdb.set_trace()
        return losses#, loss_stats
    
    def forward_test(self, img, img_meta, **kwargs):
        """Test without augmentation."""
#         assert self.with_bbox, "Bbox head must be implemented."
#         print("after preprocess\n:", img, img_meta)
#        detections = []

        output = self.backbone(img.type(torch.cuda.FloatTensor)) # batch, c, h, m
        #print(output)
        
        output_stage_one = output[0]
        
        hm_stage_one = torch.clamp(output_stage_one['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
        wh_stage_one = output_stage_one['wh']
        reg_stage_one = output_stage_one['reg']
        
        
        output_stage_two = output[1]
        hm_stage_two = torch.clamp(output_stage_two['hm2'].sigmoid_(), min=1e-4, max=1-1e-4)
 
        delta_wh = output_stage_two['delta_wh']
        delta_reg = output_stage_two['delta_reg']
        
        #hm = hm_stage_two
        #hm = hm_stage_one
        #wh = wh_stage_one
        #reg = reg_stage_one
        hm = (hm_stage_one + hm_stage_two) / 2 
        #wh = wh_stage_one + 0.5 * delta_wh
        wh = wh_stage_one + delta_wh
        reg = reg_stage_one + delta_reg
        
        dets = ctdet_decode(hm, wh, reg=reg, K=100)

        dets = post_process(dets, meta = img_meta, scale=1)

        detections = [dets]

        results = merge_outputs(detections)


        return results
    
    def forward_dummy(self, img):
        x = self.backbone(img)
        return x
