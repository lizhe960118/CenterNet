from .two_stage import TwoStageDetector
from ..registry import DETECTORS
from ..losses import CtdetLoss
import torch
from .ctdet_decetor import ctdet_decode, post_process, merge_outputs
# from mmdet.core import bbox2result


@DETECTORS.register_module
class CenterNet(TwoStageDetector):

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
        super(CenterNet, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.loss = CtdetLoss()

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
        output = self.backbone(img.type(torch.cuda.FloatTensor))[-1] # batch, c, h, m
        hm = torch.clamp(output['hm'].sigmoid_(), min=1e-4, max=1-1e-4)
#         hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
#         print("hm", hm)
#         print("wh", wh)
#         print("reg", reg)
        dets = ctdet_decode(hm, wh, reg=reg, K=100)
#         print("after process:\n", dets)
#         print(img_meta)
#         batch = kwargs
#         print(batch)
        dets = post_process(dets, meta = img_meta, scale=1)
#         print("after post_process:\n", dets)
        detections = [dets]
#         print(detections)
        results = merge_outputs(detections)
#         print(results)
#         det_bboxes = dets[:,:,:5].view(-1, 5)# (batch, k, 4)
#         det_labels = dets[:,:,5].view(-1) # (batch, k, 1)

#         x = self.extract_feat(img)
#         proposal_list = self.simple_test_rpn(
#             x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

# input is the output of network, return the det_bboxed, det_labels(0~L-1)
#         det_bboxes, det_labels = self.simple_test_bboxes(
#             x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)

#         bbox_results = bbox2result(det_bboxes, det_labels,
#                                self.backbone.heads['hm'] + 1)

        return results
    
    def forward_dummy(self, img):
        x = self.backbone(img)
        return x