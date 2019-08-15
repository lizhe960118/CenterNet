from .single_stage import SingleStageDetector
from ..registry import DETECTORS


@DETECTORS.register_module
class CenterNetFPN(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CenterNetFPN, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
        
    def forward_test(self, img, img_meta,**kwargs):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        bbox_inputs = outs + (img_meta, self.test_cfg)
        result_lists = self.bbox_head.get_bboxes(*bbox_inputs)
        #print("result_lists:", result_lists) #[1]
        return result_lists[0]
