import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init
from mmcv.cnn import normal_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class SRFPN(nn.Module):
    """SRFPN

    Args:
        in_channels (list): number of channels for output of DLA. [64]
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5 ,
                 conv_cfg=None
                ):
        super(SRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels # 64?
        self.out_channels = out_channels # [256]? 64
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.conv_cfg = conv_cfg
      

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    sum(in_channels),
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    activation=None))

    def init_weights(self):
        print("init weights in srfpn")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
#                 caffe2_xavier_init(m)

    def forward(self, inputs):
      
        assert len(inputs) == self.num_ins
        out = inputs[0]
        outs = [out]
        
        for i in range(1, self.num_outs):
            outs.append(out)
            
        outputs = []
        for i in range(self.num_outs):
            tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
        return tuple(outputs)
