import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmcv.cnn.weight_init import caffe2_xavier_init

from ..utils import ConvModule
from ..registry import NECKS


@NECKS.register_module
class MatrixFPN(nn.Module):
    """Matrix Nets 

    https://arxiv.org/pdf/1908.04646.pdf

    Args:
        in_channels (list): number of channels for each branch. [the output of backbone]
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
                 out_channels=256,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False):
        super(MatrixFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.layer_i = num_outs
        self.layer_j = num_outs

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            activation=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride = 2,
                    conv_cfg=self.conv_cfg,
                    activation=None))
            
        self.width_half_convs = nn.ModuleList()
        for i in range(7):
             self.width_half_convs.append(
                 ConvModule(
                     out_channels,
                     out_channels, 
                     kernel_size=3, 
                     padding=1,
                     stride = [1, 2],
                     conv_cfg=self.conv_cfg,
                     activation=None)
        self.width_half_dict = {1:0, 2:1, 7:2, 8:3, 13:4, 14:5, 19:6} # use i, j index to get convs
                 
        self.height_half_convs = nn.ModuleList()
        for i in range(7):
             self.height_half_convs.append(
                 ConvModule(
                     out_channels,
                     out_channels, 
                     kernel_size=3, 
                     padding=1,
                     stride = [2, 1],
                     conv_cfg=self.conv_cfg,
                     activation=None)
         self.height_half_dict = {5:0, 10:1, 11:2, 16:3, 17:4, 22:5, 23:6} 
        
    def init_weights(self):
        print("init weights in matrix fpn")
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 caffe2_xavier_init(m)

    def forward(self, inputs):
        assert len(inputs) == self.num_ins
        out = [inputs[0]]
        
        out = self.reduction_conv(out) # backbone output
        
        diagonal_layer_outputs = []

        for i in range(self.num_outs):
            out = self.fpn_convs[i](out)
            diagonal_outputs.append(out)
        
        #outputs = [0 for k in range(self.layer_i * self.layer_j)]
        outputs = [[0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],  [1, 1, 0, 0, 0]]
        for i in range(self.layer_i):
            for j in range(self.layer_j):
                if outputs[i][j] == 1:
                    continue
                if i == j:
                    outputs[i][j] = diagonal_outputs[i]
                elif i < j:
                    outputs[i][j] = self.width_half_convs[self.width_half_dict[i * self.layer_j + j]](outputs[i][j-1])
                else: # i > j
                    outputs[i][j] = self.height_half_convs[self.height_half_dict[i * self.layer_j + j]](outputs[i-1][j])
         
        final_outputs = []
         for i in range(self.layer_i):
            for j in range(self.layer_j):
                 if outputs[i][j] == 1:
                    continue
                 final_outputs.append(outputs[i][j])
            
        return tuple(final_outputs)
