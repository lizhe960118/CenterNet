import logging

#import torch.nn as nn
#import torch.utils.checkpoint as cp
#from torch.nn.modules.batchnorm import _BatchNorm

#from mmcv.cnn import constant_init, kaiming_init
#from mmcv.runner import load_checkpoint

#from mmdet.ops import DeformConv, ModulatedDeformConv, ContextBlock
#from mmdet.models.plugins import GeneralizedAttention

from ..registry import BACKBONES
#from ..utils import build_conv_layer, build_norm_layer

import os
import math
#import logging

import torch
import torch.nn as nn
from mmdet.ops import ModulatedDeformConvPack as DCN
import torch.utils.model_zoo as model_zoo

from torch.utils.checkpoint import checkpoint

BN_MOMENTUM = 0.1
#logger = logging.getLogger(__name__)

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        if self.efficient:
            x = checkpoint(func, x)
        else:
            x = func(x)

        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :] 

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

@BACKBONES.register_module
class TwoStageResNetDCN(nn.Module):
    
    resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}

    def __init__(self, depth, heads, heads2, deconv = True, out_indices=(0, 1, 2, 3), head_conv=64, frozen_stage_one= False):
        super(TwoStageResNetDCN, self).__init__()
        self.inplanes = 64
        self.depth = depth
        self.deconv = deconv
        self.out_indices = out_indices
        self.norm_eval = False
        self.heads = heads
        self.stage_two_heads = heads2
        self.deconv_with_bias = False
        self.frozen_stage_one = frozen_stage_one

        block, layers = self.resnet_spec[depth]
        self.stage_one_conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.stage_one_bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.stage_one_relu = nn.ReLU(inplace=True)
        self.stage_one_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stage_one_res_layers = []
        
        for i in range(len(layers)):
           
            planes = 64 * 2**i
            stride = 1 if i == 0 else 2
            res_layer = self._make_layer(block, planes, layers[i], stride)
            
            layer_name = 'stage_one_layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.stage_one_res_layers.append(layer_name)
             
        #self.layer1 = self._make_layer(block, 64, layers[0])  # 3 256?
        #self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 4
        #self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 24
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 3
        # torch.Size([4, 256, 200, 200])
        # torch.Size([4, 512, 100, 100])
        #torch.Size([4, 1024, 50, 50])
        #torch.Size([4, 2048, 25, 25])
        efficient = False
        self.stage_one_last_conv = conv_bn_relu(2048, 512, kernel_size=3, stride=1,
                padding=1, has_bn=True, has_relu=True, efficient=efficient)
        
        self.inplanes = 512

        # used for deconv layers
        self.stage_one_deconv_layers = self._make_deconv_layer(
            3,
            #[1024, 512, 256],
            [256, 128, 64],
            [4, 4, 4],
        )
        
        
        chl_num = 256
        
        #  self.u_skip : 将当前层和deconv各层连接起来
        self.stage_one_u_skip_layers = []
        
        _u_skip_in_layers = [256, 512, 1024]
        _u_skip_out_layers = [64, 128, 256]
        
        for i in range(len(layers)):
            in_planes = _u_skip_in_layers[i]
            out_planes = _u_skip_out_layers[i]
            
            u_skip_layer = conv_bn_relu(in_planes,out_planes, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
            
            layer_name = 'stage_one_layer{}_u_skip'.format(i + 1)
            self.add_module(layer_name, u_skip_layer)
            self.self.stage_one_u_skip_layers.append(layer_name)
            
        #self.skip
        
        #skip_1 : 采样原始层
        #skip_2: 采样deconv层
        self.skip_1_layers = []
        self.skip_2_layers = []
        _skip_in_layers = [256, 512, 1024, 2048]
        _skip_out_layers = [64, 128, 256, 512]
        
        for i in range(len(_skip_layers)):
            in_planes = _skip_in_layers[i]
            skip1 = conv_bn_relu(in_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
            layer_name = 'layer{}_skip1'.format(i + 1)
            self.add_module(layer_name, skip1)
            self.skip_1_layers.append(layer_name)
            
            pre_out_planes = _skip_out_layers[i]
            skip2 = conv_bn_relu(pre_out_planes, in_planes, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
            layer_name = 'layer{}_skip2'.format(i + 1)
            self.add_module(layer_name, skip2)
            self.skip_2_layers.append(layer_name)
            
        self.cross_stage_conv = conv_bn_relu(64, 64, kernel_size=1,
                    stride=1, padding=0, has_bn=True, has_relu=True,
                    efficient=efficient)
        
        #self.stage_two
        self.inplanes = 64
        stage_two_layers = [3, 4, 6, 3]
        self.stage_two_res_layers = []
        for i in range(len(layers)):
           
            planes = 64 * 2**i
            stride = 1 if i == 0 else 2
            stage_two_res_layer = self._make_layer(block, planes, stage_two_layers[i], stride)
            
            stage_two_layer_name = 'stage_two_layer{}'.format(i + 1)
            self.add_module(stage_two_layer_name, stage_two_res_layer)
            self.stage_two_res_layers.append(stage_two_layer_name)
           
        self.stage_two_last_conv = conv_bn_relu(2048, 512, kernel_size=3, stride=1,
                padding=1, has_bn=True, has_relu=True, efficient=efficient)
        
        self.inplanes = 512
        self.stage_two_deconv_layers = self._make_deconv_layer(
            3,
            [256, 128, 64],
            [4, 4, 4],
            stage=2
        )

        self.stage_two_u_skip_layers = []
        
#         _u_skip_layers = [64, 128, 256]
              
        for i in range(len(layers)):
            in_planes = _u_skip_in_layers[i]
            out_planes = _u_skip_out_layers[i]
            
            u_skip_layer = conv_bn_relu(in_planes,out_planes, kernel_size=1, stride=1,
                padding=0, has_bn=True, has_relu=False, efficient=efficient)
            
            layer_name = 'stage_one_layer{}_u_skip'.format(i + 1)
            self.add_module(layer_name, u_skip_layer)
            self.self.stage_one_u_skip_layers.append(layer_name)
        

        self.stage_one_out_conv = conv_bn_relu(64, 64, kernel_size=3, stride=1,
                padding=1, has_bn=True, has_relu=True, efficient=efficient)
    
        self.stage_two_out_conv = conv_bn_relu(64, 64, kernel_size=3, stride=1,
                padding=1, has_bn=True, has_relu=True, efficient=efficient)
        
        for head in self.heads:
            classes = self.heads[head]

            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                   kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

            self.__setattr__("stage_one_" + head, fc)
         
        # heads2 = [hm2, delta_wh, delta_reg]
        for head in self.stage_two_heads:
            classes = self.stage_two_heads[head]

            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                   kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                    kernel_size=1, stride=1,
                    padding=0, bias=True))
            if 'hm2' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)

            self.__setattr__("stage_two_" + head, fc)
            
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, stage=1):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        deconv_layers = []
            
        for i in range(num_layers):
            deconv_layer = []
            
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            fc = DCN(self.inplanes, planes, 
                    kernel_size=(3,3), stride=1,
                    padding=1, dilation=1, deformable_groups=1)
            # fc = nn.Conv2d(self.inplanes, planes,
            #         kernel_size=3, stride=1, 
            #         padding=1, dilation=1, bias=False)
            # fill_fc_weights(fc)
            up = nn.ConvTranspose2d(
                    in_channels=planes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias)
            fill_up_weights(up)

            deconv_layer.append(fc)
            deconv_layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            deconv_layer.append(nn.ReLU(inplace=True))
            deconv_layer.append(up)
            deconv_layer.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            deconv_layer.append(nn.ReLU(inplace=True))
            deconv_layer_i = nn.Sequential(*deconv_layer)
            
            if stage == 1:
                layer_name = 'stage_one_deconv_layer{}'.format(i + 1)
            else:
                layer_name = 'stage_two_deconv_layer{}'.format(i + 1)
            self.add_module(layer_name, deconv_layer_i)
            deconv_layers.append(layer_name)
            
            self.inplanes = planes
            

        return deconv_layers

    def forward(self, x):
        x = self.stage_one_conv1(x)
        x = self.stage_one_bn1(x)
        x = self.stage_one_relu(x)
        x = self.stage_one_maxpool(x)
        
        stage_one_downsample_outs = []
             
        for i, layer_name in enumerate(self.stage_one_res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            stage_one_downsample_outs.append(x)

        stage_one_upsample_outs = []
        x = self.stage_one_last_conv(x)
        
        stage_one_upsample_outs.append(x)
        
        for i in range(len(self.stage_one_deconv_layers)): #
            layer_name = self.stage_one_deconv_layers[i]
            deconv_layer = getattr(self, layer_name)
            x = deconv_layer(stage_one_upsample_outs[-1])
     #       print(x.shape)
            layer_name = self.stage_one_u_skip_layers[2 - i]
            u_skip = getattr(self, layer_name)
            u_skip_conv_layer = u_skip(stage_one_down_sample_outs[2 - i])
            
            x = x + u_skip_conv_layer
            stage_one_upsample_outs.append(x)
        stage_one_upsample_outs = stage_one_upsample_outs[::-1]
        
        stage_one_upsample_out = self.stage_one_out_conv(stage_one_upsample_outs[0])
        
        ##############################################################
        #######      stage_two   ##################################
        ##############################################################
    
        x = self.cross_stage_conv(x)
        
        stage_two_downsample_outs = []
#         stage_two_downsample_outs.append(x)
        
        for i in range(len(self.stage_two_res_layers)):
            layer_name = self.stage_two_res_layers[i]
            res_layer = getattr(self, layer_name)
            
            layer_name = self.skip_1_layers[i]
            skip1 = getattr(self, layer_name)
            skip_1_out = skip1(stage_one_downsample_outs[i])
            
            layer_name = self.skip_2_layers[i]
            skip2 = getattr(self, layer_name)
            skip_2_out = skip2(stage_one_upsample_outs[i])
            
            x = x + skip_1_out + skip_2_out
            
            x = res_layer(x)
            stage_two_downsample_outs.append(x)

        stage_two_upsample_outs = []
        x = self.stage_two_last_conv(x)
        
        stage_two_upsample_outs.append(x)
        
        for i in range(len(self.stage_two_deconv_layers)): #
            layer_name = self.stage_two_deconv_layers[i]
            deconv_layer = getattr(self, layer_name)
            x = deconv_layer(x)
     #       print(x.shape)
            layer_name = self.stage_two_u_skip_layers[2 - i]
            u_skip = getattr(self, layer_name)
            u_skip_conv_layer = u_skip(stage_two_down_sample_outs[2 - i])
            
            x = x + u_skip_conv_layer
            stage_two_upsample_outs.append(x)
        stage_two_upsample_outs = stage_two_upsample_outs[::-1]
        
        stage_two_upsample_out = self.stage_two_out_conv(stage_two_upsample_outs[0])
        
        stage_one_z = {}
        stage_two_z = {}
        for head in self.heads:
            stage_one_z[head] = self.__getattr__('stage_one_' + head)(stage_one_upsample_out)
        for head in self.stage_two_heads:
            stage_two_z[head] = self.__getattr__('stage_two_' + head)(stage_one_upsample_out)
                
        return tuple([stage_one_z, stage_two_z])

    def init_weights(self, pretrained=None):
        num_layers = self.depth
        print("init weights in resnet dcn")
       
        for layer_name in self.stage_one_deconv_layers:
            deconv_layer = getattr(self, layer_name)
            m = deconv_layer.named_modules()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
        for layer_name in self.stage_two_deconv_layers:
            deconv_layer = getattr(self, layer_name)
            m = deconv_layer.named_modules()
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
                
    def _freeze_stages(self):
        if self.frozen_stage_one:
            self.bn1.eval()
            for m in [self.conv1, self.bn1, self.relu, self.maxpool]:
                for param in m.parameters():
                    param.requires_grad = False
            
            for i in range(len(self.stage_one_res_layers)):
                layer_name = self.stage_one_res_layers[i]
                res_layer = getattr(self, layer_name)
                res_layer.eval()
                for param in res_layer.parameters():
                    param.requires_grad = False
                    
            for i in range(len(self.stage_one_u_skip_layers)):
                layer_name = self.stage_one_u_skip_layers[i]
                u_skip_layer = getattr(self, layer_name)
                u_skip_layer.eval()
                for param in u_skip_layer.parameters():
                    param.requires_grad = False     
                    
            for i in range(len(self.stage_one_deconv_layers)):
                layer_name = self.stage_one_deconv_layers[i]
                deconv_layer = getattr(self, layer_name)
                deconv_layer.eval()
                for param in deconv_layer.parameters():
                    param.requires_grad = False                       
                    
#         for i in range(1, self.frozen_stages + 1):
#             m = getattr(self, 'layer{}'.format(i))
#             m.eval()
#             for param in m.parameters():
#                 param.requires_grad = False
    def train(self, mode=True):
        super(TwoStageResNetDCN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()