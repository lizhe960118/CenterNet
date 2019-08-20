import torch
import torch.nn as nn
from mmcv.cnn import normal_init
import numpy as np
import cv2
import math
#import torch.nn.functional as F

from mmdet.core import multi_apply, multiclass_nms, distance2bbox, force_fp32
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, Scale, ConvModule

INF = 1e8


@HEADS.register_module
class MatrixCenterHead(nn.Module):

    def __init__(self,
                 num_classes, # init 80
                 in_channels,
                 feat_channels=256,
                 stacked_convs=1,
                 strides=(4,   4,   4,
                          8,   8,   8,   8, 
                          16,  16,  16,  16,  16,
                               32,  32,  32,  32, 
                                    64,  64,  64),
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, INF)),
                 flags = [[0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 0, 0], [1, 0, 0, 0, 0],  [1, 1, 0, 0, 0]],
                 index_map = {0:0,  1:1,   2:2,
                              5:3,  6:4,   7:5,   8:6,
                              10:7, 11:8,  12:9,  13:10, 14:11,
                                    16:12, 17:13, 18:14, 19:15,
                                           22:16, 23:17, 24:18}, # use i * 5 +  j to get the featmap
                 loss_hm = dict(
                     type="CenterFocalLoss"
                 ), # 这里实现 CenterFocalLoss
                 loss_wh = dict(
                     type="L1Loss",
                     loss_weight=0.1
                 ),
                 loss_offset = dict(
                    type="L1Loss",
                    loss_weight=1.0
                 ),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(MatrixCenterHead, self).__init__()

        self.num_classes = num_classes
        # self.cls_out_channels = num_classes - 1
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.featmap_sizes = None
        self.loss_hm = build_loss(loss_hm)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.flags = flags
        self.index_map = index_map

        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.wh_convs = nn.ModuleList()
        self.offset_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.wh_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
            self.offset_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.center_hm = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1, bias=True)
        self.center_wh = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        self.center_offset = nn.Conv2d(self.feat_channels, 2, 3, padding=1, bias=True)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        #self.scales = nn.ModuleList([Scale(1.0) for _ in 19])

    def init_weights(self):
#         for m in self.cls_convs:
#             normal_init(m.conv, std=0.01)
#         for m in self.wh_convs:
#             normal_init(m.conv, std=0.01)
#         for m in self.offset_convs:
#             normal_init(m.conv, std=0.01)
            
        #bias_hm = bias_init_with_prob(0.01) # 这里的初始化？
        #normal_init(self.center_hm, std=0.01, bias=bias_hm)
        self.center_hm.bias.data.fill_(-2.19)
        nn.init.constant_(self.center_wh.bias, 0)
        nn.init.constant_(self.center_offset.bias, 0)
#         normal_init(self.center_hm, std=0.01)
#         normal_init(self.center_wh, std=0.01)
#         normal_init(self.center_offset, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        cls_feat = x
        wh_feat = x
        offset_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.center_hm(cls_feat)

        for wh_layer in self.wh_convs:
            wh_feat = wh_layer(wh_feat)
        wh_pred = self.center_wh(wh_feat)
        
        for offset_layer in self.offset_convs:
            offset_feat = offset_layer(offset_feat)
        offset_pred = self.center_offset(offset_feat)
        
        return cls_score, wh_pred, offset_pred

    @force_fp32(apply_to=('cls_scores', 'wh_preds', 'offset_preds'))
    def loss(self,
             cls_scores,
             wh_preds,
             offset_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):

        assert len(cls_scores) == len(wh_preds) == len(offset_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        self.featmap_sizes = featmap_sizes
        
        all_level_points = self.get_points(featmap_sizes, offset_preds[0].dtype,
                                            offset_preds[0].device)
        #print(img_metas)
        #self.c = img_metas['c']
        #self.s = img_metas['s']
        self.tensor_dtype = offset_preds[0].dtype
        self.tensor_device = offset_preds[0].device
        heatmaps, wh_targets, offset_targets = self.center_target(gt_bboxes, gt_labels, img_metas, all_level_points) # 所有层的concat的， 每张图对应一个

        num_imgs = cls_scores[0].size(0) # batch_size
        #print(num_imgs)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ] # cls_scores(num_levels, batch_size, 80, h, w)  => (num_levels, batch_size * w * h, 80)
        flatten_wh_preds = [
            wh_pred.permute(0, 2, 3, 1).reshape(-1, 2) # batchsize, h, w, 2 => batchsize, h, w, 2
            for wh_pred in wh_preds
        ]
        flatten_offset_preds = [
            offset_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            for offset_pred in offset_preds
        ]
       
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_wh_preds = torch.cat(flatten_wh_preds)
        flatten_offset_preds = torch.cat(flatten_offset_preds)
       
        # targets
        flatten_heatmaps = torch.cat(heatmaps)
        flatten_wh_targets = torch.cat(wh_targets) # torch.Size([all_level_points, 2])
        flatten_offset_targets = torch.cat(offset_targets)

        # repeat points to align with bbox_preds
        # flatten_points = torch.cat(
        #     [points.repeat(num_imgs, 1) for points in all_level_points])

        # pos_inds = flatten_labels.nonzero().reshape(-1)
        #print(flatten_wh_targets.shape)
        #print(flatten_wh_targets.nonzero())
        center_inds = flatten_wh_targets[...,0].nonzero().reshape(-1) 
        #print(center_inds)
        num_center = len(center_inds)
        #print(num_center)
        
        # what about use the centerness * labels to indict an object
        # loss_cls = self.loss_cls(
        #     flatten_cls_scores, flatten_labels, # labels gt is small area
        #     avg_factor=num_pos + num_imgs)  # avoid num_pos is 0
        flatten_cls_scores = torch.clamp(flatten_cls_scores.sigmoid_(), min=1e-4, max=1-1e-4)
        loss_hm = self.loss_hm(flatten_cls_scores, flatten_heatmaps)
        
        pos_wh_targets = flatten_wh_targets[center_inds]
        #print(pos_wh_targets.shape)
        pos_wh_preds = flatten_wh_preds[center_inds]
        
        pos_offset_preds = flatten_offset_preds[center_inds]
        pos_offset_targets = flatten_offset_targets[center_inds]
        
        if num_center > 0:
            # TODO: use the iou loss
            # center_points = flatten_points[center_inds]
            # center_decoded_bbox_preds = wh_offset2bbox(center_points, pos_wh_preds, pos_offset_preds)
            # center_decoded_bbox_targets = wh_offset2bbox(center_points, pos_wh_targets, pos_offset_targets)
            loss_wh = self.loss_wh(pos_wh_preds, pos_wh_targets, avg_factor=num_center + num_imgs)
            #loss_wh = F.l1_loss(pos_wh_preds, pos_wh_targets, reduction='sum') / (num_center + num_imgs)
            #loss_wh = 0.1 * loss_wh
            loss_offset = self.loss_offset(pos_offset_preds, pos_offset_targets, avg_factor=num_center + num_imgs)
        else:
            loss_wh = pos_wh_preds.sum()
            loss_offset = pos_offset_preds.sum()
     
        return dict(
              loss_hm = loss_hm,
              loss_wh = loss_wh,
              loss_offset = loss_offset)

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device) # 以一定间隔取x的值
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range) # 得到featmap的所有点
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points

    def center_target(self, gt_bboxes_list, gt_labels_list, img_metas, all_level_points):

        assert len(self.featmap_sizes) == len(self.regress_ranges)

        # get heatmaps and targets of each image
        # heatmaps in heatmaps_list: [num_points, 80]
        # wh_targets: [num_points, 2] => [batch_size, num_points, 2]
        heatmaps_list, wh_targets_list, offset_targets_list = multi_apply(
            self.center_target_single,
            gt_bboxes_list,
            gt_labels_list,
            img_metas
            )

        # split to per img, per level
        num_points = [center.size(0) for center in all_level_points] # 每一层多少个点 all_level_points [[12414, 2], []]
        
        heatmaps_list = [heatmaps.split(num_points, 0) for heatmaps in heatmaps_list]
        wh_targets_list = [wh_targets.split(num_points, 0) for wh_targets in wh_targets_list]
        offset_targets_list = [offset_targets.split(num_points, 0) for offset_targets in offset_targets_list]

        # concat per level image, 同一层的concat # [(batch_size，featmap_size[1]), ...)
        concat_lvl_heatmaps = []
        concat_lvl_wh_targets = []
        concat_lvl_offset_targets = []
        num_levels = len(self.featmap_sizes)
        for i in range(num_levels):
            concat_lvl_heatmaps.append(
                torch.cat([heatmaps[i] for heatmaps in heatmaps_list])) # (num_levels, batch_size * w * h, 80)
            concat_lvl_wh_targets.append(
                torch.cat(
                    [wh_targets[i] for wh_targets in wh_targets_list]))
            concat_lvl_offset_targets.append(
                torch.cat(
                    [offset_targets[i] for offset_targets in offset_targets_list]))
        return concat_lvl_heatmaps, concat_lvl_wh_targets, concat_lvl_offset_targets


 
    def center_target_single(self, gt_bboxes, gt_labels, img_meta):
        """
        single image
        gt_bboxes:torch.Size([6, 4])
        gt_labels:torch.Size([6]) tensor([34, 34, 34, 34, 34, 34], device='cuda:0')
        featmap_sizes:(list[tuple]): Multi-level feature map sizes.
        regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        """
        # transform the gt_bboxes, gt_labels to numpy
        gt_bboxes = gt_bboxes.data.cpu().numpy()
        gt_labels = gt_labels.data.cpu().numpy()
        
        #print(gt_bboxes, gt_labels)
        num_objs = gt_labels.shape[0]
        #print(num_objs)
        # heatmaps [level1, level2, level3, level4, level5]
        num_levels = len(self.featmap_sizes)

        heatmaps_targets = []
        wh_targets = []
        offset_targets = []
        # get the target shape for each image
        for i in range(num_levels): # 一共有19层
            h, w = self.featmap_sizes[i]
            hm = np.zeros((self.cls_out_channels, h, w), dtype=np.float32)
            heatmaps_targets.append(hm)
            wh = np.zeros((h, w, 2), dtype=np.float32)
            wh_targets.append(wh)
            offset = np.zeros((h, w, 2), dtype=np.float32)
            offset_targets.append(offset)

        for k in range(num_objs):
            bbox = gt_bboxes[k]
            cls_id = gt_labels[k]
            
            if img_meta['flipped']:
                bbox[[0, 2]] = img_meta['width'] - bbox[[2, 0]] - 1
                
            # condition: in the regress_ranges
            origin_h, origin_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
         
            
            # 根据h, w在哪一层将output设置为当前层的
            index_i = 0
            index_j = 0
            for i in range(5):
                min_regress_distance, max_regress_distance = self.regress_ranges[i]
                if (origin_w > min_regress_distance) and (origin_w <= max_regress_distance):
                    index_i = i
                    break
            for j in range(5):
                min_regress_distance, max_regress_distance = self.regress_ranges[j]
                if (origin_h > min_regress_distance) and (origin_h <= max_regress_distance):
                    index_j = j
                    break
             
            index_level = index_map(index_i * 5 + index_j)
            output_h, output_w = self.featmap_sizes[index_level]
            #print(output_h, output_w)
            hm = heatmaps_targets[index_level]
            wh = wh_targets[index_level]
            offset = offset_targets[index_level]
            
            # c, s is passed by meta
            trans_output = get_affine_transform(img_meta['c'], img_meta['s'], 0, [output_w, output_h])
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1) #x1, x2
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            #print(h, w)
            # 转换到当层
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                  [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                #print(ct)
                ct_int = ct.astype(np.int32)
                hm[cls_id, ct_int[1], ct_int[0]] = 1
                
                #if (ct_int[1] - 1) > 0:
                #    hm[cls_id, ct_int[1] - 1, ct_int[0]] = 0.5
                #if (ct_int[0] - 1) > 0:
                #    hm[cls_id, ct_int[1], ct_int[0] - 1] = 0.5
                #if (ct_int[1] + 1) < output_h:
                #    hm[cls_id, ct_int[1] + 1, ct_int[0]] = 0.5
                #if (ct_int[0] + 1) < output_w:
                #    hm[cls_id, ct_int[1], ct_int[0] + 1] = 0.5
                draw_umich_gaussian(hm[cls_id], ct_int, radius)

                h, w = 1. * h, 1. * w
                offset_count = ct - ct_int # h, w
                # ct_int即表明在featmap的位置 ct_int[1] * output_w + ct_int[0] 
                # TODO:如果当前位置有物体的中心，现在是直接覆盖
                # 这里设置监督信号，第1位表示w，第2位表示h
                # 这里对featmap进行缩放？
#                 wh[ct_int[1], ct_int[0], 0] = w / output_w# output_h, output_w <= y, x
#                 wh[ct_int[1], ct_int[0], 1] = h / output_h
#                 offset[ct_int[1], ct_int[0], 0] = offset_count[0] / output_w
#                 offset[ct_int[1], ct_int[0], 0] = offset_count[1] / output_h
                wh[ct_int[1], ct_int[0], 0] = w 
                wh[ct_int[1], ct_int[0], 1] = h 
                offset[ct_int[1], ct_int[0], 0] = offset_count[0]
                offset[ct_int[1], ct_int[0], 0] = offset_count[1]
            
            
            heatmaps_targets[index_level] = hm
            wh_targets[index_level] = wh
            offset_targets[index_level] = offset

        flatten_heatmaps_targets = [
            hm.transpose(1, 2, 0).reshape(-1, self.cls_out_channels)
            for hm in heatmaps_targets
        ]
        #for i in range(len(flatten_heatmaps_targets)):
        #    print(flatten_heatmaps_targets[i].shape)
            
        heatmaps_targets = np.concatenate(flatten_heatmaps_targets, axis=0) 
        #print(heatmaps_targets.shape) # (13343, 80)
        #print(heatmaps_targets)
        
        flatten_wh_targets = [
            wh.reshape(-1, 2) for wh in wh_targets
        ]
        wh_targets = np.concatenate(flatten_wh_targets)
        
        flatten_offset_targets = [
            offset.reshape(-1, 2) for offset in offset_targets
        ]
        offset_targets = np.concatenate(flatten_offset_targets)

        # transform the heatmaps_targets, wh_targets, offset_targets into tensor
        heatmaps_targets = torch.from_numpy(np.stack(heatmaps_targets))
        heatmaps_targets = torch.tensor(heatmaps_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        wh_targets = torch.from_numpy(np.stack(wh_targets))
        wh_targets = torch.tensor(wh_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        offset_targets = torch.from_numpy(np.stack(offset_targets))
        offset_targets = torch.tensor(offset_targets.detach(), dtype=self.tensor_dtype, device=self.tensor_device)
        
        return heatmaps_targets, wh_targets, offset_targets

    # test use
    @force_fp32(apply_to=('cls_scores', 'wh_preds', 'offset_preds'))
    def get_bboxes(self,
                    cls_scores,
                    wh_preds,
                    offset_preds,
                    img_metas,
                    cfg):
        assert len(cls_scores) == len(wh_preds) == len(offset_preds)
        # cls_scores => [num_levels] => [batch featmap] => [batch, 80, h, w]
        # wh_preds  => [num_levels] => [featmap] => [2, h, w]
        # offset_preds => [num_levels] => [featmap] => [2, h, w]
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        result_list = []
        #print(cls_scores[0].shape) # torch.Size([1, 80, 84, 56])
        #print(img_metas)

        for img_id in range(len(img_metas)): # 每个batch中id
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ] # =>[num_levels] => [80, h, w]
            wh_pred_list = [
                wh_preds[i][img_id].detach() for i in range(num_levels)
            ]
            offset_pred_list = [
                offset_preds[i][img_id].detach() for i in range(num_levels)
            ]
            #img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            c = img_metas[img_id]['c']
            s = img_metas[img_id]['s']
            det_bboxes = self.get_bboxes_single(cls_score_list,  wh_pred_list,
                                                offset_pred_list,
                                                featmap_sizes, c, s,
                                                scale_factor, cfg) # 对每一张图像进行解调
            result_list.append(det_bboxes)
        return result_list # [batch_size]

    def get_bboxes_single(self,
                        cls_scores,
                        wh_preds,
                        offset_preds,
                        featmap_sizes,
                        c, 
                        s,
                        scale_factor,
                        cfg):
        assert len(cls_scores) == len(wh_preds) == len(offset_preds) == len(featmap_sizes)
        
        detections = []
        for cls_score, wh_pred, offset_pred, featmap_size in zip(
                cls_scores, wh_preds, offset_preds, featmap_sizes): # 取出每一层的点
            assert cls_score.size()[-2:] == wh_pred.size()[-2:] == offset_pred.size()[-2:] == featmap_size
            
            output_h, output_w = featmap_size
            #实际上得到了每一层的hm, wh, offset
            hm = torch.clamp(cls_score.sigmoid_(), min=1e-4, max=1-1e-4).unsqueeze(0) # 增加一个纬度
            #wh_pred[0, :, :] = wh_pred[0, :, :] * output_w
            #wh_pred[1, :, :] = wh_pred[1, :, :] * output_h # 2, output_h, output_w
            wh = wh_pred.unsqueeze(0) # 这里需要乘以featuremap的尺度
            #offset_pred[0, : ,:] =  offset_pred[0, : ,:] * output_w
            #offset_pred[1, : ,:] =  offset_pred[1, : ,:] * output_h
            reg = offset_pred.unsqueeze(0)
            
            dets = ctdet_decode(hm, wh, reg=reg, K=40)
            dets = post_process(dets, c, s, output_h, output_w, scale=scale_factor, num_classes=self.num_classes)
            detections.append(dets)
        
        results = merge_outputs(detections, self.num_classes) # 单张图的结果

        return results

#num_classes = 80

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

def gaussian_small_radius(det_size, min_overlap=0.7):
    height, width = det_size
    
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)
    
    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)
    
    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
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

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=40):
    batch, cat, height, width = heat.size() # 1， 80, 128, 128
    
    #print("batch, cat, height, width\n", batch, cat, height, width)
   
    if height * width <= K:
        K = height * width 
    #print("k:", K)
    
    heat = _nms(heat)
    
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds) # inds 对应 h, w的尺度
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
        
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1) # 0, 1, 2
    
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

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

def _topk(scores, K=40):
    batch, cat, height, width = scores.size() # 1， 80，height, width
    #print("batch, cat, height, width\n", batch, cat, height, width)
    #print("k:", K)

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float() # y-> h, x-> w
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)

    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def post_process(dets, c, s, out_height, out_width, scale, num_classes):
    dets = dets.detach().cpu().numpy()
#     print("dets", dets) # (1, 100, 6)

    dets = dets.reshape(1, -1, dets.shape[2]) # （x1, y1, x2, y2)

    dets = ctdet_post_process(
        dets.copy(), [c], [s],
        out_height, out_width, num_classes)
    
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
    return dets[0]

def ctdet_post_process(dets, c, s, h, w, num_classes):
    ret = []
#     print(dets.shape) # (1, 100, 6)
#     print(c)
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
            dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
            dets[i, :, 2:4], c[i], s[i], (w, h))

        classes = dets[i, :, -1] # 类别这里是80
            
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist() # 这里将框按照类别进行分类
        ret.append(top_preds)

    return ret    
  
def merge_outputs(detections, num_classes):
#     print(detections)
    results = {}
    max_per_image = 100
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)
#         if len(self.scales) > 1 or self.opt.nms:
        results[j] = soft_nms(results[j], Nt=0.5, method=2, threshold=0.01)
#         print(results)
    scores = np.hstack([results[j][:, 4] for j in range(1, num_classes + 1)])

    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
#     print("after merge out\n", results)
    return results2coco_boxes(results, num_classes)

def results2coco_boxes(results, num_classes): 
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    bboxes = [0 for i in range(num_classes)]
    for j in range(1, num_classes + 1):
        if len(results[j]) == 0:
            bboxes[j - 1] = np.zeros((0, 5), dtype=np.float32)
            continue
        bboxes[j - 1] = results[j]
#     print(bboxes) # xyxy
    return bboxes


def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.01, method=0):
    N = boxes.shape[0]
#     cdef float iw, ih, box_area
#     cdef float ua
#     cdef int 
    pos = 0
#     cdef float 
    maxscore = 0
#     cdef int 
    maxpos = 0
#     cdef float x1,x2,y1,y2,tx1,tx2,ty1,ty2,ts,area,weight,ov

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
                                
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    boxes = boxes[keep]
    return boxes
  
def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1) 
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords
