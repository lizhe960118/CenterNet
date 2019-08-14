from .coco import CocoDataset
from .registry import DATASETS
import numpy as np
import cv2
import os
import math
import time
import torch
import random
from mmcv.parallel import DataContainer as DC

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

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

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

@DATASETS.register_module
class CenterFPN_dataset(CocoDataset):

    # for Voc
    CLASSES_1 = ["aeroplane", "bicycle", "bird", "boat",
     "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
     "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
     "train", "tvmonitor"]

    # for coco
    CLASSES_2 = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def __init__(self, use_coco=True, **kwargs):
        super(CenterFPN_dataset, self).__init__(**kwargs)
        self.use_coco = use_coco
        self.CLASSES = self.CLASSES_1 if use_coco else self.CLASSES_2
#         print(kwargs)

    def _get_border(self, border, size): # 128, 512
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def prepare_train_img(self, index):
        if self.use_coco:
            self.max_objs = 128
            self.num_classes = 80
            _valid_ids = [
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
              82, 84, 85, 86, 87, 88, 89, 90]
            cat_ids = {v: i for i, v in enumerate(_valid_ids)}
        else:
            self.max_objs = 50
            self.num_classes = 20
            cat_ids = {v: i for i, v in enumerate(np.arange(1, 21, dtype=np.int32))}

        # import pdb; pdb.set_trace()
        img_id = self.img_infos[index]['id']
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_prefix, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        # if self.opt.keep_res: 
        # to keep the res
        input_h = (height | self.size_divisor) + 1
        input_w = (width | self.size_divisor) + 1
        s = np.array([input_w, input_h], dtype=np.float32)
        # else:
        #s = max(img.shape[0], img.shape[1]) * 1.0
        #input_h, input_w = self.img_scales[0][1], self.img_scales[0][0]

        flipped = False
        # to random crop
        s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        w_border = self._get_border(128, img.shape[1])
        h_border = self._get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        
        # to random flip
        flip_pro = 0.3
        if np.random.random() < flip_pro:
            flipped = True
            img = img[:, ::-1, :]
            c[0] =  width - c[0] - 1

        trans_input = get_affine_transform(
          c, s, 0, [input_w, input_h])
        meta = {}
        meta['c'] = c
        meta['s'] = s
        meta['flipped'] = flipped 
        meta['height'] = height
        meta['width'] = width
     
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
        inp = inp.transpose(2, 0, 1)  # h, w, c => c, h, w
        #print(height, width, inp.shape)

        gt_bboxes = []
        gt_labels = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox']) # 输入网络 x1, y1, x2, y2
            cls_id = int(cat_ids[ann['category_id']])
            gt_bboxes.append(bbox)
            gt_labels.append(np.array(cls_id))

        # inp = inp.type(torch.cuda.FloatTensor)
        inp = torch.from_numpy(inp).type(torch.FloatTensor)
        gt_bboxes = torch.from_numpy(np.stack(gt_bboxes, axis=0))
        gt_labels = torch.from_numpy(np.stack(gt_labels, axis=0))
        #data = dict(img=DC(to_tensor(img), stack=True),img_meta=DC(img_meta, cpu_only=True),gt_bboxes=DC(to_tensor(gt_bboxes)))
        ret = {'img': DC(inp, stack=True),'img_meta':DC(meta, cpu_only=True), 'gt_bboxes':DC(gt_bboxes), 'gt_labels': DC(gt_labels)}
        return ret
    
    def prepare_test_img(self, index):
        if self.use_coco:
            self.max_objs = 128
            self.num_classes = 80
            _valid_ids = [
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
              14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
              24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
              37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
              48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
              58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
              72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
              82, 84, 85, 86, 87, 88, 89, 90]
            cat_ids = {v: i for i, v in enumerate(_valid_ids)}
        else:
            self.max_objs = 50
            self.num_classes = 20
            cat_ids = {v: i for i, v in enumerate(np.arange(1, 21, dtype=np.int32))}
        
        img_id = self.img_infos[index]['id']
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_prefix, file_name)

        img = cv2.imread(img_path)
        img_shape = img.shape
        height, width = img.shape[0], img.shape[1]
        
        #scale = random.choice([0.6, 0.8, 1.0, 1.2, 1.4])
        scale = 1
        new_height = int(height * scale)
        new_width  = int(width * scale)
        
        fix_res = False
        if fix_res:
            input_h, input_w = self.img_scales[0][1], self.img_scales[0][0]
            c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            input_h = (new_height | self.size_divisor) + 1
            input_w = (new_width | self.size_divisor) + 1
            c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
            s = np.array([input_w, input_h], dtype=np.float32)
        
        trans_input = get_affine_transform(c, s, 0, [input_w, input_h])
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # 先变换到resized_image
        inp_image = cv2.warpAffine(resized_image, trans_input, 
                                   (input_w, input_h),
                                   flags=cv2.INTER_LINEAR)
        inp_image = (inp_image.astype(np.float32) / 255.)
        inp_image = (inp_image - self.img_norm_cfg['mean']) / self.img_norm_cfg['std']
        
        #images = inp_image.transpose(2, 0, 1).reshape(1, 3, input_w, input_h)
        images = inp_image.transpose(2, 0, 1).reshape(3, input_w, input_h)
        
        flip_test = False
        if flip_test:
            images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            
        images = torch.from_numpy(images).type(torch.FloatTensor)
        
        meta = {} 
        meta['c'] = c
        meta['s'] = s
        # meta['scale_factor'] = scale_factor
        # meta['img_shape'] = img.shape
        meta['scale_factor'] = scale
        #meta['flip_test'] = flip_test
        
        ret = {'img': DC(images, stack=True), 'img_meta':DC(meta, cpu_only=True)}
        return ret
    