import torch
import torch.nn as nn
import numpy as np
import cv2
# import random

num_classes = 20
    
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
    batch, cat, height, width = scores.size()
#     print("batch, cat, height, width\n", batch, cat, height, width)

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
#     print('topk_score, topk_ind\n',topk_score, topk_ind)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


    
def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size() # 1, 80, 128, 128
#     print(batch)

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    
#     print("heat\n", heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
#     print("scores, inds, clses, ys, xs\n", scores, inds, clses, ys, xs)
    
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
        
    clses  = clses.view(batch, K, 1).float()
#     print(clses)
    scores = scores.view(batch, K, 1) # 0, 1, 2
    
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
#     print("detections\n", detections)
    return detections

def post_process(dets, meta, scale=1):
    dets = dets.detach().cpu().numpy()
#     print("dets", dets) # (1, 100, 6)
    meta['c'] =  meta['c'].detach().cpu().numpy()[0]
    meta['s'] = meta['s'].detach().cpu().numpy()[0]
    meta['out_height'] = meta['out_height'].detach().cpu().numpy()[0]
    meta['out_width'] = meta['out_width'].detach().cpu().numpy()[0]
    dets = dets.reshape(1, -1, dets.shape[2]) # （x1, y1, x2, y2)
    # 根据output_size, 过滤掉一些不满足条件的框
#     output_h = meta['out_height']
#     output_w = meta['out_width']
#     keep = []
#     for index in range(dets.shape[1]):
#         x1 = dets[0, index, 0]
#         y1 = dets[0, index, 1]
#         x2 = dets[0, index, 2]
#         y2 = dets[0, index, 3]
#         if (x1 > 0 and x1 < output_w - 1) and (x2 > 0 and x2 < output_w - 1) and (y1 > 0 and y1 < output_h - 1) and (y2 > 0 and y2 < output_h - 1):
#             keep.append(index)
#     dets = dets[:, keep, :]
#     print(dets.shape) # (1, 74, 6)
    
#     bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
#     bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], num_classes)
    
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        dets[0][j][:, :4] /= scale
#     print(dets[0].shape)
    return dets[0]

def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
    ret = []
#     c, s, h, w = c[0], s[0], h[0], w[0] # to do
#     print(dets.shape) # (1, 100, 6)
#     print(c)
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, :2] = transform_preds(
              dets[i, :, 0:2], c[i], s[i], (w, h))
        dets[i, :, 2:4] = transform_preds(
              dets[i, :, 2:4], c[i], s[i], (w, h))
        
#     # 经过仿射逆变换变成x1, y1, x2, y2, 过滤掉不满足条件的框
#         input_h = h * 4
#         input_w = w * 4
#     #     print(input_h, input_w)
#         keep = []
#         for index in range(dets.shape[1]):
#             x1 = dets[i, index, 0]
#             y1 = dets[i, index, 1]
#             x2 = dets[i, index, 2]
#             y2 = dets[i, index, 3]
#             if (x1 > 0 and x1 < input_w) and (x2 > 0 and x2 < input_w) and (y1 > 0 and y1 < input_h) and (y2 > 0 and y2 < input_h) and (x1 < x2) and (y1 < y2):
#                 keep.append(index)
#         dets = dets[:, keep, :]
#     print(dets.shape) # (1, 92, 6) 

        classes = dets[i, :, -1] # 类别这里是80
#         print(classes)
            
        for j in range(num_classes):
            inds = (classes == j)
            top_preds[j + 1] = np.concatenate([
                dets[i, inds, :4].astype(np.float32),
                dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist() # 这里将框按照类别进行分类
        ret.append(top_preds)
#     print("ret", ret)
    return ret    

def merge_outputs(detections):
#     print(detections)
    results = {}
    max_per_image = 100
    for j in range(1,  num_classes + 1):
        results[j] = np.concatenate(
            [detection[j] for detection in detections], axis=0).astype(np.float32)
#         if len(self.scales) > 1 or self.opt.nms:
        results[j] = soft_nms(results[j], Nt=0.5, method=2, threshold=0.01)
#         print(results)
    scores = np.hstack(
      [results[j][:, 4] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 4] >= thresh)
            results[j] = results[j][keep_inds]
#     print("after merge out\n", results)
    return results2coco_boxes(results)

def results2coco_boxes(results): 
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """           
#  bboxes, labels, num_classes           
#     if bboxes.shape[0] == 0:
#         return [
#             np.zeros((0, 5), dtype=np.float32) for i in range(num_classes - 1)
#         ]
#     else:
#         bboxes = bboxes.cpu().numpy()
#         labels = labels.cpu().numpy()
#     return [bboxes[labels == i, :] for i in range(num_classes - 1)]
#     vis_thresh = 0.01
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
    src[1, :] = center + src_dir + scale_tmp * shift # could not broadcast input array from shape (2,2) into shape (2)
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

def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1) 
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords