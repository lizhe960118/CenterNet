import sys
sys.path.insert(0, "/hdd/lizhe/coco/PythonAPI/")

import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
#import skimage.io as io
import pylab,json
import os
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_img_id(file_name):
    ls = []
    myset = []
    annos = json.load(open(file_name, 'r'))
    for anno in annos:
        ls.append(anno['image_id'])
    myset = {}.fromkeys(ls).keys()
    return myset

if __name__ == '__main__':

    annType = ['segm', 'bbox', 'keypoints']#set iouType to 'segm', 'bbox' or 'keypoints'
    annType = annType[1]  # specify type here
#     cocoGt_file = "../../trainset/1/coco/annotations/instances_testdev2017.json"#标注集json文件
    cocoGt_file = "/hdd/lizhe/coco//annotations/instances_minival2014.json"
    cocoGt = COCO(cocoGt_file)#取得标注集中coco json对象

#     cocoDt_file = "./results/CenterNet-104/250000/testing/results.json"#结果json文件
    cocoDt_file = "results.json"
    cocoDt = cocoGt.loadRes(cocoDt_file)#取得结果集中image json对象

    imgIds = get_img_id(cocoDt_file)
    print(len(imgIds))
    imgIds = sorted(imgIds)#按顺序排列coco标注集image_id

    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds#参数设置
    cocoEval.params.catIds = sorted(cocoGt.getCatIds())
    cocoEval.evaluate()#评价
    cocoEval.accumulate()#积累
    cocoEval.summarize()#总结