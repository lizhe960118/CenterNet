import os.path as osp
# import xml.etree.ElementTree as ET
# from PIL import Image
import cv2

# import mmcv
import numpy as np

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class txtDataset(CustomDataset):
    
    CLASSES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')
    img_ids = []
    
    def __init__(self, min_size=None, **kwargs):
        super(txtDataset, self).__init__(**kwargs)
        self.cat2label = {cat: i + 1 for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        
    def load_annotations(self, train_txt_file):
        img_infos = []
#         img_ids = []
        with open(train_txt_file, 'r') as file_to_read:
            lines = file_to_read.readlines()
            for line in lines:
                image_name = line.split("/")[-1]
                img_id = image_name.split(".")[0]
                self.img_ids.append(img_id)
                filename = 'images/{}.jpg'.format(img_id)
                abs_image_path = osp.join(self.img_prefix, filename)
                Img = cv2.imread(abs_image_path)
                image_size = Img.shape
                height = image_size[0]
                width = image_size[1] 
                img_infos.append(
                    dict(id=img_id, filename=filename, width=width, height=height))
        return img_infos
#         img_ids = mmcv.list_from_file(ann_file)
#         for img_id in img_ids:
#             filename = 'images/{}.jpg'.format(img_id)
#             txt_path = osp.join(self.img_prefix, 'annotations',
#                                 '{}.txt'.format(img_id))
#             tree = ET.parse(xml_path)
#             root = tree.getroot()
#             size = root.find('size')
#             width = int(size.find('width').text)
#             height = int(size.find('height').text)

#             img_infos.append(
#                 dict(id=img_id, filename=filename, width=width, height=height))
#         return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        txt_path = osp.join(self.img_prefix, 'rcnn_labels',
                            '{}.txt'.format(img_id))
        width = self.img_infos[idx]['width']
        height = self.img_infos[idx]['height']
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
#        with open(wd + '/annotations/' + ann, newline='') as csvfile:
#         spamreader = csv.reader(csvfile)
#         for row in spamreader:
#             if row[4] == '0':
#                 continue
#             bb = convert(img_size, tuple(map(int, row[:4])))
#             ans = ans + str(int(row[5])-1) + ' ' + ' '.join(str(a) for a in bb) + '\n'
#             with open(outpath, 'w') as outfile:
#                 outfile.write(ans)

        with open(txt_path, 'r') as file_to_read:
            lines = file_to_read.readlines()
            for line in lines:
                row = line.split(" ")
                label = int(row[0]) # 0~9
#                 if label < 0 or label > 9:
#                     continue
#                 if row[4] == '0': # 表示背景
#                     continue
                bbox = [
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4])
                       ]
                # 这里要判断一下是否标注错误
#                 if (bbox[0] < 0) or ((bbox[2] - bbox[0]) > width) or (bbox[1] < 0) or ((bbox[3] - bbox[1]) > height):
#                     continue
                ignore = False
                if self.min_size:
                    assert not self.test_mode
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w < self.min_size or h < self.min_size:
                        ignore = True
                if ignore:
                    bboxes_ignore.append(bbox)
                    labels_ignore.append(label)
                else:
                    bboxes.append(bbox)
                    labels.append(label + 1)
                
#         for obj in root.findall('object'):
#             name = obj.find('name').text
#             label = self.cat2label[name] # 从1开始标注
#             difficult = int(obj.find('difficult').text)
#             bnd_box = obj.find('bndbox')
#             bbox = [
#                 int(bnd_box.find('xmin').text),
#                 int(bnd_box.find('ymin').text),
#                 int(bnd_box.find('xmax').text),
#                 int(bnd_box.find('ymax').text)
#             ]
#             ignore = False
#             if self.min_size:
#                 assert not self.test_mode
#                 w = bbox[2] - bbox[0]
#                 h = bbox[3] - bbox[1]
#                 if w < self.min_size or h < self.min_size:
#                     ignore = True
#             if difficult or ignore:
#                 bboxes_ignore.append(bbox)
#                 labels_ignore.append(label)
#             else:
#                 bboxes.append(bbox)
#                 labels.append(label)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann
