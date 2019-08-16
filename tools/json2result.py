from pycocotools.coco import COCO
import json
import numpy as np
import mmcv

def json2result(result_json_file, dataset_json_file):
    results = [] # 每个位置是图像

    with open(result_json_file, "r") as f:
        result_data = json.load(f)
    

    dataset = COCO(dataset_json_file)  
    img_ids = dataset.getImgIds()
#     num_classes = 4
    num_classes = 80
#     cat_ids = {v: i for i, v in enumerate(range(1, 5))}
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
    
    '''
    # 先按照image_id分组：
    image_ids = []
    for idx in range(len(img_ids)):
        image_id = img_ids[idx]
        image_ids.append(image_id)
        
    modify_result_data = dict()
    for image_id in image_ids:
        print(image_id, len(image_ids))
        for data in result_data:
            #print(data)
            if data['image_id'] == image_id:
                this_image_id = modify_result_data.get(image_id, [])
                this_image_id.append(data)  
                modify_result_data[image_id] = this_image_id
     '''
    

    for idx in range(len(img_ids)):
        
        image_id = img_ids[idx] # 找到img_id这张图对应的所有result
        
        bboxes = [0 for i in range(num_classes)]

        cur_result = [[] for i in range(num_classes)]

        '''
        this_image = modify_result_data.get(image_id, default=None)
        
        if this_image:
             #print(this_image)
            for data in this_image:
                x1, y1, w, h = data['bbox']
                label = int(cat_ids[data['category_id']])
                cur_result[label].append(bbox)
        '''
        for data in result_data:
            #print(data)
            if data['image_id'] == image_id:
                # 表示它是属于这张图的一框
                x1, y1, w, h = data['bbox']
                score = float(data['score'])
                bbox = [x1, y1, x1 + w - 1, y1 + h - 1, score]
                label = int(cat_ids[data['category_id']])
                cur_result[label].append(bbox)

        for i in range(num_classes):
            if len(cur_result[i]) == 0:
                bboxes[i] = np.zeros((0, 5), dtype=np.float32)
            else:
                bboxes[i] = np.array(cur_result[i],dtype=np.float32)
        
        #print(bboxes)
        results.append(bboxes)
        #print(results)
        
        if (idx + 1) % 100 == 0:
            print(idx+1, len(img_ids))
            
    return results

if __name__ == '__main__':
    result_json_file = 'results.json'
    dataset_json_file = '/hdd/lizhe/coco/annotations/instances_minival2014.json'
#     dataset_json_file = "test.json"
    outputs = json2result(result_json_file, dataset_json_file)
    mmcv.dump(outputs, "result_centernet.pkl")
#     mmcv.dump(outputs, "result_renchefei.pkl")
