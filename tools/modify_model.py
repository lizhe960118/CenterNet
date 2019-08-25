import torch
from collections import OrderedDict

#filename = "fcos_r50_caffe_fpn_gn_2x_4gpu_20190516_-93484354.pth"
filename = './ctdet_coco_resdcn101.pth'

checkpoint = torch.load(filename)

if isinstance(checkpoint, OrderedDict):
    state_dict = checkpoint
elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    raise RuntimeError(
        'No state_dict found in checkpoint file {}'.format(filename))

#if list(state_dict.keys())[0].startswith('module.'):
    #         state_dict = {k[7:]: v for k, v in checkpoint.items()}
# if state_dict == checkpoint:
modify_state_dict = {}
for k, v in state_dict.items():
    print(k)
#    if k.startswith("bbox_head."):
#        continue
#    modify_state_dict[k] = v
#state_dict = modify_state_dict
# else:
#     state_dict = {"backbone." + k[7:]: v for k, v in checkpoint['state_dict'].items()}
#checkpoint['state_dict'] = modify_state_dict

#save_filename = "pre_train_fpn.pth"
#torch.save(checkpoint, save_filename)