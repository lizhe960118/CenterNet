# choose dataset
use_coco = False
# model settings
model = dict(
    type='CenterNet',
#     pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='DLA',
        base_name='dla34',
        heads=dict(hm=80 if use_coco else 20,
            wh=2,
            reg=2)
        )
    )
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         style='pytorch'),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_scales=[8],
#         anchor_ratios=[0.5, 1.0, 2.0],
#         anchor_strides=[4, 8, 16, 32, 64],
#         target_means=[.0, .0, .0, .0],
#         target_stds=[1.0, 1.0, 1.0, 1.0],
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#     bbox_roi_extractor=dict(
#         type='SingleRoIExtractor',
#         roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
#         out_channels=256,
#         featmap_strides=[4, 8, 16, 32]),
#     bbox_head=dict(
#         type='SharedFCBBoxHead',
#         num_fcs=2,
#         in_channels=256,
#         fc_out_channels=1024,
#         roi_feat_size=7,
#         num_classes=81,
#         target_means=[0., 0., 0., 0.],
#         target_stds=[0.1, 0.1, 0.2, 0.2],
#         reg_class_agnostic=False,
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))
# # model training and testing settings
train_cfg = dict(a = 10)
#     rpn=dict(
#         assigner=dict(
#             type='MaxIoUAssigner',
#             pos_iou_thr=0.7,
#             neg_iou_thr=0.3,
#             min_pos_iou=0.3,
#             ignore_iof_thr=-1),
#         sampler=dict(
#             type='RandomSampler',
#             num=256,
#             pos_fraction=0.5,
#             neg_pos_ub=-1,
#             add_gt_as_proposals=False),
#         allowed_border=0,
#         pos_weight=-1,
#         debug=False),
#     rpn_proposal=dict(
#         nms_across_levels=False,
#         nms_pre=2000,
#         nms_post=2000,
#         max_num=2000,
#         nms_thr=0.7,
#         min_bbox_size=0),
#     rcnn=dict(
#         assigner=dict(
#             type='MaxIoUAssigner',
#             pos_iou_thr=0.5,
#             neg_iou_thr=0.5,
#             min_pos_iou=0.5,
#             ignore_iof_thr=-1),
#         sampler=dict(
#             type='RandomSampler',
#             num=512,
#             pos_fraction=0.25,
#             neg_pos_ub=-1,
#             add_gt_as_proposals=True),
#         pos_weight=-1,
#         debug=False))
test_cfg = dict(a = 5)
#     rpn=dict(
#         nms_across_levels=False,
#         nms_pre=1000,
#         nms_post=1000,
#         max_num=1000,
#         nms_thr=0.7,
#         min_bbox_size=0),
#     rcnn=dict(
#         score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100)
#     # soft-nms is also supported for rcnn testing
#     # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
# )
# dataset settings
dataset_type = 'Ctdet'
if use_coco:
    data_root = '/data/lizhe/coco/'
#     'ctdet': {'default_resolution': [512, 512], 'num_classes': 80, 
#                 'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
#                 'dataset': 'coco'},
    img_norm_cfg = dict(
        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)
#     img_norm_cfg = dict(
#         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
else:
    data_root = '/data/lizhe/voc/'
    img_norm_cfg = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        use_coco=use_coco,
        ann_file=data_root + 'annotations/' +
            ('instances_trainval2014.json' if use_coco else 'pascal_trainval0712.json'),
        # ann_file=data_root + 'annotations/pascal_train2012.json' if ,
        img_prefix=data_root + ('images/trainval2014/' if use_coco else 'images/'),
        # img_scale=(1133, 800),
        img_scale=(512, 512),
#         img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
#         size_divisor=31,
        # flip_ratio=0.,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'annotations/' +
            ('instances_minival2014.json' if use_coco else 'pascal_val2012.json'),
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
        # img_scale=(1333, 800),
#         img_scale=(800, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
#        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file= data_root + 'annotations/' +
            ('instances_minival2014.json' if use_coco else 'pascal_val2012.json'),
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
#         img_scale=(1333, 800),
#         img_scale=(800, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
#       size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr=0.00025)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
#lr_config = dict(
#    policy='step',
#    warmup='linear',
#    warmup_iters=500,
#    warmup_ratio=1.0 / 3,
#    step = [80, 90],
#    gamma = 0.1
#)
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    power=1., 
    min_lr=2.5e-6)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centernet_hg'
load_from = None
# load_from = '/home/lizhe/ctdet_coco_dla_1x.pth'
resume_from = None
#resume_from = None
workflow = [('train', 1)]