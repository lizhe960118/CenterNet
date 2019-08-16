# model settings
model = dict(
    type='CenterNetFPN',
    #pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        style='caffe'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1, # 从P2开始
        #start_level=0,
        add_extra_convs=True, # 使用卷积从P5到P6
        #extra_convs_on_inputs=True, # use origin image C5
        extra_convs_on_inputs=False,  # use P5
        num_outs=5, # 输出5层
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CenterHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=1,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        #strides=[4, 8, 16, 32, 64],
        loss_hm=dict(
            type='CenterFocalLoss'),
        loss_wh = dict(type="L1Loss",loss_weight=1.0),
        loss_offset = dict(type="L1Loss",loss_weight=1.0))
)
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False
)
test_cfg = dict(
    a = 5
    #nms_pre=1000,
    #min_bbox_size=0,
    #score_thr=0.05,
    #nms=dict(type='nms', iou_thr=0.5),
    #max_per_img=100
)
# dataset settings
dataset_type = 'CenterFPN_dataset'

data_root = '/hdd/lizhe/voc/'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        use_coco = False,
        ann_file=data_root + 'annotations/pascal_trainval0712.json',
        img_prefix=data_root + 'images/',
        #img_scale=(1333, 800),
        #img_scale=(800,800),
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        use_coco = False,
        ann_file=data_root + 'annotations/pascal_val2012.json',
        img_prefix=data_root + 'images/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        use_coco = False,
        ann_file=data_root + 'annotations/pascal_val2012.json',
        img_prefix=data_root + 'images/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr= 0.001, betas=(0.9, 0.999), eps=1e-8)
    
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[35, 45])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 50
# device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/center_fpn_r50_caffe_fpn_gn_1x_4gpu'
#load_from = 'pre_train_fpn.pth'
load_from = None
resume_from = None
workflow = [('train', 1)]
