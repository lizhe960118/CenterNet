# model settings
model = dict(
    type='CenterNetFPN',
    backbone=dict(
        type='HourglassNet2'),
    neck=dict(
        type='DLAFPN',
        in_channels=[256],
        out_channels=256),
    bbox_head=dict(
        type='CenterHead',
        num_classes=20,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        regress_ranges=((-1, 32),(32, 64), (64, 128), (128, 256), (256, 1e8)),
        loss_hm=dict(
            type='CenterFocalLoss'),
        loss_wh = dict(type="L1Loss",loss_weight=0.1),
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
)
# dataset settings
dataset_type = 'CenterFPN_dataset'

data_root = '/data/lizhe/voc/'

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        use_coco = False,
        ann_file=data_root + 'annotations/pascal_trainval0712.json',
        img_prefix=data_root + 'images/',
        #img_scale=(1333, 800),
        #img_scale=(800,800),
        #img_scale=(1024, 1024),
        img_scale=(512, 512),
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
        #img_scale=(1333, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr= 0.00025, betas=(0.9, 0.999), eps=1e-8)
    
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[40])
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
total_epochs = 48
#device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/center_hgfpn_2gpu'
#load_from = 'pre_train_fpn.pth'
load_from = '/home/lizhe/CenterNet/HgNet_voc.pth'
resume_from = None
workflow = [('train', 1)]
