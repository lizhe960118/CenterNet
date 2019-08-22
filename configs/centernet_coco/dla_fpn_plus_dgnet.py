 # fp16 setti`ngs
# fp16 = dict(loss_scale=4.)

# model settings
model = dict(
    type='CenterNetFPN',
    #pretrained='open-mmlab://resnet50_caffe',
    backbone=dict(
        type='DLA2',
        base_name='dla34'),
    neck=dict(
        type='DLAFPN',
        #in_channels=[64, 128, 256, 512],
        in_channels=[64],
        out_channels=256,
        num_outs=4
    ),
    bbox_head=dict(
        type='CenterHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=2,
        feat_channels=256,
        strides=(4, 8, 16, 32, 64), # 512 => 128, 64, 32, 16
        #regress_ranges=((-1, 32), (32, 128), (128, 512), (512, 1e8)),
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384), (384, 1e8)),
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
data_root = '/root/train/trainset/1/coco/'
img_norm_cfg = dict(
        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_trainval2014.json',
        img_prefix=data_root + 'images/trainval2014/',
        #img_scale=(1333, 800),
        #img_scale=(800,800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/minival2014/',
        img_scale=(1333, 800),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + 'images/minival2014/',
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
    #type='SGD',
    #lr=0.01,
    #momentum=0.9,
    #weight_decay=0.0001,
    #paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[5, 8, 11],
    gamma=0.2
)
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
total_epochs = 12
#total_epochs = 5
#device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/root/train/trainset/1/centernet_dla_fpn_plus_cache'
#load_from = 'pre_train_fpn.pth'
load_from = '/root/train/trainset/1/CenterNet/ctdet_coco_dla_2x.pth'
#load_from = None
#resume_from = '/data/lizhe/model/centernet_dla_fpn_cache/latest.pth'
resume_from = None
workflow = [('train', 1)]
