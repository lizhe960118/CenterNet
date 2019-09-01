use_coco = True
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
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[4, 8, 16, 32, 64],
        regress_ranges= ((-1, 48),(48,96),(96,192), (192,384),(384,1e8)),
        #regress_ranges=((-1, 32),(32, 64), (64, 128), (128, 256), (256, 1e8)),
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

data_root = '/data/lizhe/coco/'

img_norm_cfg = dict(
        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)

data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        use_coco=use_coco,
        ann_file=data_root + 'annotations/' +
            ('instances_trainval2014.json' if use_coco else 'pascal_train2012.json'),
        img_prefix=data_root + ('images/trainval2014/' if use_coco else 'images/'),
        img_scale=(512, 512),
#         img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
#         size_divisor=31,
#        flip_ratio=0.,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
#         img_scale=(800, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
#         img_scale=(800, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))

# optimizer
optimizer = dict(type='Adam', lr= 0.00005, betas=(0.9, 0.999), eps=1e-8)
    
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[16]
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
total_epochs = 24
#device_ids = range(2)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/center_hgfpn_2gpu'
#load_from = 'pre_train_fpn.pth'
load_from = '/home/lizhe/CenterNet/ctdet_coco_hg.pth'
#load_from = None
#resume_from = '/data/lizhe/model/centernet_hgfpn_cache/latest.pth'
resume_from = None
workflow = [('train', 1)]
