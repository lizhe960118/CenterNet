model = dict(
    type='CenterNetFPN',
#     pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet5',
        extra=dict(
            stage2=dict(
                NUM_MODULES=1,
                NUM_BRANCHES=2,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4),
                NUM_CHANNELS=(32, 64),
                FUSE_METHOD = 'SUM'
            ),
            stage3=dict(
                NUM_MODULES=4,
                NUM_BRANCHES=3,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4),
                NUM_CHANNELS=(32, 64, 128),
                FUSE_METHOD = 'SUM'
            ),
            stage4=dict(
                NUM_MODULES=3,
                NUM_BRANCHES=4,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4, 4),
                NUM_CHANNELS=(32, 64, 128, 256),
                FUSE_METHOD = 'SUM'))),
    neck=dict(
        type='HRFPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256),
    bbox_head=dict(
        type='CenterHead',
        num_classes=4,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        #strides=[8, 16, 32, 64, 128],
        strides=[4, 8, 16, 32, 64],
        regress_ranges=((-1, 64),(64, 192),(192, 512), (512, 768), (768, 1e8)),
        loss_hm=dict(
            type='CenterFocalLoss'),
        loss_wh = dict(type="L1Loss",loss_weight=0.1),
        loss_offset = dict(type="L1Loss",loss_weight=1.0))
)

train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(a = 5)

dataset_type = 'RenCheFPNDataset'

data_root = '/root/train/trainset/1/'

img_norm_cfg = dict(
    mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)

data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "deepv_resize/" + 'train.json',
        img_prefix=data_root + "deepv_resize/",
        # img_scale=(1133, 800),
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
#         size_divisor=31,
        # flip_ratio=0.,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file= data_root + 'deepv_49w_src/' + 'test.json',
        img_prefix=data_root + 'deepv_49w_src/',
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
        ann_file= data_root + 'deepv_49w_src/' + 'test.json',
        img_prefix=data_root + 'deepv_49w_src/',
#         img_scale=(1333, 800),
#        img_scale = (1024, 1024),
         img_scale=(800, 800),
#        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='Adam', lr=0.00015)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step = [4],
    gamma = 0.1
)
#lr_config = dict(
#    policy='poly',
#    warmup='constant',
#    warmup_iters=500,
#    warmup_ratio=1.0 / 3,
#    power=1., 
#    min_lr=1e-10)
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
total_epochs = 5
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
work_dir = './work_dirs/centernet_hg'
load_from = '/root/train/trainset/1/renche_cache/latest.pth'
#resume_from = None
resume_from = None
workflow = [('train', 1)]
