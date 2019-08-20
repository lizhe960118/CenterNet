model = dict(
    type='CenterNet',
#     pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet3',
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
                FUSE_METHOD = 'SUM'
            )
         ),
    
        heads=dict(
            hm=4, wh=2, reg=2)
        )
    )

train_cfg = dict(a = 10)
test_cfg = dict(a = 5)

dataset_type = 'RenCheDataset'

data_root = '/root/train/trainset/1/'

img_norm_cfg = dict(
    mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)

data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "deepv_resize/" + 'train.json',
        img_prefix=data_root + "deepv_resize/",
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
	img_scale = (2048, 2048),
#         img_scale=(800, 800),
#        img_scale=(512, 512),
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
    interval=50,
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
load_from = None
#resume_from = None
resume_from = '/data/lizhe/model/hr32_cache/epoch_44.pth'
workflow = [('train', 1)]
