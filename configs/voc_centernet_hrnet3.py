use_coco = False
# model settings
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
                NUM_CHANNELS=(18, 36),
                FUSE_METHOD = 'SUM'
            ),
            stage3=dict(
                NUM_MODULES=4,
                NUM_BRANCHES=3,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4),
                NUM_CHANNELS=(18, 36, 72),
                FUSE_METHOD = 'SUM'
            ),
            stage4=dict(
                NUM_MODULES=3,
                NUM_BRANCHES=4,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4, 4, 4),
                NUM_CHANNELS=(18, 36, 72, 144),
                FUSE_METHOD = 'SUM'
            )
         ),
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4,),
#                 num_channels=(64,)),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(32, 64)),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(32, 64, 128)),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(32, 64, 128, 256))),
    
        heads=dict(
            hm=80 if use_coco else 20, wh=2, reg=2)
        )
    )

train_cfg = dict(a = 10)
test_cfg = dict(a = 5)

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
    imgs_per_gpu=12,
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
        use_coco=use_coco,
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
        use_coco=use_coco,
        ann_file= data_root + 'annotations/' +
            ('instances_minival2014.json' if use_coco else 'pascal_test2007.json'),
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
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step = [80, 90],
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
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
work_dir = './work_dirs/centernet_hg'
load_from = None
#resume_from = None
resume_from = '/data/lizhe/model/hr3_cache_1/epoch_44.pth'
workflow = [('train', 1)]