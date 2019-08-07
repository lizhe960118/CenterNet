# choose dataset
use_coco = True
# model settings
model = dict(
    type='CenterNet',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet2',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(18, 36, 72)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(18, 36, 72, 144))),
        heads=dict(
            hm=80 if use_coco else 21, wh=2, reg=2)
        )
    )

train_cfg = dict(a = 10)
test_cfg = dict(a = 5)

dataset_type = 'Ctdet'
if use_coco:
    data_root = '/hdd/lizhe/coco/'
    img_norm_cfg = dict(
        mean=[0.408, 0.447, 0.470], std=[0.289, 0.274, 0.278], to_rgb=True)
else:
    data_root = '/hdd/lizhe/voc/'
    img_norm_cfg = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        use_coco=use_coco,
        ann_file=data_root + 'annotations/' +
            ('instances_trainval2014.json' if use_coco else 'pascal_train2012.json'),
        # ann_file=data_root + 'annotations/pascal_train2012.json' if ,
        img_prefix=data_root + ('images/trainval2014/' if use_coco else 'images/'),
        # img_scale=(1133, 800),
#         img_scale=(512, 512),
        img_scale=(800, 800),
        img_norm_cfg=img_norm_cfg,
#         size_divisor=31,
        # flip_ratio=0.,
        with_mask=False,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
        # img_scale=(1333, 800),
        img_scale=(800, 800),
#         img_scale=(512, 512),
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
#         img_scale=(1333, 800),
        img_scale=(800, 800),
#         img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
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
total_epochs = 12
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/centernet_hg'
load_from = None
resume_from = None
workflow = [('train', 1)]
