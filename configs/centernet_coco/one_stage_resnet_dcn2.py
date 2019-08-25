# choose dataset
use_coco = True
# model settings
model = dict(
    type='CenterNet',
#     pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='OneStageResNetDCN2',
        depth=101,
        heads=dict(hm=80 if use_coco else 20,
            wh=2,
            reg=2)
        )
    )

train_cfg = dict(a = 10)

test_cfg = dict(a = 5)

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
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        use_coco=use_coco,
        ann_file=data_root + 'annotations/' +
            ('instances_trainval2014.json' if use_coco else 'pascal_train2012.json'),
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
        ann_file=data_root + 'annotations/instances_minival2014.json',
        img_prefix=data_root + ('images/minival2014/' if use_coco else 'images/'),
        # img_scale=(1333, 800),
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
#         img_scale=(1333, 800),
#         img_scale=(800, 800),
        img_scale=(512, 512),
        img_norm_cfg=img_norm_cfg,
        size_divisor=31,
#         flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
optimizer = dict(type='SGD', lr=0.00015, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[4],
    gamma = 0.2
)
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
total_epochs = 6
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/one_stage_resnet_dcn'
#load_from = '/data/lizhe/model/one_stage_resnetdcn_cache/latest.pth'
load_from = '/home/lizhe/CenterNet/ctdet_coco_resdcn101.pth'
# resume_from = '/home/lizhe/ctdet_coco_dla_1x.pth'
resume_from = None
workflow = [('train', 1)]

