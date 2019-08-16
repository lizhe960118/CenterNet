# model settings
model = dict(
    type='CenterNet',
#     pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet2',
#         extra=dict(
#             stage1=dict(
#                 num_modules=1,
#                 num_branches=1,
#                 block='BOTTLENECK',
#                 num_blocks=(4, ),
#                 num_channels=(64, )),
#             stage2=dict(
#                 num_modules=1,
#                 num_branches=2,
#                 block='BASIC',
#                 num_blocks=(4, 4),
#                 num_channels=(18, 36)),
#             stage3=dict(
#                 num_modules=4,
#                 num_branches=3,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4),
#                 num_channels=(18, 36, 72)),
#             stage4=dict(
#                 num_modules=3,
#                 num_branches=4,
#                 block='BASIC',
#                 num_blocks=(4, 4, 4, 4),
#                 num_channels=(18, 36, 72, 144))),
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,)),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    
        heads=dict(
            hm = 10, wh=2, reg=2)
        )
    )

train_cfg = dict(a = 10)
test_cfg = dict(a = 5)
# dataset settings
dataset_type = 'Ctdet_txt'
data_root = '/hdd/lizhe/visdrone/'

img_norm_cfg = dict(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

data = dict(
#   batch_size
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= data_root + 'VisDrone2018-DET-train/rcnn_train.txt',
        img_prefix= data_root + 'VisDrone2018-DET-train',
        img_scale=(1024, 1024),
#         img_scale=(416, 416),
#         img_scale=(512, 512),
#         img_scale=(800, 800),
#         img_scale=(1000, 1000),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2018-DET-val/rcnn_valid.txt',
        img_prefix=data_root + 'VisDrone2018-DET-val/',
        img_scale=(1024, 1024),
#         img_scale=(1333, 800),
#         img_scale=(416, 416),
#         img_scale=(512, 512),
#         img_scale=(800, 800),
#         img_scale=(1000, 1000),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=True,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'VisDrone2018-DET-val/rcnn_valid.txt',
        img_prefix=data_root + 'VisDrone2018-DET-val',
        img_scale=(1024, 1024),
#         img_scale=(1333, 800),
#         img_scale=(416, 416),
#         img_scale=(512, 512),
#         img_scale=(800, 800),
#         img_scale=(1000, 1000),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_label=False,
        test_mode=True))
# optimizer
# optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.00025)
# optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[45, 60],
    gamma = 0.5
)
# lr_config = dict(
#     policy='poly',
#     warmup='constant',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     power=1., 
#     min_lr=1e-5)
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
total_epochs = 70
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './work_dirs/faster_rcnn_r50_fpn_1x'
work_dir = './work_dirs/centernet_hg'
load_from = None
# resume_from = None
resume_from = 'hr3_cache/epoch_4.pth'
workflow = [('train', 1)]
