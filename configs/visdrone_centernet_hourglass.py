model = dict(
    type='CenterNet',
#     pretrained='modelzoo://centernet_hg',
    backbone=dict(
        type='HourglassNet',
        heads=dict(hm=10,
            wh=2,
            reg=2)
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
    imgs_per_gpu=3,
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
optimizer_config = dict()
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=1.0 / 3,
#     step=[45, 50])
# learning policy
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[90, 100])
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
# work_dir = './work_dirs/centernet_hg'
load_from = None
# resume_from = "dla_cache/epoch_44.pth"
resume_from = None
workflow = [('train', 1)]
