_base_ = [
    '../../_base_/models/swin/swin_base.py', '../../_base_/default_runtime.py'
]
num_cls=16
model=dict(backbone=dict(patch_size=(2,4,4), drop_path_rate=0.2), test_cfg=dict(max_testing_views=2),cls_head=dict(num_classes=num_cls,loss_cls=dict(type='CrossEntropyLoss')))

# dataset settings
dataset_type = 'RawframeDataset'
data_root = 'data/ego4d/rawframes'
data_root_val = 'data/ego4d/rawframes'
ann_file_train = 'data/ego4d/ego4d_train_rawframes.txt'
ann_file_val = 'data/ego4d/ego4d_val_rawframes.txt'
ann_file_test = 'data/ego4d/ego4d_val_rawframes.txt'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False) #K400 stats
train_pipeline = [
    dict(type='SampleFrames', clip_len=num_cls, num_clips=1,frame_uniform=True), #CE: I CHANGED THIS ON MMACTION/PIPELINES/LOADING
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label','samp_rate'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=num_cls, num_clips=1,frame_uniform=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label','samp_rate'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='SampleFrames', clip_len=num_cls, num_clips=1,frame_uniform=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 224)),
    dict(type='ThreeCrop', crop_size=224),
    dict(type='Flip', flip_ratio=0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label','samp_rate'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
data = dict(
    videos_per_gpu=4,
    workers_per_gpu=0,
    drop_last=True,
    val_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    test_dataloader=dict(
        videos_per_gpu=1,
        workers_per_gpu=1
    ),
    train=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=data_root,
        pipeline=train_pipeline,
        num_classes=num_cls,
        with_offset=True,
        filename_tmpl='{}.jpeg'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=data_root_val,
        pipeline=val_pipeline,
        num_classes=num_cls,
        with_offset=True,
        filename_tmpl='{}.jpeg'),
    test=dict(
        type=dataset_type,
        ann_file=ann_file_test,
        data_prefix=data_root_val,
        pipeline=test_pipeline,
        num_classes=num_cls,
        with_offset=True,
        filename_tmpl='{}.jpeg'))
evaluation = dict(
    interval=5, metrics=['top_k_accuracy', 'mean_class_accuracy'])

# optimizer
optimizer = dict(type='AdamW', lr=3e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(
                     custom_keys={
                         'absolute_pos_embed': dict(decay_mult=0.),
                         'relative_position_bias_table': dict(decay_mult=0.),
                         'norm': dict(decay_mult=0.),
                         'backbone': dict(lr_mult=0.1)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=2.5
)
total_epochs = 30

# runtime settings
checkpoint_config = dict(interval=1)
work_dir = './work_dirs/ego4d_swin_16cls_debug'
find_unused_parameters = False


# do not use mmdet version fp16
# fp16 = None
# optimizer_config = dict(
#     type="DistOptimizerHook",
#     update_interval=8,
#     grad_clip=None,
#     coalesce=True,
#     bucket_size_mb=-1,
#     use_fp16=True,
# )

