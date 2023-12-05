# dataset settings
dataset_type='BRATS2020Dataset'
# Change path for your purposes
data_root='path_to_project/assist-vision/henrik/swin_and_vit_/1_ScientificData/Data/Brats21/'

#img_norm_cfg = dict(
#    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (256, 256)
img_scale = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, color_type="unchanged"),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=img_scale),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    #dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    #dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    #dict(type='Resize', img_scale=img_scale),
    #dict(type='ImageToTensor', keys=['img']),
    #dict(type='Collect', keys=['img']),

    dict(type='LoadImageFromFile', to_float32=True, color_type="unchanged"),
    #dict(type='Resize', img_scale=img_scale),
    #dict(type='ImageToTensor', keys=['img']),
    #dict(type='Collect', keys=['img']),
    dict(
        type='MultiScaleFlipAug',
        img_scale = (256, 256),
    #    # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
           # dict(type='RandomFlip'),
           # dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2, # batch size of a single GPU
    workers_per_gpu=2, # Worker to pre-fetch data for each single GPU
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='masks/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='masks/validation',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='masks/test',
        pipeline=test_pipeline))
