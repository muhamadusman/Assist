_base_ = [
    '/proj/assist/users/x_muhak/assist-vision/henrik/swin_and_vit_/configs/_base_/models/upernet_swin_brats2020.py', '/proj/assist/users/x_muhak/assist-vision/henrik/swin_and_vit_/configs/_base_/datasets/mydatasets/noaug/assist_1/brats2021_2dslices_001.py', 
    '/proj/assist/users/x_muhak/assist-vision/henrik/swin_and_vit_/configs/_base_/default_runtime.py', '/proj/assist/users/x_muhak/assist-vision/henrik/swin_and_vit_/configs/_base_/schedules/schedule_20k.py'
] 

model = dict(
    backbone=dict(
        in_channels=4,
        embed_dims=128,
        depths=[2,2,18,2],
        num_heads=[4,8,16,32],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True
    ),
    decode_head=dict(
        in_channels=[128,256,512,1024],
        num_classes=4,
        loss_decode=dict(
            type='DiceCrossEntropyLoss',
            include_background=False,
            to_onehot_y=False,
            sigmoid=False,
            softmax=True,
            other_act=None,
            squared_pred=True,
            jaccard=False,
            reduction='mean',
            smooth_nr=1e-05,
            smooth_dr=1e-05,
            ce_weight=None,
            lambda_dice=1.0,
            lambda_ce=1.0,
            batch=False,
            loss_weight=1.0)
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=4,
        loss_decode=dict(
            type='DiceCrossEntropyLoss',
            include_background=False,
            to_onehot_y=False,
            sigmoid=False,
            softmax=True,
            other_act=None,
            squared_pred=True,
            jaccard=False,
            reduction='mean',
            smooth_nr=1e-05,
            smooth_dr=1e-05,
            ce_weight=None,
            lambda_dice=1.0,
            lambda_ce=1.0,
            batch=False,
            loss_weight=0.4)
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='epochpoly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, 
                 min_lr=0.0, 
                 by_epoch=False)

workflow = [('train', 1), ('val', 1)] # Remove ('val', 1) if not running validation

# By default, models are trained on 8 GPUs with 2 images per GPU

data=dict(samples_per_gpu=8,workers_per_gpu=1) # Batch_size = 8, workers = 1

runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_iters=None,
    max_epochs=25)

checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=1)  # The save interval.

evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaluation/eval_hook.py for details.
    interval=20,  # SHould be bigger number than max_epochs if no evaluation is to be run.
    metric= ['mIoU', 'mDice', 'mFscore'],
    by_epoch=True)  # The evaluation metric.

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])