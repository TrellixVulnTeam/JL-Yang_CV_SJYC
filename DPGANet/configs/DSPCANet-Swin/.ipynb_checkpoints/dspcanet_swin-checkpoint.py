_base_ = [
    '../_base_/models/dspcanet_swin_base.py', '../_base_/datasets/vaihigen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_120e.py'
]

# _base_ = [
#     '../_base_/models/upernet_swin.py', '../_base_/datasets/potsdam.py',
#     '../_base_/default_runtime.py', '../_base_/schedules/schedule_120e.py'
# ]

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00007, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=800,
                 warmup_ratio=1e-6,
                 power=2.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=8)