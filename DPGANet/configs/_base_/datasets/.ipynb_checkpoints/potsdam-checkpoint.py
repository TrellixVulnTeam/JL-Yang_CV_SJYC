dataset_type = 'Potsdam'
data_root = '../Potsdam512_plus/img_rgbd'
img_norm_cfg = dict(
    mean=[86.5195108 ,92.50345439, 85.86990128 ,45.67174189], std=[35.81035133 ,35.38084113 ,36.78625018, 54.96004285],to_rgb = False)

crop_size = (512, 512) #数据集图片大小
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'), 
    dict(type='LoadAnnotations', reduce_zero_label=False),  
#     dict(type='Resize', img_scale=(512, 512), ratio_range=(1, 1)), 
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0),  
    dict(type='RandomFlip', prob=0.5),   #随机翻转
#     dict(type='PhotoMetricDistortion'),  #照片失真
    dict(type='Normalize', **img_norm_cfg),   #归一化
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255), 
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile',color_type='unchanged'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True), # Ensure the long and short sides are divisible by 32
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='../an/train_doufu',
        # split="splits/train.txt",
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='../an/test_boundary',
        # split="splits/val.txt" ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='../an/test_boundary',
        # split="splits/test.txt",
        pipeline=test_pipeline))

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val',
#         ann_dir='../an/val',
#         # split="splits/train.txt",
#         pipeline=train_pipeline
#         ),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val',
#         ann_dir='../an/val',
#         # split="splits/val.txt" ,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='val',
#         ann_dir='../an/val',
#         # split="splits/test.txt",
#         pipeline=test_pipeline))