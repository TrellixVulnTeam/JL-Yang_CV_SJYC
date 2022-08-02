dataset_type = 'Vaihingen'
data_root = '../Vaihingen512/img_rgbd'
img_norm_cfg = dict(
    mean=[120.82158307,  81.82497408,  81.23439422,  30.12282968], std=[55.08118199, 39.5522171,  38.16091649, 37.94494516],to_rgb = False)

crop_size = (512, 512) #数据集图片大小
train_pipeline = [
    dict(type='LoadImageFromFile', color_type='unchanged'), 
    dict(type='LoadAnnotations', reduce_zero_label=False),  
    # dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2)),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=1),  
    dict(type='RandomFlip', prob=0.5,direction='horizontal'),   #随机翻转
    dict(type='RandomFlip', prob=0.5,direction='vertical'),   #随机翻转
    dict(type='RandomRotate', prob=0.5,degree=90),
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
        img_scale=( 512 , 512 ),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        # img_ratios=[0.75, 1.0, 1.25],
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
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train',
        ann_dir='../an/train',
        # split="splits/train.txt",
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='../an/test',
        # split="splits/val.txt" ,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test',
        ann_dir='../an/test',
        # split="splits/test.txt",
        pipeline=test_pipeline))

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='trainOver',
#         ann_dir='../an/trainOver',
#         # split="splits/train.txt",
#         pipeline=train_pipeline
#         ),
#     val=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='test',
#         ann_dir='../an/test',
#         # split="splits/val.txt" ,
#         pipeline=test_pipeline),
#     test=dict(
#         type=dataset_type,
#         data_root=data_root,
#         img_dir='test',
#         ann_dir='../an/test',
#         # split="splits/test.txt",
#         pipeline=test_pipeline))

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