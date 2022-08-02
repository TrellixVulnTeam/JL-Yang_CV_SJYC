# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    # pretrained='swin_tiny_patch4_window7_224.pth',
    # pretrained='swin_base_patch4_window12_384_22k.pth',
    # pretrained='swin_base_patch4_window7_224_22k.pth',
    pretrained='swin_large_patch4_window12_384_22k.pth',
    # pretrained='swin_small_patch4_window7_224.pth',
    backbone=dict(
        type='RFNet',
        ####################################
        embed_dim=96,  #small tiny
        # embed_dim=128, # base
        # embed_dim = 192, #large
        #########################
#         depths=[2, 2, 6, 2], #tiny 
        depths=[2, 2, 18, 2], # small base large
        ###########################
        num_heads=[3, 6, 12, 24], # small tiny
        # num_heads=[4, 8, 16, 32],  # base 
        # num_heads = [ 6 , 12 , 24 , 48 ], #large
        #############################
        window_size=7,  # small tiny
        # window_size=12, # base large
        #########################
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 96*2, 96*4, 96*8], # small tiny
        # in_channels=[128, 256, 512, 1024],  # base
        # in_channels = [ 192 , 192*2 , 192*4 , 192*8 ],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=96*4,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
#     model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
