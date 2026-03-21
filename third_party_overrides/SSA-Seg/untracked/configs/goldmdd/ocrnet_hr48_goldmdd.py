_base_ = ['../ocrnet/ocrnet_hr48_512x512_160k_ade20k.py']

exec(
    compile(
        open('/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/_base_/goldmdd_dataset.py', 'rb').read(),
        '/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/_base_/goldmdd_dataset.py',
        'exec'))
exec(
    compile(
        open('/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/_base_/goldmdd_runtime.py', 'rb').read(),
        '/deac/csc/yangGrp/cuij/third_party/SSA-Seg/configs/goldmdd/_base_/goldmdd_runtime.py',
        'exec'))

model = dict(
    pretrained=None,
    decode_head=[
        dict(
            type='FCNHead',
            in_channels=[48, 96, 192, 384],
            channels=sum([48, 96, 192, 384]),
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            kernel_size=1,
            num_convs=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            concat_input=False,
            dropout_ratio=-1,
            num_classes=14,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        dict(
            type='OCRHead',
            in_channels=[48, 96, 192, 384],
            channels=512,
            ocr_channels=256,
            input_transform='resize_concat',
            in_index=(0, 1, 2, 3),
            norm_cfg=dict(type='SyncBN', requires_grad=True),
            dropout_ratio=-1,
            num_classes=14,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ])
