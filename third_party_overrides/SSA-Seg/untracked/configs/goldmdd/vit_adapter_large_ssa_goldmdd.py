_base_ = ['../vit-adapter/upernet_augreg_adapter_large_512_160k_ade20k_ssa.py']

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
    backbone=dict(init_cfg=None),
    decode_head=dict(num_classes=14),
    auxiliary_head=dict(num_classes=14),
    test_cfg=dict(mode='whole'))
