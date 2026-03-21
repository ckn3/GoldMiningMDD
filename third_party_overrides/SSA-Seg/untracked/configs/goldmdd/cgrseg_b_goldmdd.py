_base_ = ['../cgrseg/cgrseg-b_ade20k_160k.py']

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
    backbone=dict(init_cfg=None),
    decode_head=dict(num_classes=14))
