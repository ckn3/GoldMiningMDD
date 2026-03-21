_base_ = ['../afformer/afformer_base_ade20k_ssa.py']

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
    decode_head=dict(num_classes=14))
