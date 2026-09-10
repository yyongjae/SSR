"""Stage 1 RPED on the 25x25 BEVDepth + HDMapNet caches.

Same joint-memory readout as the BEVFusion/MapTRv2 parent.  Use this only
as a cache-pair ablation; the main RPED line is 100x100 pair B.
"""
_base_ = ['./RPED_teacher_bevfusion_maptrv2.py']

feature_root = \
    '/data2/byounggun/rideflux/pretrained_checkpoints/distill_bev_cache'

model = dict(
    feature_root=feature_root,
    teachers=dict(
        _delete_=True,
        bevdepth=dict(cache_name='bevdepth'),
        hdmapnet=dict(cache_name='hdmapnet')))
