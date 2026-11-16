"""modules/dinov2_lora.py, the image_size path of the feature builder, and the model built on them.  CPU, no download."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import timm
import torch
import torch.nn as nn

from navsim.agents.para_ssr.configs.default import ParaSSRConfig
from navsim.agents.para_ssr.modules.dinov2_lora import Dinov2LoRAEncoder, LoRAQKV, is_dinov2
from navsim.agents.para_ssr.para_ssr_features import ParaSSRFeatureBuilder, build_lidar2img
from navsim.agents.para_ssr.para_ssr_model import ParaSSRModel

ARCH = "vit_small_patch14_reg4_dinov2.lvd142m"
HW = (56, 98)   # 4 x 7 tokens, cheap; the code path is the one of 672 x 1148


def _encoder(rank=4, seed=0):
    torch.manual_seed(seed)
    return Dinov2LoRAEncoder(ARCH, HW, rank, pretrained=False)


def _plain_vit_with_the_weights_of(enc):
    plain = timm.create_model(ARCH, pretrained=False, img_size=HW, num_classes=0)
    sd = {k.replace("attn.qkv.qkv.", "attn.qkv."): v for k, v in enc.vit.state_dict().items() if "linear_" not in k}
    plain.load_state_dict(sd, strict=True)
    return plain


def test_only_dinov2_tags_take_this_path():
    assert is_dinov2(ARCH) and not is_dinov2("resnet34.tv_in1k") and not is_dinov2("vit_small_patch16_224")


def test_output_is_the_feature_map_of_the_patch_tokens():
    enc = _encoder()
    out = enc(torch.zeros(2, 3, *HW))
    assert isinstance(out, list) and len(out) == 1
    assert out[0].shape == (2, enc.num_features, 4, 7) and enc.num_features == 384


def test_token_layout_survives_the_reshape():
    """Row r, column c of the map is the token of patch (r, c): BEVFormer reads this map by image position."""
    enc = _encoder()
    plain = _plain_vit_with_the_weights_of(enc).eval()
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        got = enc.eval()(x)[0]
        tokens = plain.forward_features(x)[:, plain.num_prefix_tokens:]
    for r, c in ((0, 0), (0, 6), (3, 0), (2, 5)):
        assert torch.equal(got[0, :, r, c], tokens[0, r * 7 + c])


def test_an_untrained_adapter_returns_the_pretrained_output():
    enc = _encoder()
    plain = _plain_vit_with_the_weights_of(enc).eval()
    x = torch.randn(2, 3, *HW)
    with torch.no_grad():
        tokens = plain.forward_features(x)[:, plain.num_prefix_tokens:]
        assert torch.equal(enc.eval()(x)[0], tokens.transpose(1, 2).reshape(2, 384, 4, 7))


def test_only_the_lora_terms_train():
    enc = _encoder(rank=4)
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    want = {f"vit.blocks.{i}.attn.qkv.linear_{m}_{t}.weight" for i in range(12) for m in "ab" for t in "qv"}
    assert trainable == want
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == 12 * 4 * 4 * 384


def test_lora_moves_q_and_v_and_leaves_k():
    torch.manual_seed(1)
    qkv = nn.Linear(8, 24)
    lora = LoRAQKV(qkv, 2)
    nn.init.ones_(lora.linear_b_q.weight)
    nn.init.ones_(lora.linear_b_v.weight)
    x = torch.randn(1, 3, 8)
    q0, k0, v0 = qkv(x).chunk(3, -1)
    q1, k1, v1 = lora(x).chunk(3, -1)
    assert torch.equal(k1, k0) and not torch.equal(q1, q0) and not torch.equal(v1, v0)


def test_there_is_no_batchnorm_and_no_randomness_in_training_mode():
    enc = _encoder()
    nn.init.ones_(enc.vit.blocks[0].attn.qkv.linear_b_q.weight)
    assert not any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in enc.modules())
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        assert torch.equal(enc.train()(x)[0], enc.eval()(x)[0])


def test_a_training_step_moves_the_lora_terms_and_no_pretrained_weight():
    enc = _encoder().train()
    frozen = {n: p.detach().clone() for n, p in enc.named_parameters() if not p.requires_grad}
    params = [p for p in enc.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.1)
    x = torch.randn(2, 3, *HW)
    for step in range(2):      # the A matrices get a zero gradient on the first step, while the B matrices are zero
        opt.zero_grad()
        enc(x)[0].square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)     # plain DDP needs every one
        if step == 0:
            assert all(p.grad.abs().sum() > 0 for n, p in enc.named_parameters() if p.requires_grad and "linear_b" in n)
        opt.step()
    assert all(torch.equal(p, frozen[n]) for n, p in enc.named_parameters() if n in frozen)
    assert all(p.grad.abs().sum() > 0 for n, p in enc.named_parameters() if p.requires_grad and "linear_a" in n)


def test_the_checkpoint_round_trips_strictly_without_the_pretrained_file():
    a, b = _encoder(seed=0), _encoder(seed=1)
    nn.init.ones_(a.vit.blocks[3].attn.qkv.linear_b_v.weight)
    b.load_state_dict(a.state_dict(), strict=True)
    x = torch.randn(1, 3, *HW)
    with torch.no_grad():
        assert torch.equal(a.eval()(x)[0], b.eval()(x)[0])


def test_a_wrong_image_size_stops_the_run():
    with pytest.raises(AssertionError):
        _encoder()(torch.zeros(1, 3, 56, 112))


@pytest.mark.parametrize("hw", [(55, 98), (56, 100)])
def test_a_size_off_the_patch_grid_is_refused(hw):
    with pytest.raises(ValueError, match="multiple of the patch"):
        Dinov2LoRAEncoder(ARCH, hw, 4, pretrained=False)


def test_rank_must_be_positive():
    with pytest.raises(ValueError, match="vit_lora_rank"):
        Dinov2LoRAEncoder(ARCH, HW, 0, pretrained=False)


def test_the_config_gives_the_size_after_the_crop():
    cfg = SimpleNamespace(image_architecture=ARCH, image_size=(98, 70), crop_top=14, vit_lora_rank=2, backbone_pretrained=False)
    enc = Dinov2LoRAEncoder.from_config(cfg)
    assert tuple(enc.vit.patch_embed.img_size) == HW
    with pytest.raises(ValueError, match="needs image_size"):
        Dinov2LoRAEncoder.from_config(SimpleNamespace(image_architecture=ARCH, image_size=None, crop_top=0, vit_lora_rank=2))


# ---------------------------------------------------------------------------------------------- feature builder
def _input(height=108, width=192):
    image = np.zeros((height, width, 3), np.uint8)
    image[height // 2 :, :, 0] = 255        # the bottom half is red, so a wrong resize or crop shows
    camera = SimpleNamespace(image=image, intrinsics=np.array([[100.0, 0.0, 96.0], [0.0, 100.0, 54.0], [0.0, 0.0, 1.0]]),
                             sensor2lidar_rotation=np.eye(3), sensor2lidar_translation=np.zeros(3), distortion=None)
    return SimpleNamespace(cameras=[SimpleNamespace(cam_f0=camera)] * 4)


def _builder(**kw):
    return ParaSSRFeatureBuilder(ParaSSRConfig(camera_names=("cam_f0",), **kw))


def test_image_size_replaces_the_scale_and_keeps_the_projection_true():
    b = _builder(image_size=(98, 70), crop_top=14, image_scale=0.5, undistort=False)
    imgs, l2i, hw = b._get_cameras(_input(), 3)
    assert imgs.shape == (1, 3, 56, 98) and hw.tolist() == [[56, 98]]
    want = build_lidar2img(np.array([[100.0, 0.0, 96.0], [0.0, 100.0, 54.0], [0.0, 0.0, 1.0]]), np.eye(3), np.zeros(3),
                           (98 / 192, 70 / 108), 14)
    np.testing.assert_allclose(l2i[0].numpy(), want, rtol=1e-5, atol=1e-5)


def test_without_image_size_the_scale_path_and_the_cache_name_are_unchanged():
    plain, sized = _builder(image_scale=0.5, crop_top=4, undistort=False), _builder(image_scale=0.5, crop_top=4, undistort=False, image_size=(98, 56))
    assert plain._get_cameras(_input(), 3)[0].shape == (1, 3, 50, 96)
    assert plain.get_unique_name() != sized.get_unique_name()
    # the digest of a config without image_size, computed before this field existed
    assert plain._config.image_size is None and "image_size" not in _fields_of(plain)


def _fields_of(builder):
    import navsim.agents.para_ssr.para_ssr_features as F
    seen = []
    keep = F.cache_key
    F.cache_key = lambda prefix, fields: seen.extend(name for name, _ in fields) or keep(prefix, fields)
    try:
        builder.get_unique_name()
    finally:
        F.cache_key = keep
    return seen


# ------------------------------------------------------------------------------------------------------- model
def _model_config():
    return ParaSSRConfig(
        use_lidar=False, image_architecture=ARCH, backbone_pretrained=False, frozen_stages=-1, vit_lora_rank=2,
        image_size=(98, 56), crop_top=0, bev_h=8, bev_w=8, embed_dims=16, num_heads=2, ffn_channels=32, encoder_num_layers=1,
        encoder_num_points_in_pillar=2, encoder_num_points_sca=2, encoder_attn_dropout=0.0, encoder_ffn_dropout=0.0,
        latent_num_layers=1, num_scenes=2, num_query=4, max_agents=4, det_num_decoder_layers=1, map_num_vec=3,
        map_num_pts_per_vec=4, map_num_decoder_layers=1, use_grid_mask=False, plan_anchor=False,
        camera_names=("cam_f0", "cam_l0", "cam_r0"))


@pytest.mark.parametrize("bn_mode", ["moving", "frozen"])
def test_the_model_trains_through_the_encoder_in_both_bn_modes(bn_mode):
    torch.manual_seed(3)
    model = ParaSSRModel(replace(_model_config(), backbone_bn_mode=bn_mode)).train()
    feats = model.extract_img_feat(torch.randn(1, 3, 3, 56, 98))
    assert feats[0].shape == (1, 3, 16, 4, 7)
    feats[0].square().mean().backward()
    assert model.image_encoder.vit.blocks[0].attn.qkv.linear_b_q.weight.grad is not None
    assert model.image_encoder.vit.blocks[0].attn.qkv.qkv.weight.grad is None
    model.train()       # the train() override must not touch requires_grad on a model with no ResNet stage
    assert not model.image_encoder.vit.pos_embed.requires_grad


def test_a_resnet_stage_freeze_on_a_vit_is_refused():
    with pytest.raises(ValueError, match="frozen_stages"):
        ParaSSRModel(replace(_model_config(), frozen_stages=1))
