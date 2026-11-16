import pytest
import torch

from vietformer.block import FeedForward, LayerNorm, TransformerBlock
from vietformer.positional import LearnedPositionalEmbedding, SinusoidalPositionalEncoding


def test_sinusoidal_shape_and_known_values():
    pe = SinusoidalPositionalEncoding(block_size=50, d_model=16)
    out = pe(10)
    assert out.shape == (10, 16)
    assert torch.allclose(out[0, 0::2], torch.zeros(8))          # sin(0) = 0
    assert torch.allclose(out[0, 1::2], torch.ones(8))           # cos(0) = 1
    assert out.abs().max() <= 1.0
    assert len(list(pe.parameters())) == 0                       # không có tham số học


def test_sinusoidal_positions_are_all_different():
    pe = SinusoidalPositionalEncoding(block_size=64, d_model=32)(64)
    distances = torch.cdist(pe, pe) + torch.eye(64)              # cộng eye để bỏ qua đường chéo
    assert distances.min() > 0.1


def test_learned_positional_shape_and_trainable():
    pe = LearnedPositionalEmbedding(block_size=50, d_model=16)
    assert pe(10).shape == (10, 16)
    assert sum(p.numel() for p in pe.parameters()) == 50 * 16


@pytest.mark.parametrize("module", [SinusoidalPositionalEncoding(8, 16), LearnedPositionalEmbedding(8, 16)])
def test_positions_beyond_block_size_raise_clear_error(module):
    assert module(8).shape == (8, 16)
    with pytest.raises(ValueError, match="block_size"):
        module(9)


def test_layernorm_matches_pytorch():
    x = torch.randn(2, 5, 32) * 3 + 1
    ours, reference = LayerNorm(32), torch.nn.LayerNorm(32)
    assert torch.allclose(ours(x), reference(x), atol=1e-5)
    y = ours(x)
    assert torch.allclose(y.mean(-1), torch.zeros(2, 5), atol=1e-5)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(2, 5), atol=1e-3)


def test_feedforward_and_block_keep_shape():
    x = torch.randn(2, 7, 32)
    assert FeedForward(32)(x).shape == (2, 7, 32)
    assert TransformerBlock(d_model=32, n_head=4, block_size=16)(x).shape == (2, 7, 32)


def test_block_is_causal():
    torch.manual_seed(0)
    block = TransformerBlock(d_model=32, n_head=4, block_size=16).eval()
    x = torch.randn(1, 8, 32)
    x_changed = x.clone()
    x_changed[:, 5:] += 1.0
    assert torch.allclose(block(x)[:, :5], block(x_changed)[:, :5], atol=1e-6)
