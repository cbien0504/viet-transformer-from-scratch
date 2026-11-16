import torch
import torch.nn.functional as F

from vietformer.attention import MultiHeadAttention, SelfAttentionHead, causal_mask, scaled_dot_product_attention


def test_causal_mask_is_lower_triangular():
    mask = causal_mask(4)
    assert mask.shape == (4, 4) and mask.dtype == torch.bool
    assert torch.equal(mask, torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool))


def test_attention_weights_sum_to_one_and_respect_mask():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 5, 8).unbind(0)                 # mỗi cái (B=2, T=5, d=8)
    out, weights = scaled_dot_product_attention(q, k, v, causal_mask(5))
    assert out.shape == (2, 5, 8) and weights.shape == (2, 5, 5)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 5))
    assert torch.all(weights[:, ~causal_mask(5)] == 0)          # không có trọng số nào cho tương lai


def test_matches_pytorch_reference_implementation():
    torch.manual_seed(0)
    q, k, v = torch.randn(3, 2, 4, 6, 16).unbind(0)              # (B, heads, T, d)
    ours, _ = scaled_dot_product_attention(q, k, v, causal_mask(6))
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_head_and_multihead_output_shapes():
    x = torch.randn(2, 7, 32)                                    # (B, T, d_model)
    head = SelfAttentionHead(d_model=32, head_dim=8, block_size=16)
    assert head(x).shape == (2, 7, 8)
    assert head.last_attn.shape == (2, 7, 7)
    mha = MultiHeadAttention(d_model=32, n_head=4, block_size=16)
    assert mha(x).shape == (2, 7, 32)
    assert mha.last_attn.shape == (2, 4, 7, 7)
    assert torch.allclose(mha.last_attn.sum(-1), torch.ones(2, 4, 7))


def test_future_tokens_do_not_change_past_outputs():
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, n_head=4, block_size=16).eval()
    x = torch.randn(1, 10, 32)
    x_changed = x.clone()
    x_changed[:, 6:] = torch.randn(1, 4, 32)                     # chỉ đổi các token từ vị trí 6 trở đi
    assert torch.allclose(mha(x)[:, :6], mha(x_changed)[:, :6], atol=1e-6)
    assert not torch.allclose(mha(x)[:, 6:], mha(x_changed)[:, 6:])


def test_multihead_equals_concatenated_single_heads():
    torch.manual_seed(0)
    d_model, n_head, block_size = 16, 4, 8
    head_dim = d_model // n_head
    mha = MultiHeadAttention(d_model, n_head, block_size).eval()
    heads = [SelfAttentionHead(d_model, head_dim, block_size).eval() for _ in range(n_head)]
    with torch.no_grad():  # head h dùng các hàng [h*head_dim, (h+1)*head_dim) của ma trận lớn
        for h, head in enumerate(heads):
            rows = slice(h * head_dim, (h + 1) * head_dim)
            head.query.weight.copy_(mha.query.weight[rows])
            head.key.weight.copy_(mha.key.weight[rows])
            head.value.weight.copy_(mha.value.weight[rows])
    x = torch.randn(2, block_size, d_model)
    concatenated = torch.cat([head(x) for head in heads], dim=-1)  # (B, T, d_model)
    assert torch.allclose(mha(x), mha.proj(concatenated), atol=1e-6)
