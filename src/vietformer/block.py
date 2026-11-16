"""Khối Transformer (pre-norm) – viên gạch được xếp chồng N lần để tạo nên mô hình.

    x = x + Attention(LayerNorm(x))     # các token trao đổi thông tin với nhau
    x = x + FeedForward(LayerNorm(x))   # từng token tự "suy nghĩ" riêng về thông tin vừa nhận

* LayerNorm: đưa vector của MỖI token về trung bình 0, phương sai 1 (rồi co giãn bằng tham số
  học được) để giá trị không phình to hay teo nhỏ qua nhiều lớp. "Pre-norm" nghĩa là chuẩn hóa
  TRƯỚC khi vào attention/feed-forward (GPT-2 trở đi), giúp mô hình sâu dễ huấn luyện hơn.
* Residual (x + ...): mỗi khối chỉ cần học phần "bổ sung" vào x; gradient có đường tắt để chảy
  thẳng về các lớp đầu nên mạng sâu không bị mất gradient.
* Feed-forward: Linear phóng to 4 lần → GELU → Linear thu về, áp dụng độc lập cho từng token.
* Dropout: khi huấn luyện, ngẫu nhiên tắt một phần neuron để mô hình không học thuộc lòng.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
    """LayerNorm tự viết (cho kết quả như `nn.LayerNorm`): chuẩn hóa theo chiều cuối (d_model)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))   # gamma: hệ số co giãn, học được
        self.bias = nn.Parameter(torch.zeros(d_model))    # beta: độ dịch, học được

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)                       # (B, T, 1): trung bình của mỗi token
        var = x.var(dim=-1, keepdim=True, unbiased=False)         # (B, T, 1): phương sai của mỗi token
        x_hat = (x - mean) / torch.sqrt(var + self.eps)           # trung bình 0, phương sai 1
        return self.weight * x_hat + self.bias


class FeedForward(nn.Module):
    """Mạng 2 lớp áp dụng cho từng token: d_model → 4·d_model → GELU → d_model."""

    def __init__(self, d_model: int, dropout: float = 0.0, expansion: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, expansion * d_model),   # (B, T, C) -> (B, T, 4C)
            nn.GELU(),                                 # phi tuyến mượt, giống ReLU nhưng không gãy ở 0
            nn.Linear(expansion * d_model, d_model),   # (B, T, 4C) -> (B, T, C)
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Một khối Transformer pre-norm: attention + feed-forward, mỗi phần có residual."""

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_head, block_size, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # (B, T, C): trao đổi thông tin giữa các token
        x = x + self.ffn(self.ln2(x))    # (B, T, C): xử lý riêng từng token
        return x
