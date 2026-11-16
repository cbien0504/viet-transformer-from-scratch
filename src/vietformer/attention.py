"""Self-attention: cơ chế để mỗi token "nhìn" các token khác và lấy thông tin từ chúng.

Mỗi token tạo ra ba vector từ embedding của nó:

* Query (Q) – "tôi đang tìm gì?"
* Key   (K) – "tôi chứa gì?"  (để các token khác so khớp)
* Value (V) – "nếu được chọn, tôi đưa ra thông tin gì?"

Điểm chú ý của token i dành cho token j là tích vô hướng q_i · k_j, chia cho √d để giá trị không
quá lớn, rồi softmax theo hàng để thành trọng số dương có tổng bằng 1. Đầu ra của token i là trung
bình có trọng số của các value. Causal mask cấm token i nhìn token j > i (tương lai), vì khi sinh
văn bản ta chưa biết các token phía sau.

    Attention(Q, K, V) = softmax(Q·Kᵀ / √d + mask) · V
"""

from __future__ import annotations

import math

import torch
from torch import nn


def causal_mask(T: int, device=None) -> torch.Tensor:
    """Ma trận tam giác dưới (T, T): True ở (i, j) nghĩa là token i ĐƯỢC nhìn token j (j ≤ i)."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None, dropout: nn.Module | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(q·kᵀ / √d) · v, viết tường minh từng bước.

    q, k, v: (..., T, d) – "..." có thể là (B,) hoặc (B, n_head).
    mask:    (T, T) kiểu bool, True = giữ, False = chặn.
    Trả về (out (..., T, d), weights (..., T, T)); mỗi hàng của weights có tổng bằng 1.
    """
    d = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)            # (..., T, d) @ (..., d, T) -> (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))      # vị trí bị chặn -> -inf -> softmax = 0
    weights = torch.softmax(scores, dim=-1)                    # chuẩn hóa theo hàng: tổng mỗi hàng = 1
    attn = dropout(weights) if dropout is not None else weights
    out = attn @ v                                             # (..., T, T) @ (..., T, d) -> (..., T, d)
    return out, weights


class SelfAttentionHead(nn.Module):
    """Một head self-attention: chiếu x thành Q, K, V (head_dim chiều) rồi tính attention."""

    def __init__(self, d_model: int, head_dim: int, block_size: int, dropout: float = 0.0, causal: bool = True):
        super().__init__()
        self.query = nn.Linear(d_model, head_dim, bias=False)   # W_Q
        self.key = nn.Linear(d_model, head_dim, bias=False)     # W_K
        self.value = nn.Linear(d_model, head_dim, bias=False)   # W_V
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(block_size))
        self.last_attn: torch.Tensor | None = None  # trọng số attention lần chạy gần nhất (để vẽ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)                                              # x: (B, T, d_model)
        q, k, v = self.query(x), self.key(x), self.value(x)        # mỗi cái (B, T, head_dim)
        mask = self.mask[:T, :T] if self.causal else None
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.last_attn = weights.detach()                          # (B, T, T)
        return out                                                 # (B, T, head_dim)


class MultiHeadAttention(nn.Module):
    """Nhiều head chạy song song bằng reshape/transpose, rồi trộn lại bằng một lớp Linear.

    Về mặt toán học, cách này giống hệt việc tạo n_head đối tượng `SelfAttentionHead` rồi nối
    (concat) đầu ra (notebook 05 kiểm chứng điều đó), nhưng chỉ cần vài phép nhân ma trận lớn.
    """

    def __init__(self, d_model: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model phải chia hết cho n_head")
        self.n_head = n_head
        self.head_dim = d_model // n_head
        # Q, K, V của TẤT CẢ các head trong một phép nhân: head h dùng các chiều [h*head_dim, (h+1)*head_dim).
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)  # trộn thông tin từ các head
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", causal_mask(block_size))
        self.last_attn: torch.Tensor | None = None  # (B, n_head, T, T) của lần chạy gần nhất

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x = x.view(B, T, self.n_head, self.head_dim)  # (B, T, C) -> (B, T, n_head, head_dim)
        return x.transpose(1, 2)                      # -> (B, n_head, T, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                                       # x: (B, T, C) với C = d_model
        q = self.split_heads(self.query(x))                     # (B, n_head, T, head_dim)
        k = self.split_heads(self.key(x))                       # (B, n_head, T, head_dim)
        v = self.split_heads(self.value(x))                     # (B, n_head, T, head_dim)
        out, weights = scaled_dot_product_attention(q, k, v, self.mask[:T, :T], self.attn_dropout)
        self.last_attn = weights.detach()                       # (B, n_head, T, T)
        out = out.transpose(1, 2)                               # (B, n_head, T, head_dim) -> (B, T, n_head, head_dim)
        out = out.contiguous().view(B, T, C)                    # -> (B, T, C): nối các head lại
        return self.resid_dropout(self.proj(out))               # (B, T, C)
