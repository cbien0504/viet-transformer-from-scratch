"""Embedding: biến mỗi token id (một số nguyên) thành một vector số thực để mạng nơ-ron xử lý.

`nn.Embedding(vocab_size, dim)` thực chất chỉ là một ma trận kích thước vocab_size × dim: token id
i lấy ra hàng thứ i. Việc này tương đương nhân vector one-hot với ma trận, nhưng nhanh hơn nhiều.
Các hàng là tham số nên được cập nhật bằng gradient như mọi trọng số khác: sau khi huấn luyện,
token dùng trong ngữ cảnh giống nhau thường có vector gần nhau.

Hai thử nghiệm của notebook 02 dùng cùng một lớp `TokenEmbedding`:

* (a) học từ đầu: bảng vocab_size × d_model khởi tạo ngẫu nhiên, học cùng mô hình;
* (b) khởi tạo từ vector tiếng Việt có sẵn (fastText 300 chiều hoặc word2vec): bảng
  vocab_size × 300 rồi một lớp Linear chiếu 300 → d_model. `freeze=True` giữ nguyên bảng gốc và
  chỉ học lớp chiếu; `freeze=False` cho phép tinh chỉnh (finetune) cả bảng.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Bảng tra cứu token id → vector d_model chiều (có thể khởi tạo từ vector có sẵn)."""

    def __init__(self, vocab_size: int, d_model: int, pretrained_dim: int | None = None, freeze: bool = False):
        super().__init__()
        table_dim = pretrained_dim or d_model
        self.table = nn.Embedding(vocab_size, table_dim)
        # Bảng có số chiều khác d_model (ví dụ 300 ≠ 128) thì cần một lớp chiếu tuyến tính.
        self.proj = nn.Linear(table_dim, d_model, bias=False) if table_dim != d_model else nn.Identity()
        self.table.weight.requires_grad_(not freeze)  # freeze: bảng không nhận gradient

    @property
    def frozen(self) -> bool:
        return not self.table.weight.requires_grad

    def load_vectors(self, vectors: torch.Tensor) -> None:
        """Chép ma trận vector có sẵn (vocab_size × pretrained_dim) vào bảng embedding."""
        if tuple(vectors.shape) != tuple(self.table.weight.shape):
            raise ValueError(f"Cần ma trận {tuple(self.table.weight.shape)}, nhận được {tuple(vectors.shape)}")
        with torch.no_grad():
            self.table.weight.copy_(vectors)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.table(idx)   # (B, T) -> (B, T, table_dim): tra hàng của từng token id
        return self.proj(x)   # (B, T, table_dim) -> (B, T, d_model)
