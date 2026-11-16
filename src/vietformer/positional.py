"""Positional encoding: cho mô hình biết mỗi token đứng ở vị trí nào trong câu.

Attention tự nó không quan tâm thứ tự: "Chó cắn người" và "Người cắn chó" có cùng một tập token.
Ta cộng thêm vào embedding của token ở vị trí t một vector đặc trưng cho vị trí t, để hai câu trên
có biểu diễn khác nhau. Hai cách phổ biến:

* `SinusoidalPositionalEncoding` (bài báo "Attention Is All You Need", 2017): công thức sin/cos cố
  định, không có tham số học. Mỗi cặp chiều dao động với một tần số khác nhau, giống các kim đồng
  hồ quay nhanh chậm khác nhau, nên mỗi vị trí có một "dấu vân tay" riêng.
* `LearnedPositionalEmbedding` (GPT, GPT-2): một bảng block_size × d_model học như embedding
  thường. Linh hoạt, nhưng không dùng được cho vị trí ≥ block_size.

Cả hai đều nhận độ dài chuỗi T và trả về tensor (T, d_model) để cộng vào embedding (B, T, d_model).
"""

from __future__ import annotations

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))."""

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model phải là số chẵn (chiều chẵn dùng sin, chiều lẻ dùng cos)")
        position = torch.arange(block_size).unsqueeze(1).float()     # (T, 1): 0, 1, 2, ...
        two_i = torch.arange(0, d_model, 2).float()                   # (d/2,): 0, 2, 4, ...
        frequency = 1.0 / (10000 ** (two_i / d_model))                # (d/2,): từ 1 giảm dần tới ~1/10000
        pe = torch.zeros(block_size, d_model)
        pe[:, 0::2] = torch.sin(position * frequency)                 # (T, 1) * (d/2,) -> (T, d/2)
        pe[:, 1::2] = torch.cos(position * frequency)
        # Buffer: được lưu và chuyển thiết bị cùng mô hình, nhưng KHÔNG phải tham số để học.
        self.register_buffer("pe", pe)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.pe.size(0):
            raise ValueError(f"T = {T} vượt quá block_size = {self.pe.size(0)} đã tính sẵn; hãy tạo lại với block_size lớn hơn")
        return self.pe[:T]  # (T, d_model)


class LearnedPositionalEmbedding(nn.Module):
    """Một bảng embedding cho các vị trí 0..block_size-1, học cùng mô hình."""

    def __init__(self, block_size: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(block_size, d_model)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.table.num_embeddings:
            raise ValueError(f"T = {T} vượt quá block_size = {self.table.num_embeddings}: bảng không có vector cho các vị trí đó")
        positions = torch.arange(T, device=self.table.weight.device)  # (T,): 0, 1, ..., T-1
        return self.table(positions)                                  # (T, d_model)
