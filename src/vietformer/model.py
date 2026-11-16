"""Mô hình MiniGPT: ghép mọi thành phần thành một Transformer decoder-only hoàn chỉnh.

    token id (B, T)
      → TokenEmbedding + positional encoding          (B, T, d_model)
      → Dropout
      → N × TransformerBlock                          (B, T, d_model)
      → LayerNorm cuối
      → Linear (lm_head)                              (B, T, vocab_size)  = logits

Logits ở vị trí t là "điểm số" cho từng token trong vocab làm token thứ t+1. Softmax các điểm số
này cho ra phân phối xác suất của token tiếp theo. Toàn bộ siêu tham số nằm trong MỘT dataclass
duy nhất là `GPTConfig`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .block import LayerNorm, TransformerBlock
from .embedding import TokenEmbedding
from .positional import LearnedPositionalEmbedding, SinusoidalPositionalEncoding


@dataclass
class GPTConfig:
    vocab_size: int = 4000              # số token trong vocab (phải khớp với tokenizer)
    block_size: int = 64                # độ dài ngữ cảnh tối đa (số token mô hình nhìn được)
    n_layer: int = 4                    # số khối Transformer xếp chồng
    n_head: int = 4                     # số head trong mỗi multi-head attention
    d_model: int = 128                  # số chiều của vector biểu diễn mỗi token
    dropout: float = 0.1                # tỉ lệ dropout khi huấn luyện
    pos_encoding: str = "learned"       # "learned" (kiểu GPT) hoặc "sinusoidal" (bài báo gốc)
    pretrained_dim: int | None = None   # None: embedding học từ đầu; 300: bảng 300 chiều cho fastText
    freeze_embedding: bool = False      # True: không cập nhật bảng embedding (thường dùng với vector có sẵn)


class MiniGPT(nn.Module):
    """Transformer decoder-only nhỏ: dự đoán token tiếp theo tại mọi vị trí."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = TokenEmbedding(config.vocab_size, config.d_model, config.pretrained_dim, config.freeze_embedding)
        if config.pos_encoding == "learned":
            self.pos_emb = LearnedPositionalEmbedding(config.block_size, config.d_model)
        elif config.pos_encoding == "sinusoidal":
            self.pos_emb = SinusoidalPositionalEncoding(config.block_size, config.d_model)
        else:
            raise ValueError(f"pos_encoding không hợp lệ: {config.pos_encoding!r}")
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_head, config.block_size, config.dropout) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Khởi tạo kiểu GPT-2: trọng số nhỏ quanh 0 để lúc đầu mọi token gần như đồng xác suất.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token id. targets: (B, T) token đúng ở vị trí kế tiếp (hoặc None).

        Trả về (logits (B, T, vocab_size), loss hoặc None).
        """
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"Chuỗi dài {T} vượt quá block_size={self.config.block_size}")
        x = self.tok_emb(idx) + self.pos_emb(T)   # (B, T, C) + (T, C) -> (B, T, C) (broadcast theo B)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)                          # (B, T, C)
        x = self.ln_f(x)                          # (B, T, C)
        logits = self.lm_head(x)                  # (B, T, C) -> (B, T, vocab_size)

        loss = None
        if targets is not None:
            # cross_entropy cần dạng (N, vocab) và (N,): gộp B và T thành một trục.
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    def attention_maps(self) -> list[torch.Tensor]:
        """Trọng số attention của lần forward gần nhất: list n_layer phần tử, mỗi phần tử (B, n_head, T, T)."""
        return [block.attn.last_attn for block in self.blocks]


def save_checkpoint(path: str | Path, model: MiniGPT, **extra) -> None:
    """Lưu cấu hình + trọng số (+ thông tin thêm như đường dẫn tokenizer, lịch sử loss)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.config), "model_state": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[MiniGPT, dict]:
    """Tạo lại mô hình từ checkpoint. Trả về (model ở chế độ eval, dict checkpoint)."""
    checkpoint = torch.load(path, map_location=device)
    model = MiniGPT(GPTConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint
