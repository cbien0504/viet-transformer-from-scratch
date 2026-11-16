"""Sinh văn bản: lặp lại "dự đoán phân phối token tiếp theo → chọn một token → nối vào chuỗi".

Hai núm vặn chính khi chọn token:

* temperature: chia logits cho T trước softmax. T < 1 làm phân phối "nhọn" hơn (an toàn, dễ lặp lại),
  T > 1 làm phân phối "phẳng" hơn (đa dạng nhưng dễ vô nghĩa). T = 0 nghĩa là luôn chọn token có
  xác suất cao nhất (greedy).
* top-k: chỉ giữ k token có điểm cao nhất rồi mới lấy mẫu, loại bỏ "đuôi dài" các token hiếm.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import MiniGPT, load_checkpoint
from .tokenizer import VietTokenizer
from .utils import resolve_path


def next_token_probs(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
    """Biến logits (B, vocab) thành xác suất (B, vocab) sau khi áp temperature và top-k."""
    if temperature <= 0:  # greedy: dồn toàn bộ xác suất vào token điểm cao nhất
        return torch.nn.functional.one_hot(logits.argmax(dim=-1), logits.size(-1)).float()
    logits = logits / temperature
    if top_k is not None and top_k < logits.size(-1):
        kth_best = torch.topk(logits, top_k, dim=-1).values[..., -1:]          # (B, 1): điểm cao thứ k
        logits = logits.masked_fill(logits < kth_best, float("-inf"))        # các token còn lại -> xác suất 0
    return torch.softmax(logits, dim=-1)


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sinh thêm `max_new_tokens` token sau ngữ cảnh idx (B, T). Trả về (B, T + max_new_tokens)."""
    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]        # chỉ giữ block_size token gần nhất
        logits, _ = model(idx_cond)                          # (B, T, vocab)
        probs = next_token_probs(logits[:, -1, :], temperature, top_k)  # chỉ cần dự đoán ở vị trí cuối
        next_id = torch.multinomial(probs, num_samples=1, generator=generator)  # (B, 1)
        idx = torch.cat([idx, next_id], dim=1)
    model.train(was_training)
    return idx


def generate_text(
    model,
    tokenizer,
    prompt: str = "",
    max_new_tokens: int = 60,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int | None = None,
) -> str:
    """Tiện ích: chuỗi gợi ý → chuỗi sinh ra (gồm cả gợi ý). Gợi ý rỗng thì bắt đầu từ token <n>."""
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt) if prompt else [tokenizer.newline_id]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
    out = generate(model, idx, max_new_tokens, temperature, top_k, generator)
    text = tokenizer.decode(out[0].tolist())
    return text if prompt else text.lstrip("\n")


def load_model_and_tokenizer(checkpoint_path: str | Path, device="cpu") -> tuple[MiniGPT, VietTokenizer, dict]:
    """Nạp mô hình từ checkpoint cùng đúng tokenizer đã dùng khi huấn luyện (đường dẫn lưu trong checkpoint)."""
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    tokenizer = VietTokenizer(resolve_path(checkpoint["tokenizer_path"]))
    return model, tokenizer, checkpoint
