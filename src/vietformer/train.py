"""Huấn luyện mô hình ngôn ngữ: dự đoán token tiếp theo.

Lấy ngẫu nhiên các đoạn liên tiếp dài block_size trong corpus. Đầu vào x là đoạn đó, nhãn y là
chính đoạn đó dịch sang phải 1 token. Tại MỌI vị trí t, mô hình phải đoán y[t] = x[t+1] chỉ dựa vào
x[0..t] (nhờ causal mask), nên một đoạn dài T cho ta T bài tập cùng lúc. Loss là cross-entropy giữa
phân phối dự đoán và token thật.

Ta luôn tách một phần corpus làm tập validation (val) mà mô hình không được học: train loss giảm
mà val loss tăng nghĩa là mô hình đang học thuộc lòng (overfitting) thay vì học quy luật.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch

from .model import save_checkpoint


def split_train_val(data: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Chia dãy token: (1 - val_fraction) phần đầu để train, phần cuối để val."""
    n_train = int(len(data) * (1 - val_fraction))
    return data[:n_train], data[n_train:]


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device=None, generator: torch.Generator | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lấy ngẫu nhiên batch_size đoạn. Trả về x, y cùng kích thước (B, T); y là x dịch phải 1 token."""
    starts = torch.randint(0, len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in starts])           # (B, T)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in starts])   # (B, T)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, batch_size: int = 32, eval_iters: int = 20, seed: int = 1234) -> float:
    """Loss trung bình trên `eval_iters` batch. Dùng seed cố định để lần đánh giá nào cũng gặp
    đúng các batch đó, nhờ vậy các điểm trên đường loss so sánh được với nhau."""
    was_training = model.training
    model.eval()  # tắt dropout khi đánh giá
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, model.config.block_size, batch_size, device, generator)
        losses.append(model(x, y)[1].item())
    model.train(was_training)
    return sum(losses) / len(losses)


def make_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW; chỉ áp weight decay (kéo trọng số về 0) cho các ma trận, không cho bias và LayerNorm."""
    params = [p for p in model.parameters() if p.requires_grad]   # bỏ qua tham số bị freeze
    groups = [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def lr_at(step: int, max_steps: int, lr: float, warmup_steps: int) -> float:
    """Learning rate: tăng tuyến tính trong warmup_steps bước đầu, sau đó giảm dần theo cosine về lr/10."""
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return lr / 10 + (lr - lr / 10) * 0.5 * (1 + math.cos(math.pi * progress))


def train(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    *,
    max_steps: int = 2000,
    batch_size: int = 32,
    lr: float = 1e-3,
    warmup_steps: int = 100,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 100,
    eval_iters: int = 20,
    checkpoint_path: str | Path | None = None,
    checkpoint_extra: dict | None = None,
    restore_best: bool = True,
    verbose: bool = True,
) -> dict:
    """Vòng lặp huấn luyện. Trả về `history` gồm các bước đánh giá và train/val loss tương ứng.

    Mô hình có val loss thấp nhất được lưu vào `checkpoint_path` (nếu có) và, khi `restore_best`,
    được nạp lại vào `model` khi kết thúc (một dạng "early stopping").
    """
    device = next(model.parameters()).device
    optimizer = make_optimizer(model, lr, weight_decay)
    history = {"step": [], "train_loss": [], "val_loss": [], "best_step": 0, "best_val_loss": float("inf")}
    best_state = None
    start_time = time.time()
    model.train()

    for step in range(max_steps + 1):
        # 1) Thỉnh thoảng đo loss trên train và val
        if step % eval_interval == 0 or step == max_steps:
            train_loss = estimate_loss(model, train_data, batch_size, eval_iters)
            val_loss = estimate_loss(model, val_data, batch_size, eval_iters)
            history["step"].append(step)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"], history["best_step"] = val_loss, step
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if checkpoint_path is not None:
                    save_checkpoint(checkpoint_path, model, history=history, **(checkpoint_extra or {}))
            if verbose:
                print(f"bước {step:5d} | train loss {train_loss:.3f} | val loss {val_loss:.3f} | {time.time() - start_time:5.0f}s")
        if step == max_steps:
            break

        # 2) Một bước huấn luyện: forward -> loss -> backward -> cập nhật trọng số
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, max_steps, lr, warmup_steps)
        x, y = get_batch(train_data, model.config.block_size, batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # chặn gradient quá lớn
        optimizer.step()

    history["train_time_s"] = time.time() - start_time
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
        if checkpoint_path is not None:  # lưu lại để checkpoint chứa đủ lịch sử loss của cả quá trình
            save_checkpoint(checkpoint_path, model, history=history, **(checkpoint_extra or {}))
    if verbose:
        print(f"Xong sau {history['train_time_s']:.0f}s. Val loss tốt nhất {history['best_val_loss']:.3f} ở bước {history['best_step']}.")
    return history
