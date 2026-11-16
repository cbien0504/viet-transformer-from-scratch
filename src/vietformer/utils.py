"""Tiện ích dùng chung: đặt seed để tái lập kết quả, tự chọn thiết bị tính toán, đếm tham số.

Vì sao cần seed? Khởi tạo trọng số, chia batch, dropout và lấy mẫu khi sinh văn bản đều
ngẫu nhiên. Cố định seed giúp bạn chạy lại notebook và nhận đúng các con số như lần trước.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

# Các thư mục của repo (tính từ vị trí file này: src/vietformer/utils.py).
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"


def set_seed(seed: int = 42) -> None:
    """Đặt seed cho random (Python), numpy và torch (CPU lẫn mọi GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer: str | None = None) -> torch.device:
    """Chọn thiết bị: cuda (GPU NVIDIA) > mps (GPU Apple Silicon) > cpu.

    Truyền `prefer="cpu"` (hoặc "cuda", "mps") nếu muốn ép một thiết bị cụ thể.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def portable_path(path: str | Path) -> str:
    """Đường dẫn tương đối so với thư mục gốc repo (nếu nằm trong repo) để checkpoint dùng được ở máy khác."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def resolve_path(path: str | Path) -> Path:
    """Ngược lại với `portable_path`: đường dẫn tương đối được hiểu là tính từ thư mục gốc repo."""
    path = Path(path)
    return path if path.is_absolute() else ROOT_DIR / path


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """Đếm số tham số của mô hình (chỉ đếm tham số được huấn luyện nếu `trainable_only=True`)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
