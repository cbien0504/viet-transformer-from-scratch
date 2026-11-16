"""vietformer: một mini-GPT (Transformer decoder-only) viết từ đầu bằng PyTorch, dùng cho việc học.

Đọc các module theo đúng thứ tự dòng dữ liệu đi qua mô hình:

    tokenizer → embedding (+ word_vectors) → positional → attention → block → model → train → generate

Các module phụ: utils (seed, thiết bị), data (tải corpus), viz (vẽ hình).
"""

__version__ = "0.1.0"
