# Đáp án – Notebook 05 · Multi-head attention

## Bài 1. Shape với B = 3, T = 10, d_model = 64, n_head = 8

`head_dim = 64 / 8 = 8`.

| Bước | Shape | Giải thích |
|---|---|---|
| `q = query(x)` | `(3, 10, 64)` | `(B, T, C)` – Q của cả 8 head nằm chung trong 64 chiều |
| `split_heads(q)` | `(3, 8, 10, 8)` | `(B, n_head, T, head_dim)` |
| `scores = q @ kᵀ` | `(3, 8, 10, 10)` | `(B, n_head, T, T)` – mỗi head một ma trận 10 × 10 |
| đầu ra sau khi nối head | `(3, 10, 64)` | `(B, T, C)` – giống hệt shape đầu vào |

```python
predicted = {
    "q sau query(x)": (3, 10, 64),
    "q sau split_heads": (3, 8, 10, 8),
    "scores = q @ kᵀ": (3, 8, 10, 10),
    "out sau khi nối head": (3, 10, 64),
}
```

## Bài 2. `d_model = 30`, `n_head = 4`

```python
try:
    MultiHeadAttention(d_model=30, n_head=4, block_size=16)
except ValueError as error:
    print(error)   # d_model phải chia hết cho n_head
```

Mỗi head nhận đúng `d_model / n_head` chiều liên tiếp của vector Q, K, V. `30 / 4 = 7,5` không phải số nguyên nên
không thể `view(B, T, 4, 7.5)`. Các giá trị hợp lệ của `n_head` là các ước của 30: **1, 2, 3, 5, 6, 10, 15, 30**.
Trong thực tế người ta chọn `d_model` là bội của `n_head` và `head_dim` thường là 64 hoặc 128.

## Bài 3. Lớp `proj` có cần không?

Có. Không có `proj` thì đầu ra chỉ là phép nối: chiều 0 tới `head_dim − 1` của đầu ra **chỉ** đến từ head 0, các chiều
tiếp theo chỉ từ head 1, và cứ thế. Hệ quả:

* Thông tin của các head không được trộn với nhau ngay trong khối attention.
* Đầu ra được cộng vào residual (`x + attn(...)`), nên head `h` chỉ ghi được vào một "lát" cố định của vector `x`.
  Có `proj`, mỗi head ghi được vào **mọi** chiều của `x`, theo bất kỳ hướng nào nó cần.

Trong bài báo gốc, `proj` chính là ma trận `W^O`: `MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O`. (Ta
cũng có thể nhìn `proj` như việc mỗi head có một ma trận ra riêng `W^O_h`, rồi cộng kết quả các head lại.)
