# Đáp án – Notebook 03 · Positional encoding

## Bài 1. Vị trí vượt quá `block_size`

```python
for module in (LearnedPositionalEmbedding(64, 128), SinusoidalPositionalEncoding(64, 128)):
    try:
        module(100)
    except ValueError as error:
        print(type(module).__name__, "->", error)
# LearnedPositionalEmbedding -> T = 100 vượt quá block_size = 64: bảng không có vector cho các vị trí đó
# SinusoidalPositionalEncoding -> T = 100 vượt quá block_size = 64 đã tính sẵn; hãy tạo lại với block_size lớn hơn
```

Cả hai đều báo lỗi, nhưng vì những lý do khác nhau về bản chất:

* **Learned:** bảng chỉ có 64 hàng; vị trí 100 *chưa bao giờ có vector* và mô hình chưa từng học gì về nó. Muốn dùng
  ngữ cảnh dài hơn thì phải tạo bảng lớn hơn **và huấn luyện lại**.
* **Sinusoidal:** chỉ là ta chưa *tính sẵn* tới vị trí 100. Công thức sin/cos cho vector của bất kỳ vị trí nào:

```python
pe = SinusoidalPositionalEncoding(block_size=128, d_model=128)
print(pe(101)[100].shape)   # torch.Size([128]) – vector của vị trí 100
```

(Có vector không có nghĩa là mô hình đã huấn luyện ở độ dài 64 sẽ dùng tốt vị trí 100; khả năng "ngoại suy" độ dài
là một chủ đề nghiên cứu riêng, và là một lý do RoPE, ALiBi ra đời.)

> Ghi chú: lớp `SinusoidalPositionalEncoding` ban đầu chỉ cắt `self.pe[:T]`. Với `T > block_size`, phép cắt của
> PyTorch **không báo lỗi** mà lặng lẽ trả về 64 hàng, và lỗi chỉ lộ ra ở một bước sau, rất khó hiểu. Vì vậy cả hai
> lớp trong repo đều kiểm tra `T` và báo lỗi rõ ràng ngay tại chỗ.

## Bài 2. Vì sao dùng cả sin lẫn cos?

Với mỗi tần số `ω`, cặp `(sin ωp, cos ωp)` là một điểm trên đường tròn đơn vị. Đi thêm `k` vị trí là **xoay** điểm đó
một góc `ωk`:

```
sin(ω(p+k)) =  cos(ωk)·sin(ωp) + sin(ωk)·cos(ωp)
cos(ω(p+k)) = −sin(ωk)·sin(ωp) + cos(ωk)·cos(ωp)
```

Ma trận xoay chỉ phụ thuộc `k`, nên "nhìn về `k` vị trí trước" là một phép biến đổi **tuyến tính** cố định mà mô hình
học được. Nếu chỉ có sin thì `sin(ω(p+k))` cần cả `cos(ωp)` – thông tin không có sẵn – nên không có phép biến đổi
tuyến tính nào như vậy. Ngoài ra chỉ có sin thì không phân biệt được các vị trí có `sin(ωp)` bằng nhau (ví dụ
`sin a = sin(π − a)`); cặp (sin, cos) xác định duy nhất góc trên đường tròn.

## Bài 3. Tích vô hướng chỉ phụ thuộc khoảng cách

```python
pe = SinusoidalPositionalEncoding(64, 128)(64)
for p in (0, 10, 20, 30):
    print(p, (pe[p] @ pe[p + 5]).item())
# 47.185 cho cả bốn giá trị của p
```

Giải thích: với mỗi cặp chiều, `sin(ωp)·sin(ω(p+k)) + cos(ωp)·cos(ω(p+k)) = cos(ωk)`. Cộng qua mọi cặp:
`pe[p] · pe[p+k] = Σ_i cos(ω_i k)` – hoàn toàn không phụ thuộc `p`.
