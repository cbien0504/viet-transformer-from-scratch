# Đáp án – Notebook 07 · Huấn luyện mini-GPT

## Bài 1. Sinusoidal hay learned?

```python
histories_pos = {}
for pos in ["learned", "sinusoidal"]:
    set_seed(42)
    m = MiniGPT(GPTConfig(vocab_size=tokenizer.vocab_size, pos_encoding=pos)).to(device)
    histories_pos[pos] = train(m, train_data, val_data, max_steps=600, lr=LR, eval_interval=100, verbose=False)
    print(pos, round(histories_pos[pos]["best_val_loss"], 3), "ở bước", histories_pos[pos]["best_step"])
viz.plot_loss_comparison(histories_pos, title="Learned vs sinusoidal")
plt.show()
```

Kết quả một lần chạy (CPU, seed 42):

| bước | 0 | 100 | 200 | 300 | 400 | 500 | 600 |
|---|---|---|---|---|---|---|---|
| val loss – learned | 8,35 | 6,57 | 5,77 | 5,58 | 5,46 | 5,41 | **5,40** |
| val loss – sinusoidal | 8,32 | 6,86 | 6,47 | 6,48 | 6,47 | 6,22 | 6,16 |

Learned học nhanh hơn hẳn. Sinusoidal "kẹt" quanh 6,47 suốt từ bước 200 tới 400 rồi mới giảm tiếp; với nhiều bước hơn
(hoặc learning rate lớn hơn) nó sẽ bắt kịp phần nào: trong một thử nghiệm khác với learning rate 1e-3 và 600 bước, val
loss tốt nhất của hai cách gần nhau (khoảng 5,47 và 5,50).

**Vì sao sinusoidal học chậm hơn lúc đầu?** Giá trị sin/cos nằm trong khoảng `[−1, 1]`, nên vector vị trí có độ dài
khoảng `√(d/2) = 8`. Trong khi đó embedding token được khởi tạo rất nhỏ (độ lệch chuẩn 0,02, độ dài khoảng 0,23). Lúc
đầu, đầu vào của khối Transformer gần như **chỉ chứa thông tin vị trí**; danh tính của token chỉ là một nhiễu nhỏ, và
mô hình phải mất nhiều bước để "phóng to" embedding token lên. Learned positional embedding được khởi tạo cùng thang
đo 0,02 với token nên hai nguồn thông tin cân bằng ngay từ đầu. Bài báo gốc xử lý chuyện này bằng cách nhân embedding
token với `√d_model` trước khi cộng vị trí.

## Bài 2. Giảm overfitting

Các cách thường dùng:

1. **Thêm dữ liệu** – cách hiệu quả nhất (ví dụ ghép thêm các truyện thơ khác, dùng `--corpus`).
2. **Tăng dropout** (0,2–0,3).
3. **Mô hình nhỏ hơn**: ít khối hơn, `d_model` nhỏ hơn – ít tham số để "học thuộc".
4. **Weight decay** mạnh hơn.
5. **Early stopping** – dừng (hoặc giữ checkpoint) ở bước có val loss thấp nhất; `train` đã làm việc này.
6. **Bớt tham số được học ở embedding**: dùng vector có sẵn và **freeze** (mục 6 cho thấy cách này giảm val loss rõ
   nhất), hoặc *weight tying* (dùng chung ma trận cho embedding và `lm_head`, bớt 512 nghìn tham số).

Ví dụ thử hai cách, cùng số bước và learning rate như mục 4:

```python
for name, cfg in [("dropout 0,3", GPTConfig(vocab_size=tokenizer.vocab_size, dropout=0.3)),
                  ("d_model 64", GPTConfig(vocab_size=tokenizer.vocab_size, d_model=64))]:
    set_seed(42)
    m = MiniGPT(cfg).to(device)
    h = train(m, train_data, val_data, max_steps=MAX_STEPS, lr=LR, eval_interval=100, verbose=False)
    print(f"{name}: val tốt nhất {h['best_val_loss']:.3f} ở bước {h['best_step']}, val cuối {h['val_loss'][-1]:.3f}")
```

Kết quả một lần chạy, so với mô hình gốc của mục 4:

| Cấu hình | Val loss tốt nhất | Train loss cuối | Khoảng cách train – val |
|---|---|---|---|
| gốc (dropout 0,1, `d_model` 128) | 5,37 (bước 700) | 3,53 | 1,86 |
| dropout 0,3 | 5,44 (bước 1000) | 4,11 | 1,34 |
| `d_model` 64 | 5,55 (bước 1000) | 4,70 | 0,85 |
| embedding fastText freeze (mục 6) | **5,11** (bước 1000) | – | – |

Đọc kết quả:

* Cả hai cách đều **thu hẹp khoảng cách** giữa train và val, tức là bớt học thuộc. Nhưng trong 1.000 bước, val loss tốt
  nhất của chúng chưa thấp hơn mô hình gốc: mô hình học chậm hơn (val loss vẫn đang giảm ở bước cuối). Tăng số bước có
  thể giúp chúng vượt lên.
* Giảm overfitting không tự động có nghĩa là val loss tốt hơn: một mô hình quá nhỏ có thể không học đủ (*underfitting*).
* Trong repo này, cách hiệu quả nhất là dùng vector fastText và đóng băng: vừa bớt tham số được học, vừa mang thêm tri
  thức từ bên ngoài.

## Bài 3. Đếm tham số MiniGPT nhỏ

`n_layer = 2`, `d_model = 64`, `vocab_size = 4000`, `block_size = 32`:

| Thành phần | Công thức | Số tham số |
|---|---|---|
| token embedding | `vocab × d` = 4000 · 64 | 256.000 |
| positional embedding | `block_size × d` = 32 · 64 | 2.048 |
| 2 khối Transformer | `2 × (12d² + 10d)` = 2 × 49.792 | 99.584 |
| LayerNorm cuối | `2d` | 128 |
| `lm_head` | `d · vocab + vocab` = 64 · 4000 + 4000 | 260.000 |
| **Tổng** | | **617.760** |

```python
cfg = GPTConfig(vocab_size=4000, n_layer=2, n_head=2, d_model=64, block_size=32)
print(count_parameters(MiniGPT(cfg)))   # 617760
```

Đây chính là preset `--preset tiny` của `scripts/train.py`. Để ý: các khối Transformer chỉ chiếm 16%; embedding và
`lm_head` chiếm hơn 80%. Muốn mô hình "thông minh" hơn mà không tăng nhiều tham số, nên dồn tham số vào các khối
(tăng `d_model`, `n_layer`) và giữ vocab vừa phải.
