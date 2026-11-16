# Đáp án – Notebook 04 · Self-attention

## Bài 1. Tính tay hàng "trong"

Điểm sau khi chia √d: `[1,5; 3; 3]` (token "trong" nhìn được cả ba token nên không có ô nào bị che).

1. Lũy thừa: `e^1,5 ≈ 4,48`, `e^3 ≈ 20,09`, `e^3 ≈ 20,09`. Tổng ≈ `44,66`.
2. Softmax: `4,48 / 44,66 ≈ 0,100`; `20,09 / 44,66 ≈ 0,450`; `0,450`. Tổng = 1.
3. Đầu ra = `0,100·v_Trăm + 0,450·v_năm + 0,450·v_trong`
   = `0,100·[1, 1, 0, 0] + 0,450·[0, 0, 2, 2] + 0,450·[1, 1, 1, 1]`
   = `[0,55; 0,55; 1,35; 1,35]`.

```python
my_weights_row2 = torch.tensor([0.100, 0.450, 0.450])
my_output_row2 = torch.tensor([0.55, 0.55, 1.35, 1.35])
# Trọng số đúng? True | Đầu ra đúng? True
```

Nhận xét: "trong" có điểm bằng nhau (3) với "năm" và với chính nó, nên chia đều phần lớn trọng số cho hai token đó.
"Trăm" có điểm thấp hơn 1,5 nên chỉ nhận khoảng 10%: chênh lệch 1,5 trên thang log tương ứng `e^1,5 ≈ 4,5` lần trên
thang xác suất.

## Bài 2. Bỏ causal mask

```python
head_no_mask = SelfAttentionHead(d_model=128, head_dim=32, block_size=64, causal=False)
with torch.no_grad():
    head_no_mask(embedding(idx))
viz.plot_attention(head_no_mask.last_attn[0], labels, title="Không có causal mask")
```

Heatmap không còn là tam giác dưới: mọi hàng có trọng số ở mọi cột.

Vì sao không được bỏ mask khi huấn luyện mô hình ngôn ngữ? Nhãn ở vị trí `t` chính là token ở vị trí `t + 1`. Không có
mask, vị trí `t` **nhìn thấy luôn đáp án** `x[t+1]` và chỉ cần học "chép token kế bên": train loss tụt về gần 0 rất
nhanh nhưng mô hình chẳng học được gì về ngôn ngữ. Khi sinh văn bản, token `t + 1` chưa tồn tại, nên mô hình hoàn toàn
vô dụng. (Các mô hình *encoder* như BERT không dùng causal mask, vì chúng được huấn luyện với nhiệm vụ khác: đoán các
từ bị che *ở giữa* câu.)

## Bài 3. Điểm bằng nhau, và điểm bị phóng to

**(a)** Mọi điểm bằng nhau thì `e^s` bằng nhau nên trọng số **đều**: `1/n` cho mỗi token được nhìn. Token đầu tiên luôn
có trọng số 1 cho chính nó (nó chỉ nhìn được mỗi mình nó).

**(b)** Nhân điểm với 10 làm chênh lệch giữa các điểm lớn lên 10 lần, nên softmax gần như dồn hết vào điểm lớn nhất:

```python
s = torch.tensor([1.5, 3.0, 3.0])
print(torch.softmax(s, -1))       # tensor([0.1004, 0.4498, 0.4498])
print(torch.softmax(s * 10, -1))  # tensor([1.5e-07, 0.5000, 0.5000])  – "Trăm" gần như biến mất
s = torch.tensor([1.5, 3.0, 2.5])
print(torch.softmax(s * 10, -1))  # tensor([3.0e-07, 0.9933, 0.0067])  – gần như one-hot
```

Đó chính là hiện tượng "bão hòa" mà phép chia √d tránh: khi phân phối gần one-hot, gradient qua softmax gần bằng 0 và
mô hình học rất chậm.
