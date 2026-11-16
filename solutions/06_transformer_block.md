# Đáp án – Notebook 06 · Khối Transformer

## Bài 1. Khối post-norm

```python
class PostNormBlock(TransformerBlock):
    def forward(self, x):
        x = self.ln1(x + self.attn(x))   # cộng residual TRƯỚC, chuẩn hóa SAU
        x = self.ln2(x + self.ffn(x))
        return x


print(PostNormBlock(64, 4, 32)(torch.randn(2, 5, 64)).shape)   # torch.Size([2, 5, 64])
```

Chạy lại thí nghiệm ở mục 2 với `PostNormBlock`, bạn sẽ thấy gradient ở **mọi** khối nhỏ xíu, cỡ 1e-7 tới 1e-6. Đừng vội kết
luận post-norm "hỏng": đó là một **cái bẫy khi thiết kế thí nghiệm**. Khối post-norm cuối cùng kết thúc bằng LayerNorm,
nên mỗi vector đầu ra luôn có trung bình 0 và phương sai 1. Vì vậy loss `out.pow(2).mean()` gần như **luôn bằng 1** dù
trọng số là gì, và gradient của nó gần bằng 0.

Dùng một loss không bị LayerNorm "triệt tiêu", ví dụ tích với một tensor ngẫu nhiên cố định:

```python
target = torch.randn(4, 16, 64)
(stack(torch.randn(4, 16, 64)) * target).mean().backward()
```

Lúc khởi tạo, cả pre-norm lẫn post-norm khi đó đều cho gradient cỡ 4e-2 tới 9e-2 ở mọi khối: post-norm không làm
gradient biến mất *ngay từ đầu* trong thí nghiệm nhỏ này. Khác biệt thật nằm ở **quá trình huấn luyện**: với post-norm,
mọi tín hiệu đều phải đi qua LayerNorm ở từng khối, gradient ở các lớp gần đầu ra lớn hơn hẳn các lớp khác, nên mô
hình sâu thường cần *warmup* learning rate cẩn thận và dễ mất ổn định (xem Xiong et al., 2020, *On Layer Normalization
in the Transformer Architecture*). Pre-norm giữ một đường residual "sạch" từ đầu vào tới đầu ra nên dễ huấn luyện hơn.

## Bài 2. Vì sao feed-forward phóng to 4 lần?

Lớp ẩn của feed-forward giống như một bộ "phát hiện đặc trưng": mỗi neuron trong `4·d_model` neuron học phản ứng với
một kiểu mẫu nào đó của vector đầu vào, và lớp thứ hai tổng hợp lại. Lớp ẩn rộng hơn thì phát hiện được nhiều kiểu
mẫu hơn, mô hình "nhớ" được nhiều hơn (nhiều nghiên cứu xem feed-forward như một bộ nhớ dạng key–value).

* Không phóng to (`expansion = 1`): feed-forward có `2·d²` tham số, dung lượng nhỏ, và hàm phi tuyến GELU cắt bớt
  thông tin ngay trong một không gian hẹp.
* `expansion = 4`: `8·d²` tham số. Con số 4 là quy ước từ bài báo gốc (`d_model = 512`, `d_ff = 2048`) và được giữ ở
  GPT-2; nó cân bằng giữa dung lượng và chi phí tính toán. Một số mô hình hiện đại dùng tỉ lệ khác, ví dụ 8/3 với SwiGLU.

## Bài 3. Số tham số của `TransformerBlock(d_model = 256)`

| Thành phần | Công thức | d = 256 |
|---|---|---|
| `ln1`, `ln2` | 2 × 2d | 1.024 |
| attention: Q, K, V (không bias) + proj (có bias) | 3d² + d² + d = 4d² + d | 262.400 |
| feed-forward: `Linear(d, 4d)` + `Linear(4d, d)` | (4d² + 4d) + (4d² + d) = 8d² + 5d | 525.568 |
| **Tổng** | **12d² + 10d** | **788.992** |

```python
print(sum(p.numel() for p in TransformerBlock(256, 4, 64).parameters()))   # 788992
```

Phần `12d²` chiếm gần như toàn bộ, nên một khối Transformer có khoảng `12·d_model²` tham số – công thức nhẩm nhanh
hay dùng (GPT-3 175B: 96 khối × 12 × 12288² ≈ 174 tỉ).
