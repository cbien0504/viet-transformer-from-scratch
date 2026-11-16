# Đáp án – Notebook 02 · Embedding

## Bài 1. Phép tương tự `vua − đàn_ông + đàn_bà`

```python
analogy = wv.phrase_vector("vua") - wv.phrase_vector("đàn_ông") + wv.phrase_vector("đàn_bà")
print(wv.most_similar(analogy, topn=10))
# [('vua', 0.82), ('bà', 0.67), ('Vua', 0.63), ('Bà', 0.56), ('mẹ', 0.55), ('chúa', 0.53), ('vợ', 0.49), ...]
```

Không ra "nữ hoàng" như ví dụ kinh điển của word2vec tiếng Anh (`king − man + woman ≈ queen`). Lý do:

1. **Chính từ đầu vào thắng.** Vector kết quả vẫn gần `vua` nhất. Trong các bài viết về analogy, người ta luôn *loại*
   các từ đầu vào (`vua`, `đàn_ông`, `đàn_bà`) khỏi danh sách ứng viên. Bỏ chúng đi, ta còn `mẹ`, `chúa`, `vợ`: hướng
   "giống cái" đã được cộng vào, nhưng chưa tới đích.
2. **Mức âm tiết.** fastText tiếng Việt không có vector cho `nữ_hoàng`, `đàn_ông`, `đàn_bà`. Ta lấy trung bình các âm
   tiết, nên `đàn_bà` chứa cả nghĩa của "đàn" (cây đàn, bầy đàn). Đáp án mong muốn, `nữ_hoàng`, cũng không phải là một
   mục trong bảng nên không bao giờ xuất hiện trong danh sách.
3. Ngay cả với tiếng Anh, phép tương tự "đẹp" như sách vở chỉ đúng với một số ít ví dụ được chọn lọc.

Thử các biến thể mức âm tiết sẽ cho kết quả dễ hiểu hơn, ví dụ `vua − ông + bà` → `bà`, `mẹ`, `vua`, `mụ`, `vợ`, `dì`...

## Bài 2. word2vec tự huấn luyện trên Truyện Kiều

```python
from vietformer.data import load_corpus
from vietformer.word_vectors import train_word2vec

w2v = train_word2vec(load_corpus())
for w in ["nàng", "hoa", "trăng"]:
    print(w, w2v.most_similar(w, 6))
```

| Từ | word2vec trên Truyện Kiều | fastText |
|---|---|---|
| nàng | giam, trú, tạm, di, sỡ, tê | hắn, chàng, nương, ngươi |
| hoa | thải, nùng, thềm, ngõ, cợt, bướm | đoá, bông, sen, cành, cúc |
| trăng | vầng, vằng, mát, quạnh, khuyết, quẽ | vầng, rằm, nguyệt, khuya, trời |

* word2vec học từ ~23 nghìn âm tiết nên **nhiễu**: phần lớn láng giềng của `nàng`, `hoa` gần như ngẫu nhiên.
* Láng giềng tốt nhất của word2vec là các âm tiết hay **đứng cạnh** nhau trong Truyện Kiều: `vầng trăng`,
  `trăng khuyết`, `quạnh quẽ`. Với dữ liệu ít, mô hình học được *kết hợp từ* (collocation) trước khi học được *nghĩa*.
* fastText cho láng giềng **cùng loại** (`hoa` → `sen`, `cúc`) nhờ hàng tỉ chữ. Phương án dự phòng chỉ giúp minh họa
  cơ chế; nếu có thể, hãy dùng fastText.

## Bài 3. Đếm tham số được huấn luyện

`Linear(a, b)` có `a·b + b` tham số; `nn.Embedding(V, d)` có `V·d`.

| Biến thể | Bảng embedding | Lớp chiếu 300 → 128 (không bias) | Lớp đầu ra `Linear(128, 4000)` | Tổng được học |
|---|---|---|---|---|
| (a) học từ đầu | 4000·128 = 512.000 | – | 128·4000 + 4000 = 516.000 | **1.028.000** |
| (b) freeze | 4000·300 = 1.200.000 (đóng băng, không tính) | 300·128 = 38.400 | 516.000 | **554.400** |
| (b) finetune | 1.200.000 | 38.400 | 516.000 | **1.754.400** |

Khớp đúng với con số notebook in ra. Để ý biến thể finetune có nhiều tham số được học nhất, và cũng overfit sớm nhất.
