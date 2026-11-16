# Đáp án – Notebook 08 · Sinh văn bản và trực quan hóa

## Bài 1. Tự viết câu mở đầu

Bài này không có một đáp án duy nhất. Ví dụ:

```python
for prompt, temperature, top_k in [("Nàng rằng:", 0.7, 20), ("Nàng rằng:", 1.0, 50), ("Chàng về", 0.8, 50)]:
    print(f"--- {prompt!r}, temperature={temperature}, top_k={top_k}")
    print(generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=temperature, top_k=top_k, seed=0))
```

Một lần chạy với checkpoint của notebook 07 cho ra (kết quả của bạn sẽ khác nếu mô hình khác):

```text
--- 'Nàng rằng:', temperature=0.7, top_k=20
Nàng rằng: Tôi chẳng xong.
Nàng rằng: Ai ai đã dễ ai?
Người đâu mà lại là hay,
Cũng đến thế này thì thôi cũng là không.

--- 'Nàng rằng:', temperature=1.0, top_k=50
Nàng rằng: Lòng đành thơ.
Nàng rằng: Hương tàn đã dễ mấy thì,
Bấy lâu trước cũng hay chẳng ngoa!
Nàng rằng: Lòng riêng sau phải ra,
```

Kinh nghiệm với mô hình này (khớp với bảng đo ở mục 2 của notebook):

* `temperature` khoảng 0,6–0,8 cùng `top_k` khoảng 20–50 cho kết quả "giống lục bát" nhất: xuống dòng đúng chỗ, độ dài
  câu gần 6–8 chữ, từ ngữ quen thuộc của Truyện Kiều ("Nàng rằng:", "thế này", "bấy lâu").
* `temperature` thấp (≤ 0,3) hoặc greedy: lặp lại cụm từ, chép nguyên văn Truyện Kiều nhiều hơn.
* `temperature` cao (≥ 1,3), nhất là khi không có top-k: xuất hiện "chữ" không có thật do ghép bừa các mảnh âm tiết,
  câu dài ngắn lộn xộn.
* Không cấu hình nào làm câu thơ có nghĩa trọn vẹn: giới hạn nằm ở mô hình (nhỏ, ít dữ liệu), không phải ở cách lấy mẫu.

## Bài 2. Top-p (nucleus) sampling

```python
def top_p_probs(logits, p=0.9, temperature=1.0):
    """Giữ nhóm token nhỏ nhất có tổng xác suất >= p, bỏ phần còn lại rồi chuẩn hóa."""
    probs = next_token_probs(logits, temperature)                   # (1, vocab_size)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
    before = sorted_probs.cumsum(dim=-1) - sorted_probs             # tổng xác suất của các token đứng TRƯỚC
    sorted_probs[before >= p] = 0.0                                 # phía trước đã đủ p thì bỏ
    probs = torch.zeros_like(probs).scatter(-1, order, sorted_probs)  # trả về đúng thứ tự token ban đầu
    return probs / probs.sum(dim=-1, keepdim=True)


print(int((top_p_probs(last_logits, p=0.9) > 0).sum()), "token còn lại sau top-p")
```

Với ngữ cảnh ở mục 1 ("…khéo là"), số token còn lại sau top-p:

| p | 0,5 | 0,9 | 0,95 |
|---|---|---|---|
| số token giữ lại | 62 | 896 | 1.615 |

Nhớ lại mục 1: 10 token mạnh nhất chỉ chiếm khoảng 23% xác suất. MiniGPT còn rất "phân vân", nên muốn gom đủ 90% xác
suất cần tới gần 900 token – nhiều hơn hẳn `top_k = 50`. Với mô hình nhỏ này, top-p 0,9 vì thế cho văn bản lộn xộn
hơn top-k 50; với mô hình lớn (tự tin hơn), top-p 0,9 thường chỉ giữ vài chục token.

Khác biệt chính với top-k: **top-k giữ một số lượng cố định** token, còn **top-p giữ một lượng xác suất cố định**.
Khi mô hình chắc chắn (một token chiếm gần hết xác suất), top-p chỉ giữ một vài token; khi mô hình phân vân, top-p tự
mở rộng ra nhiều token. Vì vậy top-p thích nghi tốt hơn với từng ngữ cảnh và được dùng rộng rãi trong các mô hình lớn
(thường kết hợp `temperature` với `p` khoảng 0,9–0,95).

Chi tiết cài đặt: điều kiện `before >= p` (thay vì `cumsum >= p`) đảm bảo luôn giữ token *làm tổng vượt qua* `p`, và
luôn giữ token mạnh nhất (phía trước nó có tổng bằng 0).

## Bài 3. Head nhìn dấu xuống dòng, head nhìn chữ ngay trước

**(a) Vì sao nhìn về dấu xuống dòng lại hữu ích?** Vector vị trí của MiniGPT là vị trí *trong đoạn 64 token*, không phải
vị trí *trong câu thơ* (đoạn có thể bắt đầu ở giữa một câu). Muốn biết "mình đang ở chữ thứ mấy của câu", mô hình phải
tìm xem câu hiện tại bắt đầu từ đâu – tức là dấu `<n>` gần nhất. Biết điều đó, nó suy ra được:

* nên viết tiếp hay đã đến lúc đặt dấu câu và xuống dòng (câu lục 6 chữ, câu bát 8 chữ, luân phiên nhau);
* câu hiện tại là câu lục hay câu bát (đếm số dấu xuống dòng gần đây);
* chữ đang viết có rơi vào các vị trí "quan trọng" của luật thơ không (chữ 2, 4, 6 theo luật bằng – trắc; chữ 6 và 8
  theo luật vần) – dù mô hình của ta chưa học được luật vần.

Vector vị trí có chu kỳ ~9 token (mục 6 của notebook) là một cách bổ trợ để làm việc này.

**(b) Head "nhìn chữ ngay trước" và induction head.**

* **Thông tin quan trọng nhất để đoán chữ tiếp theo thường là chữ hiện tại và chữ ngay trước.** Nhưng sau attention,
  vector ở vị trí `t` mới chỉ chứa thông tin của chính token `t` (cộng vị trí). Một head nhìn `t − 1` chép thông tin
  của token trước vào vị trí `t`, nhờ vậy các lớp sau "thấy" được cả cặp `(x[t−1], x[t])`: một bigram, một từ ghép hai
  âm tiết như "phong trần", "Thúy Kiều".
* **Induction head** (Olsson et al., 2022) là sự phối hợp hai head ở hai layer:
  1. Head A (layer trước) là head "chữ ngay trước": ở mỗi vị trí `j`, nó ghi lại "token đứng trước tôi là `x[j−1]`".
  2. Head B (layer sau) ở vị trí hiện tại `t` (token `x[t] = A`) đi tìm những vị trí `j` trong quá khứ mà token đứng
     trước chúng cũng là `A`, rồi chép token `x[j]` (= `B`) sang để dự đoán.

  Kết quả là quy tắc "đã thấy `A B` một lần thì lần sau gặp `A` sẽ đoán `B`". Đây là nền tảng của khả năng học trong
  ngữ cảnh (in-context learning) ở các mô hình lớn. Với thơ, cơ chế này giúp mô hình lặp lại tên riêng hay các cụm đã
  xuất hiện trước đó trong đoạn ("Thúy … Kiều", "Kim … Trọng").
* Bạn có thể tìm induction head trong MiniGPT: đưa vào một đoạn token ngẫu nhiên lặp lại hai lần, rồi xem head nào ở
  lần lặp thứ hai nhìn vào "token ngay sau lần xuất hiện trước của token hiện tại" (một đường chéo lệch trên heatmap).
