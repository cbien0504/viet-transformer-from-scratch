# Đáp án – Notebook 01 · Tokenization

## Bài 1. Đếm âm tiết khi trộn NFC và NFD

```python
n_without_nfc = len(set(re.findall(r"\w+", mixed)))
n_with_nfc = len(set(re.findall(r"\w+", normalize_text(mixed))))
print("Không chuẩn hóa:", n_without_nfc, "| Có chuẩn hóa:", n_with_nfc)
# Không chuẩn hóa: 624 | Có chuẩn hóa: 775
```

Con số đúng là **775** (sau chuẩn hóa). Con số 624 sai vì hai lý do chồng lên nhau:

1. Ở các dòng NFD, dấu được lưu thành ký tự tổ hợp riêng (`U+0300`...). Biểu thức `\w` không coi ký tự tổ hợp là
   chữ cái, nên `\w+` **cắt vụn** âm tiết tại mỗi dấu: "mệnh" thành "me" và "nh". Các mảnh như `i`, `o`, `nh`, `t`
   xuất hiện trong rất nhiều âm tiết khác nhau, nên nhiều âm tiết khác nhau bị gộp thành cùng vài mảnh và con số giảm.
2. Cùng một âm tiết xuất hiện ở dòng NFC và dòng NFD sẽ bị đếm như hai chuỗi khác nhau (theo hướng làm con số tăng).

Kết quả cuối thấp hơn hay cao hơn con số đúng tùy dữ liệu, nhưng chắc chắn **sai**. Đó là lý do phải chuẩn hóa NFC
trước mọi bước xử lý.

## Bài 2. Khi nào BPE vẫn sinh ra `<unk>`?

**(a)** Với `character_coverage=1.0`, mọi ký tự xuất hiện trong corpus đều có mặt trong vocab như một token một ký
tự. Một âm tiết lạ luôn có thể được ghép lại từ các mảnh nhỏ hơn, trường hợp xấu nhất là từng ký tự. Vì vậy chỉ cần
mọi *ký tự* của câu đã từng xuất hiện trong Truyện Kiều là không có `<unk>`. Văn bản tiếng Việt thông thường dùng gần
như cùng một bộ chữ cái và dấu nên hầu như không gặp `<unk>`.

**(b)** `<unk>` xuất hiện khi gặp một **ký tự** chưa từng có trong corpus. Truyện Kiều không có chữ số và không có các
chữ `f`, `j`, `w`, `z` (không dùng trong từ thuần Việt):

```python
example = "Năm 2024 có wifi và zalo"
print(tokenizer.encode_as_pieces(example))
# ['▁Năm', '▁', '<unk>', '▁có', '▁', '<unk>', 'i', '<unk>', 'i', '▁và', '▁', '<unk>', 'a', 'l', 'o']
print(tokenizer.decode(tokenizer.encode(example)))
# Năm  ⁇  có  ⁇ i ⁇ i và  ⁇ alo
```

Để ý cả chuỗi `2024` thành **một** token `<unk>`: SentencePiece gộp các ký tự lạ đứng liền nhau. Cách khắc phục trong
thực tế là bật `byte_fallback=True`: ký tự lạ được tách thành các byte UTF-8 (thêm 256 token vào vocab), nên tokenizer
mã hóa được *mọi* văn bản.

## Bài 3. BPE và unigram

```python
with tempfile.TemporaryDirectory() as tmp:
    for model_type in ["bpe", "unigram"]:
        tok = VietTokenizer.train(text, f"{tmp}/{model_type}", vocab_size=1000, model_type=model_type)
        print(model_type, len(tok.encode(text)), [tok.encode_as_pieces(l) for l in lines[:3]])
```

Kết quả (vocab 1000):

| | Tổng số token | Câu đầu |
|---|---|---|
| BPE | 38.106 | `▁Trăm ▁năm ▁trong ▁c õi ▁người ▁ta ,` |
| Unigram | 41.714 | `▁Tră m ▁năm ▁trong ▁c õ i ▁người ▁ta ,` |

Ở đây BPE cho chuỗi ngắn hơn khoảng 9%. Hai thuật toán khác nhau về cách xây vocab:

* **BPE** đi từ dưới lên: bắt đầu từ ký tự, mỗi bước ghép cặp hay gặp nhất.
* **Unigram** đi từ trên xuống: bắt đầu từ một vocab rất lớn rồi bỏ dần những token ít làm giảm xác suất của corpus
  (theo một mô hình xác suất). Khi tách một câu, nó chọn cách tách có xác suất cao nhất.

Không thuật toán nào luôn thắng; nhiều mô hình nổi tiếng dùng unigram (T5, ALBERT), nhiều mô hình khác dùng BPE (GPT-2,
LLaMA). Với vocab lớn (4000 trở lên), cả hai đều gần như học lại toàn bộ âm tiết của Truyện Kiều.
