# Dữ liệu

Thư mục này **không** chứa dữ liệu trong git: mọi file được tải về hoặc sinh ra khi chạy, và đã nằm trong `.gitignore`.

```bash
python scripts/download_data.py              # Truyện Kiều + tokenizer (vài giây)
python scripts/download_data.py --fasttext   # thêm vector fastText rút gọn (~35 MB tải về, ~10 giây)
```

Các notebook cũng tự tải những gì còn thiếu khi chạy lần đầu.

## 1. Corpus: Truyện Kiều (bắt buộc)

| | |
|---|---|
| Tác phẩm | *Truyện Kiều* (*Đoạn trường tân thanh*), Nguyễn Du (1766–1820) |
| Nguồn | Wikisource tiếng Việt, trang [Truyện Kiều](https://vi.wikisource.org/wiki/Truy%E1%BB%87n_Ki%E1%BB%81u), lấy qua MediaWiki API |
| Giấy phép | **Phạm vi công cộng** – trang được gắn nhãn `{{PD-old}}` (tác giả mất đã hơn 200 năm) |
| Kích thước | 3.254 câu lục bát, 22.778 âm tiết, ~104 nghìn ký tự, ~136 KB |
| File sinh ra | `truyen_kieu.txt` – mỗi câu thơ một dòng, đã chuẩn hóa Unicode NFC |

Cách làm sạch (hàm `clean_poem_wikitext` trong `src/vietformer/data.py`): lấy phần trong thẻ `<poem>…</poem>`, bỏ
template đánh số câu `{{số|N}}`, chú thích `<ref>`, liên kết wiki và định dạng in đậm/nghiêng, chuẩn hóa NFC, bỏ
khoảng trắng thừa. Văn bản được giữ nguyên chính tả gốc, kể cả vài lỗi gõ sẵn có (ví dụ `dờỉ`) và việc trộn hai kiểu
đặt dấu thanh (`hoà` / `hòa`) – đó cũng là một bài học về dữ liệu thật.

## 2. Tokenizer (sinh ra từ corpus)

`truyen_kieu_bpe4000.model` / `.vocab`: tokenizer SentencePiece BPE 4.000 token, huấn luyện ở notebook 01 hoặc tự
động khi chạy `download_data.py` / `scripts/train.py`. Mất vài giây, kết quả xác định (chạy lại cho cùng file).

## 3. Vector từ có sẵn: fastText `cc.vi.300` (tùy chọn, cho thử nghiệm b)

| | |
|---|---|
| Nguồn | [fastText – Word vectors for 157 languages](https://fasttext.cc/docs/en/crawl-vectors.html), Grave et al. (2018), file `cc.vi.300.vec.gz` |
| Giấy phép | [Creative Commons Attribution-Share-Alike 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| Đặc điểm | 2 triệu "từ" × 300 chiều, học trên Common Crawl + Wikipedia; tách theo khoảng trắng nên là **mức âm tiết** |
| File sinh ra | `fasttext_vi_50k.npz` (~31 MB): 50.000 từ phổ biến nhất, vector float16 |

File gốc nặng 1,2 GB, nhưng các dòng được xếp theo tần suất giảm dần, nên `download_fasttext_subset` chỉ **đọc dần
(stream)** phần đầu file qua mạng, dừng khi đủ 50.000 từ (~35 MB tải về), rồi đóng kết nối. Khi đọc, mỗi từ được
chuẩn hóa NFC (có hàng nghìn từ không ở dạng NFC); các mục trùng sau chuẩn hóa, số và chuỗi lẫn ký hiệu bị bỏ qua.

Nếu bạn dùng lại file `.npz` này ở nơi khác, hãy ghi nguồn fastText và giữ giấy phép CC BY-SA 3.0.

### Phương án dự phòng khi không tải được fastText

Nếu mạng bị chặn hoặc máy chủ không phản hồi, `load_word_vectors()` tự chuyển sang **word2vec skip-gram tự huấn
luyện** (`train_word2vec` trong `src/vietformer/word_vectors.py`, viết bằng PyTorch, không cần gensim) trên chính
Truyện Kiều, mất vài giây. Mọi notebook vẫn chạy, nhưng hãy lưu ý: word2vec học từ 23 nghìn âm tiết chỉ minh họa cơ
chế; nó không mang tri thức "bên ngoài" như fastText, và không có vector cho các từ hiện đại (`Sài Gòn`, `chó`...) dùng
trong hình PCA/t-SNE của notebook 02.

## Mẹo khi tải bị treo

Một số mạng (kể cả mạng gia đình) cấu hình IPv6 nhưng IPv6 không thông: trình duyệt và `curl` tự chuyển sang IPv4 nên
không ai nhận ra, còn `urllib` của Python thì treo. Code tải của repo đã ưu tiên IPv4 và đặt thời gian chờ (timeout),
nên nếu vẫn lỗi thì nhiều khả năng mạng chặn hẳn máy chủ – khi đó dùng phương án dự phòng ở trên.

## Dùng corpus của riêng bạn

Bất kỳ file văn bản UTF-8 nào cũng dùng được, càng lớn càng tốt (văn xuôi cũng được, mỗi đoạn một dòng):

```bash
python scripts/train.py --corpus data/van_ban_cua_ban.txt --vocab-size 8000
```

Tokenizer riêng sẽ được huấn luyện và lưu thành `data/van_ban_cua_ban_bpe8000.model`. Hãy chắc chắn rằng bạn có quyền
dùng văn bản đó; các nguồn hợp pháp gợi ý: tác phẩm thuộc phạm vi công cộng trên Wikisource, Wikipedia tiếng Việt (CC
BY-SA 4.0), các bộ dữ liệu ghi rõ giấy phép trên Hugging Face.
