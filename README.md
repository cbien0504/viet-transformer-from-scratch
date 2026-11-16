# viet-transformer-from-scratch

Tự tay xây một **mini-GPT** (Transformer decoder-only) bằng PyTorch, từng thành phần một, rồi huấn luyện nó trên
**Truyện Kiều** để nó "làm thơ lục bát". Repo dành cho người biết Python cơ bản, biết sơ về mạng nơ-ron và **chưa
biết Transformer**.

Mục tiêu số 1 là **dễ hiểu**, không phải hiệu năng: code ngắn, tường minh, chú thích tiếng Việt; attention được viết
bằng các phép toán tensor cơ bản (không dùng `nn.Transformer`, `nn.MultiheadAttention` hay thư viện bọc sẵn). Mọi
bước chạy được trên laptop chỉ có CPU, mỗi bước vài phút.

```text
Trăm năm trong cõi người ta,            ← câu mở đầu của Nguyễn Du, đưa vào làm gợi ý
Nghĩ có hai lời nàng mới là gì?          ← từ đây trở xuống do mô hình 1,8 triệu tham số
Nỗi lòng chẳng giữ giàng cho đến nhà.       viết ra, sau 2 phút học trên CPU (notebook 07)
Nàng rằng: Những là duyên,
Nghĩ mình một ngày cũng như rày cho!
```

Có dáng lục bát, đúng chính tả, nhưng chưa có nghĩa – một mô hình tí hon học trên 27 nghìn token chỉ làm được đến đó,
và hiểu vì sao chính là mục đích của repo.

## Luồng dữ liệu

```text
 "Trăm năm trong cõi người ta,"                          văn bản, chuẩn hóa Unicode NFC
          │  tokenizer BPE (SentencePiece, 4.000 token)              tokenizer.py      notebook 01
          ▼
 [712, 242, 168, 917, 72, 200, 3880]                       token id                (B, T)
          │  bảng embedding 4.000 × 128                               embedding.py      notebook 02
          │  + vector vị trí (learned hoặc sin/cos)                   positional.py     notebook 03
          ▼
 x                                                          vector                  (B, T, 128)
          │  ┌────────────────────── N = 4 khối Transformer ──────────────────────┐
          │  │  x = x + MultiHeadAttention(LayerNorm(x))   ← causal mask           │ attention.py  notebook 04, 05
          │  │  x = x + FeedForward(LayerNorm(x))          ← GELU                  │ block.py      notebook 06
          │  └────────────────────────────────────────────────────────────────────┘
          ▼
 LayerNorm → Linear (lm_head)                               logits                  (B, T, 4000)   model.py
          │  softmax(logits / temperature), top-k → lấy mẫu 1 token          generate.py       notebook 08
          ▼
 "<n>" (xuống dòng) → nối vào chuỗi, lặp lại → "Chữ", "tài"…   token tiếp theo
```

`B` = số chuỗi trong một batch, `T` = số token mỗi chuỗi (tối đa `block_size = 64`). Huấn luyện (`train.py`, notebook 07)
nghĩa là chỉnh mọi trọng số để tại **mỗi** vị trí, logits cho xác suất cao nhất vào token thật sự đứng sau.

## Cài đặt

Cần Python 3.10+.

```bash
git clone <url-của-repo> viet-transformer-from-scratch
cd viet-transformer-from-scratch
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Máy chỉ có CPU: cài PyTorch bản CPU trước cho nhẹ (~200 MB thay vì vài GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/download_data.py              # tải Truyện Kiều + huấn luyện tokenizer (vài giây)
python scripts/download_data.py --fasttext   # (tùy chọn) vector fastText cho thử nghiệm embedding, ~35 MB
```

Mã nguồn nằm trong `src/vietformer/`. Các notebook và script tự thêm `src/` vào đường dẫn import nên **không cần cài
gói**; nếu muốn `import vietformer` ở bất cứ đâu thì chạy thêm `pip install -e .`.

## Lộ trình học

Mở Jupyter (`jupyter lab` hoặc `jupyter notebook`, hoặc mở trực tiếp trong VS Code) và đi theo thứ tự. Mỗi notebook
giải thích một khái niệm, import code từ module tương ứng trong `src/vietformer/`, và kết thúc bằng 2–3 bài tập có đáp
án trong `solutions/`. Notebook trong repo đã được chạy sẵn, nên bạn có thể đọc kết quả trước khi tự chạy.

| # | Notebook | Bạn sẽ hiểu | Module | Thời gian chạy* |
|---|---|---|---|---|
| 01 | `01_tokenization` | NFC/NFD, dấu thanh, từ nhiều âm tiết; tách theo ký tự / âm tiết / BPE | `tokenizer.py` | ~10 giây |
| 02 | `02_embedding` | embedding = bảng tra cứu; fastText; PCA, t-SNE; thử nghiệm (a) học từ đầu vs (b) fastText freeze/finetune | `embedding.py`, `word_vectors.py` | ~2–3 phút |
| 03 | `03_positional_encoding` | "Chó cắn người" vs "Người cắn chó"; sinusoidal và learned | `positional.py` | vài giây |
| 04 | `04_self_attention` | Q, K, V; **tính tay** 3 token × 4 chiều; chia √d; causal mask | `attention.py` | vài giây |
| 05 | `05_multi_head_attention` | nhiều head bằng `view`/`transpose`, shape ở từng bước | `attention.py` | vài giây |
| 06 | `06_transformer_block` | LayerNorm, residual, feed-forward (GELU), dropout, pre-norm | `block.py` | ~10 giây |
| 07 | `07_train_mini_gpt` | ghép thành MiniGPT, train/val, loss, overfitting, checkpoint; lặp lại thử nghiệm (a)/(b) | `model.py`, `train.py` | ~6 phút (~2 phút nếu bỏ mục so sánh) |
| 08 | `08_generate_and_visualize` | temperature, top-k; attention từng head/layer; head nào làm gì, có head nào bắt vần lục bát? | `generate.py`, `viz.py` | ~20 giây |

\* Đo trên một máy Linux chỉ dùng CPU (PyTorch dùng 6 luồng), không GPU. Có GPU (CUDA) hoặc Apple Silicon (MPS) thì code tự dùng.
Notebook 08 cần checkpoint do notebook 07 (hoặc `scripts/train.py`) tạo ra.

Nên đọc kèm mã nguồn theo đúng thứ tự dữ liệu đi qua mô hình:
`tokenizer → embedding → positional → attention → block → model → train → generate`. Mỗi file chỉ vài chục dòng
có ý nghĩa, mở đầu bằng docstring tiếng Việt giải thích "cái này làm gì và vì sao". Các module phụ: `utils.py` (seed,
chọn thiết bị), `data.py` (tải corpus), `word_vectors.py` (fastText, word2vec dự phòng), `viz.py` (vẽ hình).

## Chạy bằng dòng lệnh

```bash
python scripts/train.py                                 # cấu hình mặc định, ~2 phút trên CPU
python scripts/train.py --preset tiny --max-steps 300   # chạy thử thật nhanh (~10 giây)
python scripts/train.py --embedding fasttext --freeze   # thử nghiệm (b): fastText, đóng băng
python scripts/train.py --embedding fasttext            # thử nghiệm (b): fastText, finetune
python scripts/train.py --pos sinusoidal                # positional encoding sin/cos
python scripts/train.py --help                          # mọi tùy chọn

pytest                                                  # ~5 giây
```

`scripts/train.py` lưu checkpoint tốt nhất (theo val loss) vào `checkpoints/minigpt_truyen_kieu.pt`, hình đường loss
vào `checkpoints/minigpt_truyen_kieu.png`, và in vài đoạn thơ sinh thử. Notebook 08 dùng chính checkpoint này.

Cấu hình mô hình nằm trong **một** dataclass, `GPTConfig` (`src/vietformer/model.py`):

```python
GPTConfig(vocab_size=4000, block_size=64, n_layer=4, n_head=4, d_model=128, dropout=0.1,
          pos_encoding="learned", pretrained_dim=None, freeze_embedding=False)   # ~1,8 triệu tham số
```

Seed được cố định (`set_seed`), thiết bị được chọn tự động theo thứ tự cuda → mps → cpu (`get_device`).

## Kết quả tham khảo

Đo bằng chính các notebook, trên CPU, seed cố định (máy khác có thể lệch chút ít). Loss là cross-entropy trên tập val
(10% cuối Truyện Kiều); lúc khởi tạo loss ≈ `ln 4000 ≈ 8,3`.

**MiniGPT mặc định** (1,8 triệu tham số, 1.000 bước, ~2 phút): val loss tốt nhất **5,37**, tức perplexity ≈ 215 (so với
4.000 lúc chưa học). Train loss tiếp tục giảm tới ~3,5 trong khi val loss đứng yên: overfitting rõ từ khoảng bước 500.
Văn bản sinh ra (temperature 0,8, top-k 50): **100%** âm tiết có trong Truyện Kiều (mô hình chưa học: 84%), và dưới
**1%** cụm 4 âm tiết chép nguyên văn.

**Thử nghiệm embedding (a) học từ đầu vs (b) khởi tạo từ fastText** – val loss tốt nhất, thấp hơn là tốt hơn:

| Embedding | Mô hình bigram (notebook 02) | MiniGPT (notebook 07) |
|---|---|---|
| (a) học từ đầu | **5,49** | 5,37 |
| (b) fastText, freeze | 5,70 | **5,11** |
| (b) fastText, finetune | 5,81 | 5,29 |

Kết luận **đổi chiều** giữa hai mô hình. Với bigram, embedding phải tự mang thông tin "chữ nào hay đứng sau" nên học
thẳng từ Truyện Kiều tốt hơn; fastText chỉ thắng ở các token chưa từng gặp khi train. Với MiniGPT, attention lo phần
ngữ cảnh; embedding fastText đóng băng vừa mang nghĩa từ hàng tỉ chữ vừa bớt tham số được học nên ít overfit nhất.

**Bên trong mô hình đã học** (notebook 08):
- Các head ở layer 1–3 dồn attention vào các dấu xuống dòng (ranh giới câu thơ), nhiều nhất gấp ~2,2 lần mức chia đều.
- Vector vị trí tự học được một chu kỳ ~9 token, đúng bằng độ dài trung bình một câu thơ.
- Không có head nào "bắt vần" lục bát.

## Kiểm thử

`tests/` (pytest, ~5 giây, không cần mạng) kiểm tra: shape đầu ra của mọi lớp; **causal mask** (đổi token tương lai
không làm đổi đầu ra ở vị trí hiện tại, ở mức head, khối và cả mô hình); **tổng trọng số attention bằng 1**; attention
tự viết khớp `F.scaled_dot_product_attention` của PyTorch; multi-head bằng `view`/`transpose` khớp cách nối từng head;
`LayerNorm` tự viết khớp `nn.LayerNorm`; **mô hình overfit được một batch nhỏ**; loss ban đầu ≈ `ln(vocab_size)`;
NFC/NFD cho cùng token id; `freeze` thật sự giữ nguyên bảng embedding; lưu/nạp checkpoint.

## Bảng thuật ngữ Anh – Việt

| Tiếng Anh | Tiếng Việt | Ý nghĩa ngắn gọn |
|---|---|---|
| token | token, đơn vị từ vựng | mẩu văn bản nhỏ nhất mô hình nhìn thấy (ở đây gần như là một âm tiết) |
| tokenizer | bộ tách token | biến văn bản ↔ dãy token id |
| vocabulary (vocab) | bộ từ vựng | tập mọi token mô hình biết; mỗi token có một id |
| subword, BPE | đơn vị dưới từ, mã hóa cặp byte | ghép dần các cặp ký tự hay đi cùng nhau thành token |
| OOV (out of vocabulary) | ngoài từ vựng | thứ tokenizer chưa từng gặp, thường thành `<unk>` |
| Unicode normalization (NFC/NFD) | chuẩn hóa Unicode (dựng sẵn / tổ hợp) | đưa các cách mã hóa khác nhau của cùng một chữ về một dạng |
| embedding | vector nhúng | vector số thực đại diện cho một token |
| pretrained | (được) huấn luyện sẵn | học trước trên dữ liệu khác, đem về dùng lại |
| freeze / fine-tune | đóng băng / tinh chỉnh | giữ nguyên / tiếp tục cập nhật trọng số có sẵn |
| positional encoding | mã hóa vị trí | vector cho biết token đứng ở vị trí nào |
| attention, self-attention | cơ chế chú ý, tự chú ý | mỗi token lấy trung bình có trọng số thông tin từ các token khác |
| query / key / value | truy vấn / khóa / giá trị | ba vector mỗi token tạo ra để tính attention |
| head, multi-head | đầu (chú ý), nhiều đầu | một bộ Q, K, V riêng; nhiều head chạy song song |
| causal mask | mặt nạ nhân quả | cấm token nhìn các token phía sau nó |
| softmax | hàm softmax | biến điểm số thành xác suất dương có tổng bằng 1 |
| logits | logit, điểm số thô | đầu ra trước softmax, một số cho mỗi token trong vocab |
| LayerNorm | chuẩn hóa theo lớp | đưa vector mỗi token về trung bình 0, độ lệch chuẩn 1 |
| residual connection | kết nối tắt (phần dư) | `x + f(x)`: đường tắt cho thông tin và gradient |
| feed-forward (MLP) | mạng truyền thẳng | 2 lớp Linear + GELU áp dụng cho từng token |
| dropout | dropout (bỏ ngẫu nhiên) | tắt ngẫu nhiên một phần neuron khi huấn luyện để chống học thuộc |
| batch | lô | nhóm chuỗi được xử lý cùng lúc |
| block size, context | độ dài ngữ cảnh | số token tối đa mô hình nhìn được một lúc |
| loss, cross-entropy | hàm mất mát, entropy chéo | `−log` xác suất mô hình gán cho token đúng |
| perplexity | độ bối rối | `e^loss`: mô hình "phân vân" như khi chọn ngẫu nhiên giữa bấy nhiêu token |
| gradient, backpropagation | đạo hàm, lan truyền ngược | hướng chỉnh trọng số để loss giảm |
| optimizer (AdamW), learning rate | bộ tối ưu, tốc độ học | thuật toán và bước nhảy khi cập nhật trọng số |
| warmup, weight decay | khởi động, suy giảm trọng số | tăng dần learning rate lúc đầu; kéo nhẹ trọng số về 0 |
| train / validation split | tập huấn luyện / kiểm định | dữ liệu để học / dữ liệu giữ riêng để đánh giá |
| overfitting | quá khớp, học thuộc lòng | tốt trên train nhưng kém trên dữ liệu mới |
| checkpoint | điểm lưu | file lưu trọng số (và cấu hình) của mô hình |
| temperature, top-k, greedy | nhiệt độ, top-k, tham lam | các cách chọn token khi sinh văn bản |
| decoder-only | chỉ có bộ giải mã | kiến trúc kiểu GPT: chỉ dự đoán token tiếp theo |

## Hạn chế

* **Đây là công cụ học, không phải mô hình thực dụng.** Mô hình 1,8 triệu tham số học trên ~27 nghìn token thì sinh
  được văn bản *có dáng* lục bát (nhịp câu, xuống dòng, "Nàng rằng:", từ ngữ của Truyện Kiều) nhưng **chưa mạch lạc về
  nghĩa**, không có kiến thức gì về thế giới và không làm theo chỉ dẫn được.
* Corpus rất nhỏ nên **overfitting** đến sau vài trăm bước; mô hình đôi khi chép lại nguyên cụm từ Truyện Kiều
  (notebook 07, 08 có đo tỉ lệ này).
* Chính tả đúng ở mức âm tiết là nhờ vocab gồm chủ yếu âm tiết nguyên vẹn **và** nhờ mô hình đã học, chứ không phải
  được đảm bảo tuyệt đối: vocab còn khoảng 400 mảnh âm tiết, và mô hình chưa huấn luyện ghép chúng thành chữ không có
  thật (notebook 07 đo điều này). Mô hình cũng chỉ biết từ vựng của Truyện Kiều, không biết từ hiện đại.
* fastText ở mức âm tiết; vector từ ghép (`hoàng_đế`) chỉ được xấp xỉ bằng trung bình vector các âm tiết.
* Không thống nhất kiểu đặt dấu thanh (`hoà`/`hòa`); tokenizer được huấn luyện trên cả phần val (rò rỉ nhỏ, có ghi chú
  trong notebook 07).
* Con số loss có thể lệch chút ít giữa các máy và phiên bản PyTorch dù đã cố định seed.

## Bước tiếp theo

* **RoPE** (rotary positional embedding) – cách mã hóa vị trí của LLaMA, Qwen...: xoay vector Q, K thay vì cộng vector vị trí.
* **KV cache** – khi sinh văn bản, lưu lại K, V của các token cũ thay vì tính lại từ đầu ở mỗi bước.
* **Weight tying** (dùng chung ma trận embedding và `lm_head`), **Flash attention** (`F.scaled_dot_product_attention`),
  huấn luyện mixed precision.
* **Dữ liệu lớn hơn**: Wikipedia tiếng Việt (CC BY-SA 4.0) hoặc các bộ dữ liệu có giấy phép rõ ràng trên Hugging Face;
  dùng `--corpus` và `--vocab-size 8000`.
* **Fine-tune PhoBERT** ([VinAI](https://github.com/VinAIResearch/PhoBERT)) – Transformer *encoder* đã huấn luyện sẵn
  cho tiếng Việt – cho các bài toán phân loại văn bản, nhận dạng thực thể; hoặc thử các mô hình sinh tiếng Việt lớn hơn.
* **Đọc paper gốc và tài liệu kinh điển:**
  * Vaswani et al. (2017), *Attention Is All You Need* – Transformer gốc.
  * Radford et al. (2018, 2019), *Improving Language Understanding by Generative Pre-Training* (GPT) và
    *Language Models are Unsupervised Multitask Learners* (GPT-2) – kiến trúc decoder-only mà repo này bắt chước.
  * Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding* – RoPE.
  * Olsson et al. (2022), *In-context Learning and Induction Heads* – các head "chữ ngay trước" và "sao chép" ở notebook 08.
  * Jay Alammar, *The Illustrated Transformer*; Andrej Karpathy, *nanoGPT* và video *Let's build GPT*.

## Cấu trúc repo

```text
viet-transformer-from-scratch/
├── README.md                  # file này
├── requirements.txt
├── pyproject.toml             # cấu hình gói (tùy chọn pip install -e .) và pytest
├── data/README.md             # nguồn dữ liệu, giấy phép, cách tải
├── notebooks/                 # 01 → 08, học theo thứ tự
├── solutions/                 # đáp án bài tập của từng notebook
├── src/vietformer/
│   ├── tokenizer.py           # NFC, tách ký tự / âm tiết / BPE (SentencePiece)
│   ├── embedding.py           # TokenEmbedding (học từ đầu hoặc từ vector có sẵn, freeze/finetune)
│   ├── word_vectors.py        # fastText cc.vi.300 rút gọn + word2vec dự phòng
│   ├── positional.py          # sinusoidal, learned
│   ├── attention.py           # scaled dot-product, SelfAttentionHead, MultiHeadAttention
│   ├── block.py               # LayerNorm, FeedForward, TransformerBlock (pre-norm)
│   ├── model.py               # GPTConfig, MiniGPT, lưu/nạp checkpoint
│   ├── train.py               # chia train/val, batch, vòng lặp huấn luyện
│   ├── generate.py            # temperature, top-k, sinh văn bản
│   ├── data.py, utils.py, viz.py
├── tests/                     # pytest
└── scripts/
    ├── download_data.py
    └── train.py
```

## Nguồn dữ liệu và giấy phép

* *Truyện Kiều* – Nguyễn Du, thuộc phạm vi công cộng; văn bản từ [Wikisource tiếng Việt](https://vi.wikisource.org/wiki/Truy%E1%BB%87n_Ki%E1%BB%81u).
* Vector fastText `cc.vi.300` – Facebook AI Research, giấy phép CC BY-SA 3.0 (chỉ tải về khi chạy, không nằm trong repo).

Chi tiết trong [`data/README.md`](data/README.md).
