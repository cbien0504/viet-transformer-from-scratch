"""Vector từ tiếng Việt có sẵn – nguyên liệu cho thử nghiệm (b) của notebook 02.

Nguồn chính: fastText `cc.vi.300` (Facebook AI Research, giấy phép CC BY-SA 3.0), học trên
Common Crawl + Wikipedia tiếng Việt. File gốc nặng 1,2 GB (2 triệu từ × 300 chiều), nhưng các từ
được xếp theo tần suất giảm dần, nên chỉ cần đọc dần (stream) N dòng đầu, khoảng vài chục MB, là có
đủ các âm tiết thông dụng. Hai điều cần biết:

* fastText tiếng Việt ở mức ÂM TIẾT ("hoàng", "đế"), không có từ ghép "hoàng_đế". Vector của từ
  ghép được tính bằng trung bình vector các âm tiết (`WordVectors.phrase_vector`).
* Một phần từ vựng của fastText KHÔNG ở dạng NFC. Ta chuẩn hóa NFC khi đọc, và nếu sau chuẩn hóa
  bị trùng thì giữ bản xuất hiện trước (bản phổ biến hơn).

Phương án dự phòng khi không tải được: `train_word2vec` – word2vec skip-gram + negative sampling
viết gọn bằng PyTorch, học trên chính corpus trong vài giây.
"""

from __future__ import annotations

import gzip
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import load_corpus, open_url
from .tokenizer import SyllableTokenizer, normalize_text
from .utils import DATA_DIR

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz"


class WordVectors:
    """Bảng tra cứu từ → vector (numpy), kèm vài phép toán hay dùng."""

    def __init__(self, words: list[str], vectors: np.ndarray, source: str = ""):
        self.words = list(words)
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.source = source
        self.index = {w: i for i, w in enumerate(self.words)}
        self._unit = None  # các vector đã chuẩn hóa độ dài 1, tính một lần khi cần

    def __len__(self) -> int:
        return len(self.words)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def get(self, word: str) -> np.ndarray | None:
        """Vector của một từ: thử đúng chữ, sau đó thử chữ thường; không có thì trả về None."""
        word = normalize_text(word)
        for candidate in (word, word.lower()):
            if candidate in self.index:
                return self.vectors[self.index[candidate]]
        return None

    def phrase_vector(self, phrase: str) -> np.ndarray | None:
        """Vector của từ ghép ("hoàng_đế" hoặc "hoàng đế") = trung bình vector các âm tiết."""
        parts = normalize_text(phrase).replace("_", " ").split()
        vecs = [self.get(p) for p in parts]
        if not vecs or any(v is None for v in vecs):
            return None
        return np.mean(vecs, axis=0)

    def most_similar(self, query: str | np.ndarray, topn: int = 8) -> list[tuple[str, float]]:
        """Các từ có vector gần `query` nhất theo độ tương đồng cosine.

        `query` là một từ / từ ghép (khi đó bỏ qua chính các âm tiết của nó), hoặc trực tiếp một vector.
        """
        if isinstance(query, str):
            vec = self.phrase_vector(query)
            if vec is None:
                raise KeyError(f"Không có vector cho '{query}'")
            skip = {p.lower() for p in normalize_text(query).replace("_", " ").split()}
        else:
            vec, skip = np.asarray(query, dtype=np.float32), set()
        if self._unit is None:
            self._unit = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
        sims = self._unit @ (vec / (np.linalg.norm(vec) + 1e-8))
        result = []
        for i in np.argsort(-sims):
            if self.words[i].lower() not in skip:
                result.append((self.words[i], float(sims[i])))
            if len(result) == topn:
                break
        return result

    def save(self, path: str | Path) -> None:
        # Danh sách từ được nối bằng "\n" thành một chuỗi cho gọn; vector lưu float16 để nhẹ một nửa.
        np.savez(path, words=np.array("\n".join(self.words)), vectors=self.vectors.astype(np.float16), source=self.source)

    @classmethod
    def load(cls, path: str | Path) -> "WordVectors":
        data = np.load(path, allow_pickle=False)
        return cls(str(data["words"]).split("\n"), data["vectors"].astype(np.float32), str(data["source"]))


# ---------------------------------------------------------------------------
# Nguồn 1: fastText (đọc dần N dòng đầu)
# ---------------------------------------------------------------------------

def fasttext_cache_path(max_words: int = 50_000) -> Path:
    return DATA_DIR / f"fasttext_vi_{max_words // 1000}k.npz"


def _is_clean_word(word: str) -> bool:
    """Chỉ giữ từ toàn chữ cái (âm tiết, tên riêng) hoặc toàn dấu câu; bỏ số, ký tự lỗi, từ lẫn ký hiệu."""
    return word.isalpha() or all(unicodedata.category(c).startswith("P") for c in word)


def download_fasttext_subset(max_words: int = 50_000, force: bool = False, url: str = FASTTEXT_URL) -> WordVectors:
    """Đọc dần các dòng đầu của file fastText .vec.gz qua mạng cho tới khi có `max_words` từ, lưu bản rút gọn."""
    path = fasttext_cache_path(max_words)
    if path.exists() and not force:
        return WordVectors.load(path)
    print(f"Đang đọc {max_words} từ đầu tiên của {url} (không tải cả file 1,2 GB) ...")
    words, rows, seen = [], [], set()
    n_lines = n_not_nfc = n_skipped = 0
    with open_url(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as lines:
        lines.readline()  # dòng đầu là "2000000 300": số từ và số chiều
        for raw in lines:
            n_lines += 1
            word_bytes, rest = raw.split(b" ", 1)
            original = word_bytes.decode("utf-8", errors="replace")
            word = normalize_text(original)
            n_not_nfc += word != original
            if word in seen or not _is_clean_word(word):  # trùng sau NFC (giữ bản gặp trước, phổ biến hơn) hoặc rác
                n_skipped += 1
                continue
            seen.add(word)
            words.append(word)
            rows.append(np.array(rest.split(), dtype=np.float32))
            if len(words) % 10_000 == 0:
                print(f"  ... {len(words)}/{max_words} từ")
            if len(words) >= max_words:
                break
    print(f"Đã đọc {n_lines} dòng: {n_not_nfc} từ không ở dạng NFC; bỏ {n_skipped} từ trùng sau NFC hoặc không phải chữ/dấu câu.")
    vectors = WordVectors(words, np.stack(rows), source="fastText cc.vi.300")
    path.parent.mkdir(parents=True, exist_ok=True)
    vectors.save(path)
    print(f"Đã lưu {path} ({path.stat().st_size / 1e6:.0f} MB)")
    return vectors


# ---------------------------------------------------------------------------
# Nguồn 2 (dự phòng): word2vec skip-gram tự huấn luyện
# ---------------------------------------------------------------------------

def train_word2vec(
    text: str,
    dim: int = 100,
    window: int = 3,
    negatives: int = 5,
    epochs: int = 10,
    batch_size: int = 1024,
    lr: float = 0.005,
    seed: int = 0,
) -> WordVectors:
    """Word2vec skip-gram + negative sampling trên các âm tiết (viết thường) của `text`.

    Với mỗi cặp (âm tiết tâm, âm tiết ngữ cảnh) cách nhau tối đa `window` vị trí, mô hình học để
    tích vô hướng của hai vector lớn; với `negatives` âm tiết lấy ngẫu nhiên thì học để nó nhỏ.
    Kết quả: âm tiết xuất hiện trong ngữ cảnh giống nhau sẽ có vector gần nhau.
    """
    generator = torch.Generator().manual_seed(seed)
    tokens = [t.lower() for t in SyllableTokenizer.split(text) if t.isalpha()]
    vocab = sorted(set(tokens))
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = torch.tensor([stoi[t] for t in tokens])

    # Tất cả các cặp (tâm, ngữ cảnh) trong cửa sổ ±window.
    centers, contexts = [], []
    for offset in range(1, window + 1):
        centers += [ids[:-offset], ids[offset:]]
        contexts += [ids[offset:], ids[:-offset]]
    centers, contexts = torch.cat(centers), torch.cat(contexts)

    # Phân phối lấy mẫu âm: tần suất^0.75 (như bài báo word2vec gốc).
    noise = torch.bincount(ids, minlength=len(vocab)).float() ** 0.75

    emb_in = nn.Embedding(len(vocab), dim)    # vector "tâm" – đây là vector ta giữ lại
    emb_out = nn.Embedding(len(vocab), dim)   # vector "ngữ cảnh"
    with torch.no_grad():                     # khởi tạo như word2vec gốc: nhỏ ngẫu nhiên / bằng 0
        emb_in.weight.copy_((torch.rand(len(vocab), dim, generator=generator) - 0.5) / dim)
        emb_out.weight.zero_()
    optimizer = torch.optim.Adam([*emb_in.parameters(), *emb_out.parameters()], lr=lr)

    for _ in range(epochs):
        perm = torch.randperm(len(centers), generator=generator)
        for start in range(0, len(perm), batch_size):
            b = perm[start : start + batch_size]
            neg = torch.multinomial(noise, len(b) * negatives, replacement=True, generator=generator)
            v_c = emb_in(centers[b])                            # (B, D)
            u_o = emb_out(contexts[b])                          # (B, D)
            u_n = emb_out(neg.view(len(b), negatives))          # (B, K, D)
            pos_score = (v_c * u_o).sum(-1)                     # (B,)
            neg_score = (u_n @ v_c.unsqueeze(-1)).squeeze(-1)   # (B, K)
            loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return WordVectors(vocab, emb_in.weight.detach().numpy(), source="word2vec tự huấn luyện (dự phòng)")


def load_word_vectors(prefer: str = "fasttext", max_words: int = 50_000, text: str | None = None) -> WordVectors:
    """Ưu tiên fastText (bản đã lưu, hoặc tải về); nếu lỗi thì tự huấn luyện word2vec trên corpus."""
    if prefer == "fasttext":
        try:
            return download_fasttext_subset(max_words)
        except Exception as error:  # mất mạng, bị chặn, hết thời gian chờ...
            print(f"Không tải được fastText ({error!r}).\nChuyển sang phương án dự phòng: tự huấn luyện word2vec.")
    return train_word2vec(text if text is not None else load_corpus())


def build_embedding_matrix(tokenizer, word_vectors: WordVectors, seed: int = 0) -> tuple[torch.Tensor, float]:
    """Ma trận vocab_size × dim để khởi tạo `TokenEmbedding` trong thử nghiệm (b).

    Token "▁người" lấy vector của "người". Token không tìm thấy (mảnh âm tiết, <unk>, <n>...) được
    khởi tạo ngẫu nhiên với cùng độ lớn như các vector có sẵn. Trả về (ma trận, tỉ lệ token tìm thấy).
    """
    rng = np.random.default_rng(seed)
    std = float(word_vectors.vectors.std())
    matrix = rng.normal(0.0, std, size=(tokenizer.vocab_size, word_vectors.dim)).astype(np.float32)
    found = 0
    for i in range(tokenizer.vocab_size):
        word = tokenizer.id_to_piece(i).replace("▁", "")
        vec = word_vectors.get(word) if word else None
        if vec is not None:
            matrix[i] = vec
            found += 1
    return torch.from_numpy(matrix), found / tokenizer.vocab_size
