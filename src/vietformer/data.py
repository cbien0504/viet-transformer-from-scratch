"""Dữ liệu: tải corpus, đọc corpus (luôn chuẩn hóa NFC), chuẩn bị tokenizer và dãy token id.

Corpus mặc định là Truyện Kiều của Nguyễn Du (1766–1820), tác phẩm thuộc phạm vi công cộng, lấy
từ Wikisource tiếng Việt qua MediaWiki API: 3.254 câu lục bát, khoảng 23 nghìn âm tiết. Đủ nhỏ để
huấn luyện vài phút trên CPU, đủ đặc trưng (nhịp 6–8, vần) để thấy mô hình có học được gì không.
"""

from __future__ import annotations

import json
import re
import socket
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import torch

from .tokenizer import VietTokenizer, normalize_text
from .utils import DATA_DIR

KIEU_PATH = DATA_DIR / "truyen_kieu.txt"
KIEU_PAGE_TITLE = "Truyện Kiều"
WIKISOURCE_API = "https://vi.wikisource.org/w/api.php"
USER_AGENT = "viet-transformer-from-scratch/0.1 (educational project)"
DEFAULT_VOCAB_SIZE = 4000


# ---------------------------------------------------------------------------
# Tải dữ liệu qua mạng
# ---------------------------------------------------------------------------

@contextmanager
def _prefer_ipv4():
    """Tạm thời ưu tiên IPv4 khi mở kết nối.

    Trên một số mạng, IPv6 được cấu hình nhưng không thông: urllib thử IPv6 trước và bị treo,
    trong khi trình duyệt hay curl tự chuyển sang IPv4 nên ta không nhận ra. Ở đây ta xếp các
    địa chỉ IPv4 lên đầu danh sách mà DNS trả về.
    """
    original = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(original(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)

    socket.getaddrinfo = ipv4_first
    try:
        yield
    finally:
        socket.getaddrinfo = original


def open_url(url: str, timeout: float = 30.0):
    """Mở một URL (có timeout, ưu tiên IPv4) và trả về response để đọc dần từng phần."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _prefer_ipv4():
        return urllib.request.urlopen(request, timeout=timeout)


def fetch_wikisource_wikitext(title: str) -> str:
    """Lấy mã nguồn (wikitext) của một trang trên Wikisource tiếng Việt."""
    params = {"action": "parse", "page": title, "prop": "wikitext", "format": "json", "formatversion": "2"}
    with open_url(WIKISOURCE_API + "?" + urllib.parse.urlencode(params)) as response:
        data = json.load(response)
    if "error" in data:
        raise RuntimeError(f"Wikisource báo lỗi: {data['error']}")
    return data["parse"]["wikitext"]


def clean_poem_wikitext(wikitext: str) -> list[str]:
    """Lấy các câu thơ trong thẻ <poem>…</poem>, bỏ mã wiki; trả về danh sách câu đã chuẩn hóa NFC."""
    text = "\n".join(re.findall(r"<poem>(.*?)</poem>", wikitext, flags=re.S))
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.S)    # chú thích cuối trang
    text = re.sub(r"\{\{[^{}]*\}\}", "", text)                     # template, ví dụ {{số|5}}
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)   # [[liên kết|chữ]] -> chữ
    text = text.replace("'''", "").replace("''", "")               # in đậm, in nghiêng
    lines = (" ".join(normalize_text(line).split()) for line in text.split("\n"))
    return [line for line in lines if line]


def download_truyen_kieu(path: str | Path = KIEU_PATH, force: bool = False) -> Path:
    """Tải Truyện Kiều từ Wikisource và lưu thành file văn bản, mỗi câu thơ một dòng."""
    path = Path(path)
    if path.exists() and not force:
        return path
    print(f"Đang tải Truyện Kiều từ Wikisource ({WIKISOURCE_API}) ...")
    lines = clean_poem_wikitext(fetch_wikisource_wikitext(KIEU_PAGE_TITLE))
    if len(lines) < 3000:  # bản đầy đủ có 3.254 câu
        raise RuntimeError(f"Chỉ lấy được {len(lines)} câu – có thể trang Wikisource đã đổi định dạng.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã lưu {len(lines)} câu thơ vào {path}")
    return path


# ---------------------------------------------------------------------------
# Đọc corpus, tokenizer, dãy token id
# ---------------------------------------------------------------------------

def load_corpus(path: str | Path = KIEU_PATH) -> str:
    """Đọc corpus thành một chuỗi đã chuẩn hóa NFC (tự tải Truyện Kiều nếu chưa có file)."""
    path = Path(path)
    if not path.exists():
        if path.resolve() == KIEU_PATH.resolve():
            download_truyen_kieu(path)
        else:
            raise FileNotFoundError(f"Không tìm thấy corpus: {path}")
    return normalize_text(path.read_text(encoding="utf-8")).strip()


def tokenizer_prefix(corpus_path: str | Path = KIEU_PATH, vocab_size: int = DEFAULT_VOCAB_SIZE) -> Path:
    """Tiền tố tên file tokenizer, ví dụ data/truyen_kieu_bpe4000 (→ .model và .vocab)."""
    return DATA_DIR / f"{Path(corpus_path).stem}_bpe{vocab_size}"


def get_tokenizer(
    corpus_path: str | Path = KIEU_PATH, vocab_size: int = DEFAULT_VOCAB_SIZE, retrain: bool = False
) -> VietTokenizer:
    """Nạp tokenizer BPE đã huấn luyện; nếu chưa có thì huấn luyện (mất vài giây) rồi lưu lại."""
    prefix = tokenizer_prefix(corpus_path, vocab_size)
    model_file = Path(f"{prefix}.model")
    if model_file.exists() and not retrain:
        return VietTokenizer(model_file)
    print(f"Huấn luyện tokenizer BPE (vocab_size={vocab_size}) trên {Path(corpus_path).name} ...")
    return VietTokenizer.train(load_corpus(corpus_path), prefix, vocab_size=vocab_size)


def encode_corpus(tokenizer: VietTokenizer, corpus_path: str | Path = KIEU_PATH) -> torch.Tensor:
    """Mã hóa toàn bộ corpus thành một tensor 1 chiều các token id (kiểu long)."""
    return torch.tensor(tokenizer.encode(load_corpus(corpus_path)), dtype=torch.long)
