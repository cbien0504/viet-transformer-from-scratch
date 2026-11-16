"""Tokenization: tách văn bản tiếng Việt thành các token – đơn vị nhỏ nhất mà mô hình "nhìn thấy".

Ba cách tách được so sánh trong notebook 01:

* `CharTokenizer`     – mỗi ký tự là một token: vocab rất nhỏ nhưng chuỗi rất dài.
* `SyllableTokenizer` – tách theo khoảng trắng: mỗi âm tiết là một token; âm tiết lạ thành <unk>.
* `VietTokenizer`     – subword BPE của SentencePiece, cách các mô hình thật dùng: âm tiết thường
                        gặp là một token, âm tiết hiếm được ghép từ các mảnh nhỏ hơn.

Bước bắt buộc trước mọi cách tách: chuẩn hóa Unicode NFC (`normalize_text`). Cùng một chữ "ệ" có
thể được lưu bằng 1 mã Unicode (NFC) hoặc 3 mã "e + dấu mũ + dấu nặng" (NFD); với máy tính đó là
hai chuỗi khác nhau, nên nếu không chuẩn hóa thì một từ sẽ có nhiều id khác nhau.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import sentencepiece as spm

UNK = "<unk>"      # token cho thứ tokenizer chưa từng gặp
NEWLINE = "<n>"    # token đặc biệt thay cho ký tự xuống dòng "\n"
UNK_DISPLAY = "⁇"  # cách hiển thị <unk> khi giải mã


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode về dạng NFC (dựng sẵn) và thống nhất ký tự xuống dòng thành "\\n"."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


class CharTokenizer:
    """Mỗi ký tự (sau khi chuẩn hóa NFC) là một token. Id 0 dành cho ký tự chưa gặp (<unk>)."""

    def __init__(self, text: str):
        chars = sorted(set(normalize_text(text)))
        self.itos = [UNK] + chars                                # id -> token
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}   # token -> id

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, 0) for ch in normalize_text(text)]

    def decode(self, ids) -> str:
        return "".join(self.itos[i] if i != 0 else UNK_DISPLAY for i in map(int, ids))


class SyllableTokenizer:
    """Tách theo khoảng trắng: mỗi âm tiết là một token; dấu câu và xuống dòng là token riêng.

    Tiếng Việt viết cách các âm tiết, nên đây là cách tách "tự nhiên" nhất. Nhược điểm: từ ghép
    như "hoàng đế" bị tách làm hai, và âm tiết chưa gặp khi xây vocab đều thành <unk>.
    """

    # \w+  : một chuỗi chữ cái/chữ số liền nhau (một âm tiết)
    # [^\w\s]: một dấu câu;  \n: xuống dòng
    PATTERN = re.compile(r"\w+|[^\w\s]|\n")

    def __init__(self, text: str, min_count: int = 1):
        counts = Counter(self.split(text))
        vocab = sorted(tok for tok, c in counts.items() if c >= min_count)
        self.itos = [UNK] + vocab
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def split(cls, text: str) -> list[str]:
        return cls.PATTERN.findall(normalize_text(text))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, 0) for tok in self.split(text)]

    def decode(self, ids) -> str:
        text = " ".join(self.itos[i] if i != 0 else UNK_DISPLAY for i in map(int, ids))
        text = re.sub(r" ?\n ?", "\n", text)            # bỏ khoảng trắng quanh xuống dòng
        return re.sub(r" ([,.!?;:])", r"\1", text)       # dấu câu dính vào chữ đứng trước


class VietTokenizer:
    """Tokenizer subword BPE dựa trên SentencePiece, có chuẩn hóa NFC và giữ được xuống dòng.

    * Ký tự "▁" trong token đánh dấu một khoảng trắng đứng trước, tức là đầu một âm tiết:
      "▁người" là âm tiết "người" đứng sau một khoảng trắng.
    * SentencePiece coi mỗi dòng là một câu riêng và không biết ký tự "\\n". Thơ lục bát lại cần
      xuống dòng, nên ta mã hóa từng dòng rồi chèn token đặc biệt <n> vào giữa các dòng.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)
        self.newline_id = self.sp.piece_to_id(NEWLINE)
        self.unk_id = self.sp.unk_id()

    @classmethod
    def train(
        cls,
        text: str,
        model_prefix: str | Path,
        vocab_size: int = 4000,
        model_type: str = "bpe",
        split_by_whitespace: bool = True,
    ) -> "VietTokenizer":
        """Huấn luyện SentencePiece trên `text`, lưu ra `<model_prefix>.model` và `.vocab`."""
        lines = [line for line in normalize_text(text).split("\n") if line.strip()]
        Path(model_prefix).parent.mkdir(parents=True, exist_ok=True)
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(lines),
            model_prefix=str(model_prefix),
            vocab_size=vocab_size,
            model_type=model_type,                    # "bpe" (ghép cặp) hoặc "unigram"
            character_coverage=1.0,                   # giữ MỌI ký tự có dấu của tiếng Việt
            normalization_rule_name="identity",       # ta đã tự chuẩn hóa NFC, không đổi gì thêm
            split_by_whitespace=split_by_whitespace,  # True: token không vượt qua khoảng trắng
            user_defined_symbols=[NEWLINE],           # <n> luôn là một token nguyên vẹn
            unk_id=0,
            bos_id=-1,                                # không dùng token "bắt đầu câu"
            eos_id=-1,                                # không dùng token "kết thúc câu"
            minloglevel=2,                            # bớt in log
        )
        return cls(f"{model_prefix}.model")

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        """Văn bản -> danh sách token id (mỗi dòng mã hóa riêng, nối bằng <n>)."""
        ids: list[int] = []
        for i, line in enumerate(normalize_text(text).split("\n")):
            if i > 0:
                ids.append(self.newline_id)
            ids.extend(self.sp.encode(line))
        return ids

    def decode(self, ids) -> str:
        """Danh sách token id -> văn bản (<n> trở lại thành "\\n")."""
        lines: list[str] = []
        current: list[int] = []
        for i in map(int, ids):
            if i == self.newline_id:
                lines.append(self.sp.decode(current))
                current = []
            else:
                current.append(i)
        lines.append(self.sp.decode(current))
        return "\n".join(lines)

    def id_to_piece(self, i: int) -> str:
        return self.sp.id_to_piece(int(i))

    def piece_to_id(self, piece: str) -> int:
        return self.sp.piece_to_id(piece)

    def encode_as_pieces(self, text: str) -> list[str]:
        """Văn bản -> danh sách token dạng chuỗi, ví dụ ['▁Trăm', '▁năm', ',']."""
        return [self.id_to_piece(i) for i in self.encode(text)]

    def display_tokens(self, ids) -> list[str]:
        """Token dễ đọc để làm nhãn khi vẽ hình: bỏ '▁', <n> thành '↵'."""
        return ["↵" if i == self.newline_id else self.id_to_piece(i).replace("▁", "") for i in map(int, ids)]
