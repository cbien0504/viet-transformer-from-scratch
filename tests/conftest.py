"""Dữ liệu và tokenizer nhỏ dùng chung cho các test (không cần mạng, không cần file đã tải)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # phòng khi chạy pytest ngoài thư mục gốc

from vietformer.tokenizer import VietTokenizer  # noqa: E402

# 20 câu đầu Truyện Kiều (Nguyễn Du, phạm vi công cộng).
KIEU_20 = """Trăm năm trong cõi người ta,
Chữ tài chữ mệnh khéo là ghét nhau.
Trải qua một cuộc bể dâu,
Những điều trông thấy mà đau đớn lòng.
Lạ gì bỉ sắc tư phong,
Trời xanh quen thói má hồng đánh ghen.
Cảo thơm lần giở trước đèn,
Phong tình cổ lục còn truyền sử xanh.
Rằng năm Gia Tĩnh triều Minh,
Bốn phương phẳng lặng, hai kinh vững vàng.
Có nhà viên ngoại họ Vương,
Gia tư nghĩ cũng thường thường bực trung.
Một trai con thứ rốt lòng,
Vương Quan là chữ, nối dòng nho gia.
Đầu lòng hai ả tố nga,
Thúy Kiều là chị, em là Thúy Vân.
Mai cốt cách, tuyết tinh thần,
Mỗi người một vẻ, mười phân vẹn mười.
Vân xem trang trọng khác vời,
Khuôn trăng đầy đặn, nét ngài nở nang."""


@pytest.fixture(scope="session")
def kieu_text() -> str:
    return KIEU_20


@pytest.fixture(scope="session")
def bpe_tokenizer(tmp_path_factory) -> VietTokenizer:
    prefix = tmp_path_factory.mktemp("tokenizer") / "kieu20"
    return VietTokenizer.train(KIEU_20, prefix, vocab_size=200)
