"""Tải dữ liệu cho repo.

    python scripts/download_data.py              # Truyện Kiều (bắt buộc, ~100 KB)
    python scripts/download_data.py --fasttext   # thêm vector fastText tiếng Việt rút gọn (~35 MB tải về)

Xem data/README.md để biết nguồn và giấy phép của từng loại dữ liệu.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # để import được vietformer

from vietformer.data import KIEU_PATH, download_truyen_kieu, get_tokenizer, load_corpus  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="tải lại kể cả khi file đã có")
    parser.add_argument("--fasttext", action="store_true", help="tải thêm vector fastText cc.vi.300 rút gọn")
    parser.add_argument("--max-words", type=int, default=50_000, help="số từ fastText đầu tiên cần giữ")
    args = parser.parse_args()

    path = download_truyen_kieu(KIEU_PATH, force=args.force)
    text = load_corpus(path)
    print(f"Corpus: {path} – {len(text.splitlines())} dòng, {len(text.split())} âm tiết, {len(text)} ký tự")
    tokenizer = get_tokenizer(path)
    print(f"Tokenizer: {tokenizer.model_path} (vocab_size={tokenizer.vocab_size})")

    if args.fasttext:
        from vietformer.word_vectors import download_fasttext_subset

        vectors = download_fasttext_subset(max_words=args.max_words, force=args.force)
        print(f"fastText: {len(vectors)} từ × {vectors.dim} chiều")


if __name__ == "__main__":
    main()
