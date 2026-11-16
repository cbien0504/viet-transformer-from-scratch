"""Huấn luyện MiniGPT từ dòng lệnh (cùng code với notebook 07).

    python scripts/train.py                                  # cấu hình mặc định, khoảng 2 phút trên CPU
    python scripts/train.py --preset tiny --max-steps 300    # chạy thử thật nhanh (~15 giây)
    python scripts/train.py --embedding fasttext --freeze    # thử nghiệm (b): embedding fastText, đóng băng
    python scripts/train.py --embedding fasttext             # thử nghiệm (b): embedding fastText, finetune
    python scripts/train.py --pos sinusoidal                 # positional encoding sin/cos
    python scripts/train.py --corpus data/van_ban_cua_ban.txt --vocab-size 8000

Kết quả nằm trong checkpoints/: file .pt (cấu hình + trọng số + lịch sử loss) và hình đường loss .png.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # để import được vietformer

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # vẽ ra file, không cần cửa sổ
import matplotlib.pyplot as plt  # noqa: E402

from vietformer import viz  # noqa: E402
from vietformer.data import KIEU_PATH, encode_corpus, get_tokenizer, load_corpus  # noqa: E402
from vietformer.generate import generate_text  # noqa: E402
from vietformer.model import GPTConfig, MiniGPT  # noqa: E402
from vietformer.train import split_train_val, train  # noqa: E402
from vietformer.utils import CHECKPOINT_DIR, count_parameters, get_device, portable_path, set_seed  # noqa: E402
from vietformer.word_vectors import build_embedding_matrix, load_word_vectors  # noqa: E402

PRESETS = {
    "tiny": dict(n_layer=2, n_head=2, d_model=64, block_size=32),    # ~0,5 triệu tham số
    "small": dict(n_layer=4, n_head=4, d_model=128, block_size=64),  # ~1,8 triệu tham số (mặc định)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", default=str(KIEU_PATH), help="file văn bản UTF-8 (mặc định: Truyện Kiều)")
    p.add_argument("--vocab-size", type=int, default=4000)
    p.add_argument("--preset", choices=PRESETS, default="small", help="cỡ mô hình có sẵn")
    p.add_argument("--n-layer", type=int, help="ghi đè preset")
    p.add_argument("--n-head", type=int, help="ghi đè preset")
    p.add_argument("--d-model", type=int, help="ghi đè preset")
    p.add_argument("--block-size", type=int, help="ghi đè preset")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos", choices=["learned", "sinusoidal"], default="learned", help="kiểu positional encoding")
    p.add_argument("--embedding", choices=["scratch", "fasttext", "word2vec"], default="scratch",
                   help="scratch = học từ đầu; fasttext / word2vec = khởi tạo từ vector có sẵn")
    p.add_argument("--freeze", action="store_true", help="đóng băng bảng embedding có sẵn (chỉ học lớp chiếu)")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None, help="cpu / cuda / mps (mặc định: tự chọn)")
    p.add_argument("--out", default=None, help="đường dẫn checkpoint (mặc định: checkpoints/minigpt_<corpus>.pt)")
    p.add_argument("--prompt", default="Trăm năm trong cõi người ta,", help="câu mở đầu để sinh thử sau khi train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    # 1) Dữ liệu: tokenizer + dãy token id, chia train/val
    tokenizer = get_tokenizer(args.corpus, args.vocab_size)
    data = encode_corpus(tokenizer, args.corpus)
    train_data, val_data = split_train_val(data)
    print(f"Corpus {Path(args.corpus).name}: {len(data)} token (train {len(train_data)}, val {len(val_data)})")

    # 2) Mô hình: mọi siêu tham số nằm trong GPTConfig
    sizes = dict(PRESETS[args.preset])
    for key in ("n_layer", "n_head", "d_model", "block_size"):
        if getattr(args, key) is not None:
            sizes[key] = getattr(args, key)
    vectors, pretrained_dim = None, None
    if args.embedding != "scratch":
        word_vectors = load_word_vectors(prefer=args.embedding, text=load_corpus(args.corpus))
        vectors, coverage = build_embedding_matrix(tokenizer, word_vectors)
        pretrained_dim = word_vectors.dim
        print(f"Embedding khởi tạo từ {word_vectors.source}: {coverage:.1%} token có vector")
    config = GPTConfig(vocab_size=tokenizer.vocab_size, dropout=args.dropout, pos_encoding=args.pos,
                       pretrained_dim=pretrained_dim, freeze_embedding=args.freeze, **sizes)
    model = MiniGPT(config)
    if vectors is not None:
        model.tok_emb.load_vectors(vectors)
    model.to(device)
    print(config)
    print(f"{count_parameters(model):,} tham số ({count_parameters(model, trainable_only=True):,} được huấn luyện), thiết bị: {device}")

    # 3) Huấn luyện (checkpoint có val loss thấp nhất được lưu lại)
    out = Path(args.out) if args.out else CHECKPOINT_DIR / f"minigpt_{Path(args.corpus).stem}.pt"
    history = train(model, train_data, val_data, max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
                    eval_interval=args.eval_interval, checkpoint_path=out,
                    checkpoint_extra={"tokenizer_path": portable_path(tokenizer.model_path), "args": vars(args)})
    viz.set_style()
    viz.plot_loss(history)
    plt.savefig(out.with_suffix(".png"), dpi=120, bbox_inches="tight")
    print(f"Đã lưu checkpoint: {out}\nĐã lưu đường loss: {out.with_suffix('.png')}")

    # 4) Sinh thử vài đoạn với các temperature khác nhau
    for temperature in (0.5, 0.8, 1.0):
        print(f"\n--- temperature = {temperature}, top_k = 50 ---")
        print(generate_text(model, tokenizer, args.prompt, max_new_tokens=60, temperature=temperature, top_k=50, seed=args.seed))


if __name__ == "__main__":
    main()
