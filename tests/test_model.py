import math
from pathlib import Path

import pytest
import torch

from vietformer.generate import generate, next_token_probs
from vietformer.model import GPTConfig, MiniGPT, load_checkpoint, save_checkpoint
from vietformer.train import get_batch, split_train_val

TINY = GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, d_model=32, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return MiniGPT(TINY)


def test_logits_shape_and_loss(model):
    idx = torch.randint(0, 50, (3, 10))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, 10, 50)
    assert loss.ndim == 0
    assert model(idx)[1] is None


def test_initial_loss_is_close_to_uniform_guess(model):
    idx = torch.randint(0, 50, (8, 16))
    _, loss = model(idx, torch.randint(0, 50, (8, 16)))
    assert abs(loss.item() - math.log(50)) < 0.3                  # lúc đầu ~ đoán đều trên 50 token


def test_model_is_causal(model):
    model.eval()
    idx = torch.randint(0, 50, (1, 16))
    changed = idx.clone()
    changed[0, 9:] = (changed[0, 9:] + 1) % 50                    # đổi mọi token từ vị trí 9
    assert torch.allclose(model(idx)[0][:, :9], model(changed)[0][:, :9], atol=1e-6)


def test_attention_maps_are_probabilities(model):
    model(torch.randint(0, 50, (2, 12)))
    maps = model.attention_maps()
    assert len(maps) == TINY.n_layer
    assert maps[0].shape == (2, TINY.n_head, 12, 12)
    assert torch.allclose(maps[0].sum(-1), torch.ones(2, TINY.n_head, 12))


def test_can_overfit_one_small_batch(model):
    torch.manual_seed(0)
    x = torch.randint(0, 50, (4, 16))
    y = torch.randint(0, 50, (4, 16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        _, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1                                      # học thuộc được một batch nhỏ


def test_sinusoidal_and_pretrained_variants_run():
    cfg = GPTConfig(vocab_size=50, block_size=16, n_layer=1, n_head=2, d_model=32,
                    pos_encoding="sinusoidal", pretrained_dim=300, freeze_embedding=True)
    model = MiniGPT(cfg)
    vectors = torch.randn(50, 300)
    model.tok_emb.load_vectors(vectors)
    x = torch.randint(0, 50, (2, 16))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        _, loss = model(x, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert torch.equal(model.tok_emb.table.weight, vectors)      # freeze: bảng không đổi


def test_checkpoint_roundtrip(tmp_path, model):
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer_path="abc.model")
    loaded, checkpoint = load_checkpoint(path)
    idx = torch.randint(0, 50, (1, 8))
    model.eval()
    assert torch.allclose(model(idx)[0], loaded(idx)[0])
    assert checkpoint["tokenizer_path"] == "abc.model"


def test_checkpoint_stores_portable_tokenizer_path(tmp_path, model, bpe_tokenizer):
    from vietformer.generate import load_model_and_tokenizer
    from vietformer.utils import ROOT_DIR, portable_path, resolve_path

    inside = ROOT_DIR / "data" / "x.model"
    assert Path(portable_path(inside)) == Path("data/x.model")   # trong repo -> đường dẫn tương đối
    assert resolve_path(portable_path(inside)) == inside

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tokenizer_path=portable_path(bpe_tokenizer.model_path))
    _, tokenizer, _ = load_model_and_tokenizer(path)
    assert tokenizer.vocab_size == bpe_tokenizer.vocab_size


def test_get_batch_targets_are_inputs_shifted_by_one():
    data = torch.arange(100)
    train, val = split_train_val(data, val_fraction=0.1)
    assert len(train) == 90 and len(val) == 10
    x, y = get_batch(train, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    assert torch.equal(y, x + 1)                                  # y[t] = x[t + 1]


def test_next_token_probs_temperature_and_top_k():
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])
    for temperature in (0.5, 1.0, 2.0):
        assert torch.allclose(next_token_probs(logits, temperature).sum(), torch.tensor(1.0))
    assert next_token_probs(logits, 0.5)[0, 0] > next_token_probs(logits, 2.0)[0, 0]  # T nhỏ -> nhọn hơn
    assert torch.equal(next_token_probs(logits, 0.0), torch.tensor([[1.0, 0, 0, 0]]))  # greedy
    top2 = next_token_probs(logits, 1.0, top_k=2)
    assert torch.all(top2[0, 2:] == 0) and torch.allclose(top2.sum(), torch.tensor(1.0))


def test_generate_appends_tokens(model):
    idx = torch.tensor([[1, 2, 3]])
    out = generate(model, idx, max_new_tokens=20, top_k=5)       # dài hơn block_size: phải cắt ngữ cảnh
    assert out.shape == (1, 23)
    assert torch.equal(out[:, :3], idx)
    assert out.max() < 50
