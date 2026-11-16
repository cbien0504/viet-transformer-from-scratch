import numpy as np
import torch

from vietformer.embedding import TokenEmbedding
from vietformer.word_vectors import WordVectors, build_embedding_matrix, train_word2vec


def test_token_embedding_is_a_lookup_table():
    emb = TokenEmbedding(vocab_size=10, d_model=4)
    idx = torch.tensor([[1, 2, 3], [3, 2, 1]])
    out = emb(idx)
    assert out.shape == (2, 3, 4)
    assert torch.equal(out[0, 0], emb.table.weight[1])           # token id 1 -> hàng thứ 1
    one_hot = torch.nn.functional.one_hot(idx, 10).float()
    assert torch.allclose(one_hot @ emb.table.weight, out)       # = nhân one-hot với ma trận


def test_pretrained_embedding_projects_and_freezes():
    emb = TokenEmbedding(vocab_size=10, d_model=8, pretrained_dim=300, freeze=True)
    vectors = torch.randn(10, 300)
    emb.load_vectors(vectors)
    assert torch.equal(emb.table.weight, vectors)
    out = emb(torch.tensor([[1, 2]]))
    assert out.shape == (1, 2, 8)                                # 300 chiều được chiếu xuống 8
    out.sum().backward()
    assert emb.table.weight.grad is None                         # bảng bị đóng băng
    assert emb.proj.weight.grad is not None                      # lớp chiếu vẫn học


class FakeTokenizer:
    pieces = ["<unk>", "<n>", "▁vua", "▁Hà", "ng", ","]
    vocab_size = len(pieces)

    def id_to_piece(self, i):
        return self.pieces[i]


def test_build_embedding_matrix_copies_known_words():
    wv = WordVectors(["vua", "hà", ","], np.arange(9, dtype=np.float32).reshape(3, 3))
    matrix, coverage = build_embedding_matrix(FakeTokenizer(), wv)
    assert matrix.shape == (6, 3)
    assert torch.equal(matrix[2], torch.tensor([0.0, 1.0, 2.0]))  # ▁vua -> vua
    assert torch.equal(matrix[3], torch.tensor([3.0, 4.0, 5.0]))  # ▁Hà -> hà (thử chữ thường)
    assert coverage == 3 / 6


def test_word_vectors_phrase_and_similarity():
    wv = WordVectors(["hoàng", "đế", "vua", "chó"], np.array([[1, 0], [0, 1], [1, 1], [-1, -1]], dtype=np.float32))
    assert np.allclose(wv.phrase_vector("hoàng_đế"), [0.5, 0.5])
    assert wv.phrase_vector("máy_bay") is None
    assert wv.most_similar("hoàng_đế", topn=1)[0][0] == "vua"


def test_word2vec_fallback_trains(kieu_text):
    wv = train_word2vec(kieu_text * 5, dim=16, epochs=2)
    assert wv.dim == 16
    assert wv.get("Trăm") is not None                            # tra chữ thường "trăm"
    assert np.isfinite(wv.vectors).all()
