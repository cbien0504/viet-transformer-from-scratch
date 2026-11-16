import unicodedata

from vietformer.data import clean_poem_wikitext
from vietformer.tokenizer import CharTokenizer, SyllableTokenizer, normalize_text


def test_normalize_text_turns_nfd_into_nfc():
    nfd = unicodedata.normalize("NFD", "Tiếng Việt")
    assert nfd != "Tiếng Việt" and len(nfd) > len("Tiếng Việt")  # cùng chữ nhưng khác mã Unicode
    assert normalize_text(nfd) == "Tiếng Việt"
    assert normalize_text("a\r\nb\rc") == "a\nb\nc"


def test_char_tokenizer_roundtrip_and_unknown(kieu_text):
    tok = CharTokenizer(kieu_text)
    assert tok.decode(tok.encode(kieu_text)) == kieu_text
    assert tok.encode("ê")[0] != 0                 # ký tự có trong corpus
    assert tok.encode("ζ") == [0]                  # ký tự lạ -> <unk>


def test_syllable_tokenizer_splits_syllables_and_punctuation(kieu_text):
    assert SyllableTokenizer.split("Trăm năm, người ta.\nChữ") == ["Trăm", "năm", ",", "người", "ta", ".", "\n", "Chữ"]
    tok = SyllableTokenizer(kieu_text)
    assert tok.decode(tok.encode("Trăm năm trong cõi người ta,")) == "Trăm năm trong cõi người ta,"
    assert tok.encode("máy tính")[0] == 0          # "máy" không có trong 20 câu đầu -> <unk>


def test_bpe_roundtrip_keeps_newlines(bpe_tokenizer, kieu_text):
    ids = bpe_tokenizer.encode(kieu_text)
    assert bpe_tokenizer.decode(ids) == kieu_text
    assert ids.count(bpe_tokenizer.newline_id) == kieu_text.count("\n")


def test_bpe_gives_same_ids_for_nfc_and_nfd(bpe_tokenizer):
    text = "Trăm năm trong cõi người ta"
    assert bpe_tokenizer.encode(unicodedata.normalize("NFD", text)) == bpe_tokenizer.encode(text)


def test_bpe_unseen_syllable_is_split_not_unknown(bpe_tokenizer):
    ids = bpe_tokenizer.encode("Thăm")  # không có trong corpus nhỏ, nhưng ghép được từ các mảnh
    assert bpe_tokenizer.unk_id not in ids
    assert bpe_tokenizer.decode(ids) == "Thăm"


def test_clean_poem_wikitext_removes_wiki_markup():
    wikitext = (
        "{{đầu đề|tựa đề=X}}\n<poem>\n\tTrăm năm trong cõi người ta,\n"
        "{{số|2}}\tChữ ''tài'' chữ [[w:Mệnh|mệnh]]<ref>chú thích</ref> khéo là ghét nhau.\n</poem>\n{{PD-old}}"
    )
    assert clean_poem_wikitext(wikitext) == ["Trăm năm trong cõi người ta,", "Chữ tài chữ mệnh khéo là ghét nhau."]
