import pickle

import pytest

from fm_index import FMIndex


@pytest.fixture
def fm_index_empty():
    return FMIndex("")


@pytest.fixture
def fm_index_ucs1():
    return FMIndex("mississippi")


@pytest.fixture
def fm_index_ucs2():
    return FMIndex("にわにはにわにわとりがいる")


@pytest.fixture
def fm_index_ucs4():
    return FMIndex("🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊")


def test_len(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert len(fm_index_empty) == 0
    assert len(fm_index_ucs1) == 11
    assert len(fm_index_ucs2) == 13
    assert len(fm_index_ucs4) == 15


def test_str(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert str(fm_index_empty) == "FMIndex(len=0, max_bit=0)"
    assert str(fm_index_ucs1) == "FMIndex(len=11, max_bit=7)"
    assert str(fm_index_ucs2) == "FMIndex(len=13, max_bit=14)"
    assert str(fm_index_ucs4) == "FMIndex(len=15, max_bit=17)"


def test_item(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert fm_index_empty.item() == ""
    assert fm_index_ucs1.item() == "mississippi"
    assert fm_index_ucs2.item() == "にわにはにわにわとりがいる"
    assert fm_index_ucs4.item() == "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"


def test_contains(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert "" in fm_index_empty
    assert fm_index_empty.contains("")
    assert "issi" in fm_index_ucs1
    assert fm_index_ucs1.contains("issi")
    assert "にわ" in fm_index_ucs2
    assert fm_index_ucs2.contains("にわ")
    assert "🐉🔥🌊" in fm_index_ucs4
    assert fm_index_ucs4.contains("🐉🔥🌊")

    assert "abc" not in fm_index_empty
    assert not fm_index_empty.contains("abc")
    assert "xyz" not in fm_index_ucs1
    assert not fm_index_ucs1.contains("xyz")
    assert "こんにちは" not in fm_index_ucs2
    assert not fm_index_ucs2.contains("こんにちは")
    assert "😀😃😄" not in fm_index_ucs4
    assert not fm_index_ucs4.contains("😀😃😄")


def test_count(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert fm_index_empty.count("") == 1
    assert fm_index_ucs1.count("issi") == 2
    assert fm_index_ucs2.count("にわ") == 3
    assert fm_index_ucs4.count("🐉🔥🌊") == 3


def test_locate(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert fm_index_empty.locate("") == [0]
    assert fm_index_ucs1.locate("issi") == [4, 1]
    assert fm_index_ucs2.locate("にわ") == [6, 0, 4]
    assert fm_index_ucs4.locate("🐉🔥🌊") == [12, 6, 1]


def test_iter_locate(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert list(fm_index_empty.iter_locate("")) == [0]
    assert list(fm_index_ucs1.iter_locate("issi")) == [4, 1]
    assert list(fm_index_ucs2.iter_locate("にわ")) == [6, 0, 4]
    assert list(fm_index_ucs4.iter_locate("🐉🔥🌊")) == [12, 6, 1]


def test_startswith(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert fm_index_empty.startswith("") is True
    assert fm_index_ucs1.startswith("mis") is True
    assert fm_index_ucs2.startswith("にわに") is True
    assert fm_index_ucs4.startswith("🏰🐉") is True


def test_endswith(fm_index_empty, fm_index_ucs1, fm_index_ucs2, fm_index_ucs4):
    assert fm_index_empty.endswith("") is True
    assert fm_index_ucs1.endswith("ppi") is True
    assert fm_index_ucs2.endswith("とりがいる") is True
    assert fm_index_ucs4.endswith("⚔️🐉🔥🌊") is True


def test_large():
    large_text = ("mississippi" + "にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊") * 1000
    fm_index_large = FMIndex(large_text)

    for pattern in ["miss", "にわに", "🏰🐉🔥"]:
        assert fm_index_large.count(pattern) == large_text.count(pattern)
        for offset in fm_index_large.locate(pattern):
            assert large_text[offset : offset + len(pattern)] == pattern


def test_pickle_empty():
    fm_index = FMIndex("")
    pickled = pickle.dumps(fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 0
    assert unpickled.item() == ""
    assert str(unpickled) == "FMIndex(len=0, max_bit=0)"


def test_pickle_ucs1():
    fm_index = FMIndex("mississippi")
    pickled = pickle.dumps(fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 11
    assert unpickled.item() == "mississippi"
    assert str(unpickled) == "FMIndex(len=11, max_bit=7)"
    assert unpickled.count("issi") == 2
    assert unpickled.locate("issi") == [4, 1]
    assert unpickled.startswith("mis")
    assert unpickled.endswith("ppi")


def test_pickle_ucs2():
    fm_index = FMIndex("にわにはにわにわとりがいる")
    pickled = pickle.dumps(fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 13
    assert unpickled.item() == "にわにはにわにわとりがいる"
    assert str(unpickled) == "FMIndex(len=13, max_bit=14)"
    assert unpickled.count("にわ") == 3
    assert unpickled.locate("にわ") == [6, 0, 4]


def test_pickle_ucs4():
    fm_index = FMIndex("🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊")
    pickled = pickle.dumps(fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 15
    assert unpickled.item() == "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊"
    assert str(unpickled) == "FMIndex(len=15, max_bit=17)"
    assert unpickled.count("🐉🔥🌊") == 3
    assert unpickled.locate("🐉🔥🌊") == [12, 6, 1]
