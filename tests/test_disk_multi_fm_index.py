import pickle

import pytest

from fm_index import MultiFMIndex


@pytest.fixture
def multi_fm_index_empty():
    return MultiFMIndex([], on_disk=True)


@pytest.fixture
def multi_fm_index_empties():
    return MultiFMIndex(["", "", ""], on_disk=True)


@pytest.fixture
def multi_fm_index_ucs1():
    return MultiFMIndex(["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"], on_disk=True)


@pytest.fixture
def multi_fm_index_ucs2():
    return MultiFMIndex(["あいうあいうあいう", "xxあいうあいうxx", "あいうあいあいう"], on_disk=True)


@pytest.fixture
def multi_fm_index_ucs4():
    return MultiFMIndex(["😀😃😀😃😀😃", "xx😀😃😀😃xx", "😀😃😀😀😃"], on_disk=True)


def test_len(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert len(multi_fm_index_empty) == 0
    assert len(multi_fm_index_empties) == 3
    assert len(multi_fm_index_ucs1) == 3
    assert len(multi_fm_index_ucs2) == 3
    assert len(multi_fm_index_ucs4) == 3


def test_str(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert str(multi_fm_index_empty) == "MultiFMIndex(num_docs=0, total_num_chars=0, max_bit=0)"
    assert str(multi_fm_index_empties) == "MultiFMIndex(num_docs=3, total_num_chars=0, max_bit=0)"
    assert str(multi_fm_index_ucs1) == "MultiFMIndex(num_docs=3, total_num_chars=36, max_bit=7)"
    assert str(multi_fm_index_ucs2) == "MultiFMIndex(num_docs=3, total_num_chars=27, max_bit=14)"
    assert str(multi_fm_index_ucs4) == "MultiFMIndex(num_docs=3, total_num_chars=19, max_bit=17)"


def test_item(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.item() == []
    assert multi_fm_index_empties.item() == ["", "", ""]
    assert multi_fm_index_ucs1.item() == [
        "abcabcabcabc",
        "xxabcabcxxabc",
        "abcababcabc",
    ]
    assert multi_fm_index_ucs2.item() == [
        "あいうあいうあいう",
        "xxあいうあいうxx",
        "あいうあいあいう",
    ]
    assert multi_fm_index_ucs4.item() == [
        "😀😃😀😃😀😃",
        "xx😀😃😀😃xx",
        "😀😃😀😀😃",
    ]


def test_contains(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert "" not in multi_fm_index_empty
    assert not multi_fm_index_empty.contains("")
    assert "" in multi_fm_index_empties
    assert multi_fm_index_empties.contains("")
    assert "abcabcabcabc" in multi_fm_index_ucs1
    assert multi_fm_index_ucs1.contains("abcabcabcabc")
    assert "あいうあいうあいう" in multi_fm_index_ucs2
    assert multi_fm_index_ucs2.contains("あいうあいうあいう")
    assert "😀😃😀😃😀😃" in multi_fm_index_ucs4
    assert multi_fm_index_ucs4.contains("😀😃😀😃😀😃")

    assert "xyz" not in multi_fm_index_empty
    assert not multi_fm_index_empty.contains("xyz")
    assert "mnop" not in multi_fm_index_ucs1
    assert not multi_fm_index_ucs1.contains("mnop")
    assert "あいう" not in multi_fm_index_ucs2
    assert not multi_fm_index_ucs2.contains("あいう")
    assert "😀😃" not in multi_fm_index_ucs4
    assert not multi_fm_index_ucs4.contains("😀😃")


def test_count_all(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.count_all("") == 0
    assert multi_fm_index_empties.count_all("") == 3
    assert multi_fm_index_ucs1.count_all("abc") == 10
    assert multi_fm_index_ucs2.count_all("あいう") == 7
    assert multi_fm_index_ucs4.count_all("😀😃😀") == 4


def test_count(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.count("") == {}
    assert multi_fm_index_empties.count("") == {0: 1, 1: 1, 2: 1}
    assert multi_fm_index_ucs1.count("abc") == {0: 4, 1: 3, 2: 3}
    assert multi_fm_index_ucs2.count("あいう") == {0: 3, 1: 2, 2: 2}
    assert multi_fm_index_ucs4.count("😀😃😀") == {0: 2, 1: 1, 2: 1}


def test_count_with_doc_id(
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empties.count("", doc_id=0) == 1
    assert multi_fm_index_ucs1.count("abc", doc_id=0) == 4
    assert multi_fm_index_ucs1.count("abc", doc_id=1) == 3
    assert multi_fm_index_ucs1.count("abc", doc_id=2) == 3
    assert multi_fm_index_ucs2.count("あいう", doc_id=0) == 3
    assert multi_fm_index_ucs2.count("あいう", doc_id=1) == 2
    assert multi_fm_index_ucs4.count("😀😃😀", doc_id=0) == 2
    assert multi_fm_index_ucs4.count("😀😃😀", doc_id=1) == 1


def test_topk(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    # Empty index
    assert multi_fm_index_empty.topk("", k=1) == []
    assert multi_fm_index_empty.topk("a", k=1) == []

    # Empty documents
    result = multi_fm_index_empties.topk("", k=2)
    # Sort by doc_id for consistent comparison since counts are equal
    result_sorted = sorted(result, key=lambda x: x[0])
    assert result_sorted == [(0, 1), (1, 1)]

    # UCS1 - different counts
    assert multi_fm_index_ucs1.topk("abc", k=1) == [(0, 4)]
    assert multi_fm_index_ucs1.topk("abc", k=2) == [(0, 4), (1, 3)]
    # k=3 may have documents 1 and 2 in any order (both have count 3)
    result = multi_fm_index_ucs1.topk("abc", k=3)
    assert len(result) == 3
    assert result[0] == (0, 4)
    assert set(result[1:]) == {(1, 3), (2, 3)}

    # k larger than number of matching documents
    result = multi_fm_index_ucs1.topk("abc", k=5)
    assert len(result) == 3
    assert result[0] == (0, 4)

    # Pattern not found
    assert multi_fm_index_ucs1.topk("xyz", k=2) == []

    # UCS2
    result = multi_fm_index_ucs2.topk("あいう", k=2)
    assert result[0] == (0, 3)
    # Documents 1 and 2 both have count 2, order may vary
    assert result[1][1] == 2
    assert result[1][0] in [1, 2]

    # UCS4
    result = multi_fm_index_ucs4.topk("😀😃😀", k=2)
    assert result[0] == (0, 2)
    # Documents 1 and 2 both have count 1, order may vary
    assert result[1][1] == 1
    assert result[1][0] in [1, 2]


def test_topk_errors(multi_fm_index_ucs1):
    # k must be greater than 0
    with pytest.raises(ValueError, match="k must be greater than 0"):
        multi_fm_index_ucs1.topk("abc", k=0)


def test_locate(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.locate("") == {}
    assert multi_fm_index_empties.locate("") == {0: [0], 1: [0], 2: [0]}
    assert multi_fm_index_ucs1.locate("abc") == {
        0: [9, 6, 3, 0],
        1: [10, 2, 5],
        2: [8, 0, 5],
    }
    assert multi_fm_index_ucs2.locate("あいう") == {
        0: [6, 3, 0],
        1: [5, 2],
        2: [5, 0],
    }
    assert multi_fm_index_ucs4.locate("😀😃😀") == {
        0: [2, 0],
        1: [2],
        2: [0],
    }


def test_locate_with_doc_id(
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empties.locate("", doc_id=0) == [0]
    assert multi_fm_index_ucs1.locate("abc", doc_id=0) == [9, 6, 3, 0]
    assert multi_fm_index_ucs1.locate("abc", doc_id=1) == [10, 2, 5]
    assert multi_fm_index_ucs1.locate("abc", doc_id=2) == [8, 0, 5]
    assert multi_fm_index_ucs2.locate("あいう", doc_id=0) == [6, 3, 0]
    assert multi_fm_index_ucs2.locate("あいう", doc_id=1) == [5, 2]
    assert multi_fm_index_ucs4.locate("😀😃😀", doc_id=0) == [2, 0]
    assert multi_fm_index_ucs4.locate("😀😃😀", doc_id=1) == [2]


def test_iter_locate(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert list(multi_fm_index_empty.iter_locate("")) == []
    assert list(multi_fm_index_empties.iter_locate("")) == [(2, 0), (1, 0), (0, 0)]
    assert list(multi_fm_index_ucs1.iter_locate("abc")) == [
        (2, 8),
        (1, 10),
        (0, 9),
        (2, 0),
        (2, 5),
        (0, 6),
        (0, 3),
        (0, 0),
        (1, 2),
        (1, 5),
    ]
    assert list(multi_fm_index_ucs2.iter_locate("あいう")) == [
        (2, 5),
        (0, 6),
        (1, 5),
        (2, 0),
        (0, 3),
        (1, 2),
        (0, 0),
    ]
    assert list(multi_fm_index_ucs4.iter_locate("😀😃😀")) == [
        (2, 0),
        (0, 2),
        (1, 2),
        (0, 0),
    ]


def test_iter_locate_with_doc_id(
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert list(multi_fm_index_empties.iter_locate("", doc_id=0)) == [0]
    assert list(multi_fm_index_ucs1.iter_locate("abc", doc_id=0)) == [9, 6, 3, 0]
    assert list(multi_fm_index_ucs1.iter_locate("abc", doc_id=1)) == [10, 2, 5]
    assert list(multi_fm_index_ucs2.iter_locate("あいう", doc_id=0)) == [6, 3, 0]
    assert list(multi_fm_index_ucs2.iter_locate("あいう", doc_id=1)) == [5, 2]
    assert list(multi_fm_index_ucs4.iter_locate("😀😃😀", doc_id=0)) == [2, 0]
    assert list(multi_fm_index_ucs4.iter_locate("😀😃😀", doc_id=1)) == [2]


def test_startswith(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.startswith("") == []
    assert multi_fm_index_empties.startswith("") == [2, 1, 0]
    assert multi_fm_index_ucs1.startswith("abc") == [2, 0]
    assert multi_fm_index_ucs2.startswith("あいう") == [2, 0]
    assert multi_fm_index_ucs4.startswith("😀😃😀") == [2, 0]


def test_endswith(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.endswith("") == []
    assert multi_fm_index_empties.endswith("") == [2, 1, 0]
    assert multi_fm_index_ucs1.endswith("abc") == [2, 1, 0]
    assert multi_fm_index_ucs2.endswith("あいう") == [2, 0]
    assert multi_fm_index_ucs4.endswith("😀😃") == [2, 0]


def test_large_texts():
    large_texts = [
        ("mississippi" + "にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊") * 100,
        ("🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi" + "にわにはにわにわとりがいる") * 100,
        ("にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi") * 100,
        ("mississippi" + "にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊") * 100,
        ("🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi" + "にわにはにわにわとりがいる") * 100,
        ("にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi") * 100,
        ("mississippi" + "にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊") * 100,
        ("🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi" + "にわにはにわにわとりがいる") * 100,
        ("にわにはにわにわとりがいる" + "🏰🐉🔥🌊🏰 🐉🔥🌊 ⚔️🐉🔥🌊" + "mississippi") * 100,
    ]
    multi_fm_index_large = MultiFMIndex(large_texts, on_disk=True)

    for pattern in ["miss", "にわに", "🏰🐉🔥"]:
        for doc_id, count in multi_fm_index_large.count(pattern).items():
            assert count == large_texts[doc_id].count(pattern)
        for doc_id, offsets in multi_fm_index_large.locate(pattern).items():
            for offset in offsets:
                assert large_texts[doc_id][offset : offset + len(pattern)] == pattern


def test_pickle_empty():
    multi_fm_index = MultiFMIndex([], on_disk=True)
    pickled = pickle.dumps(multi_fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 0
    assert unpickled.item() == []
    assert str(unpickled) == "MultiFMIndex(num_docs=0, total_num_chars=0, max_bit=0)"


def test_pickle_empties():
    multi_fm_index = MultiFMIndex(["", "", ""], on_disk=True)
    pickled = pickle.dumps(multi_fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 3
    assert unpickled.item() == ["", "", ""]
    assert str(unpickled) == "MultiFMIndex(num_docs=3, total_num_chars=0, max_bit=0)"


def test_pickle_ucs1():
    multi_fm_index = MultiFMIndex(["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"], on_disk=True)
    pickled = pickle.dumps(multi_fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 3
    assert unpickled.item() == ["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"]
    assert str(unpickled) == "MultiFMIndex(num_docs=3, total_num_chars=36, max_bit=7)"
    assert unpickled.count("abc") == {0: 4, 1: 3, 2: 3}
    assert unpickled.locate("abc") == {0: [9, 6, 3, 0], 1: [10, 2, 5], 2: [8, 0, 5]}


def test_pickle_ucs2():
    multi_fm_index = MultiFMIndex(["あいうあいうあいう", "xxあいうあいうxx", "あいうあいあいう"], on_disk=True)
    pickled = pickle.dumps(multi_fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 3
    assert unpickled.item() == ["あいうあいうあいう", "xxあいうあいうxx", "あいうあいあいう"]
    assert str(unpickled) == "MultiFMIndex(num_docs=3, total_num_chars=27, max_bit=14)"
    assert unpickled.count("あいう") == {0: 3, 1: 2, 2: 2}
    assert unpickled.locate("あいう") == {0: [6, 3, 0], 1: [5, 2], 2: [5, 0]}


def test_pickle_ucs4():
    multi_fm_index = MultiFMIndex(["😀😃😀😃😀😃", "xx😀😃😀😃xx", "😀😃😀😀😃"], on_disk=True)
    pickled = pickle.dumps(multi_fm_index)
    unpickled = pickle.loads(pickled)

    assert len(unpickled) == 3
    assert unpickled.item() == ["😀😃😀😃😀😃", "xx😀😃😀😃xx", "😀😃😀😀😃"]
    assert str(unpickled) == "MultiFMIndex(num_docs=3, total_num_chars=19, max_bit=17)"
    assert unpickled.count("😀😃😀") == {0: 2, 1: 1, 2: 1}
    assert unpickled.locate("😀😃😀") == {0: [2, 0], 1: [2], 2: [0]}
