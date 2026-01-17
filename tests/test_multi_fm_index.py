import pytest

from fm_index import MultiFMIndex


@pytest.fixture
def multi_fm_index_empty():
    return MultiFMIndex([])


@pytest.fixture
def multi_fm_index_empties():
    return MultiFMIndex(["", "", ""])


@pytest.fixture
def multi_fm_index_ucs1():
    return MultiFMIndex(["abcabcabcabc", "xxabcabcxxabc", "abcababcabc"])


@pytest.fixture
def multi_fm_index_ucs2():
    return MultiFMIndex(["あいうあいうあいう", "xxあいうあいうxx", "あいうあいあいう"])


@pytest.fixture
def multi_fm_index_ucs4():
    return MultiFMIndex(["😀😃😀😃😀😃", "xx😀😃😀😃xx", "😀😃😀😀😃"])


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
    assert (
        str(multi_fm_index_empty)
        == "MultiFMIndex(num_docs=0, total_num_chars=0, num_unique_chars=0, code_unit=ucs1)"
    )
    assert (
        str(multi_fm_index_empties)
        == "MultiFMIndex(num_docs=3, total_num_chars=0, num_unique_chars=0, code_unit=ucs1)"
    )
    assert (
        str(multi_fm_index_ucs1)
        == "MultiFMIndex(num_docs=3, total_num_chars=36, num_unique_chars=4, code_unit=ucs1)"
    )
    assert (
        str(multi_fm_index_ucs2)
        == "MultiFMIndex(num_docs=3, total_num_chars=27, num_unique_chars=4, code_unit=ucs2)"
    )
    assert (
        str(multi_fm_index_ucs4)
        == "MultiFMIndex(num_docs=3, total_num_chars=19, num_unique_chars=3, code_unit=ucs4)"
    )


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


def test_locate(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.locate("") == {}
    assert multi_fm_index_empties.locate("") == {0: [0], 1: [0], 2: [0]}
    assert {
        key: sorted(value)
        for key, value in multi_fm_index_ucs1.locate("abc").items()
    } == {
        0: [0, 3, 6, 9],
        1: [2, 5, 10],
        2: [0, 5, 8],
    }
    assert {
        key: sorted(value)
        for key, value in multi_fm_index_ucs2.locate("あいう").items()
    } == {
        0: [0, 3, 6],
        1: [2, 5],
        2: [0, 5],
    }
    assert {
        key: sorted(value)
        for key, value in multi_fm_index_ucs4.locate("😀😃😀").items()
    } == {
        0: [0, 2],
        1: [2],
        2: [0],
    }


def test_iter_locate(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert list(multi_fm_index_empty.iter_locate("")) == []
    assert sorted(multi_fm_index_empties.iter_locate("")) == [(0, 0), (1, 0), (2, 0)]
    assert sorted(multi_fm_index_ucs1.iter_locate("abc")) == [
        (0, 0),
        (0, 3),
        (0, 6),
        (0, 9),
        (1, 2),
        (1, 5),
        (1, 10),
        (2, 0),
        (2, 5),
        (2, 8),
    ]
    assert sorted(multi_fm_index_ucs2.iter_locate("あいう")) == [
        (0, 0),
        (0, 3),
        (0, 6),
        (1, 2),
        (1, 5),
        (2, 0),
        (2, 5),
    ]
    assert sorted(multi_fm_index_ucs4.iter_locate("😀😃😀")) == [
        (0, 0),
        (0, 2),
        (1, 2),
        (2, 0),
    ]


def test_startswith(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.startswith("") == []
    assert sorted(multi_fm_index_empties.startswith("")) == [0, 1, 2]
    assert sorted(multi_fm_index_ucs1.startswith("abc")) == [0, 2]
    assert sorted(multi_fm_index_ucs2.startswith("あいう")) == [0, 2]
    assert sorted(multi_fm_index_ucs4.startswith("😀😃😀")) == [0, 2]


def test_endswith(
    multi_fm_index_empty,
    multi_fm_index_empties,
    multi_fm_index_ucs1,
    multi_fm_index_ucs2,
    multi_fm_index_ucs4,
):
    assert multi_fm_index_empty.endswith("") == []
    assert sorted(multi_fm_index_empties.endswith("")) == [0, 1, 2]
    assert sorted(multi_fm_index_ucs1.endswith("abc")) == [0, 1, 2]
    assert sorted(multi_fm_index_ucs2.endswith("あいう")) == [0, 2]
    assert sorted(multi_fm_index_ucs4.endswith("😀😃")) == [0, 2]


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
    multi_fm_index_large = MultiFMIndex(large_texts)

    for pattern in ["miss", "にわに", "🏰🐉🔥"]:
        for doc_id, count in multi_fm_index_large.count(pattern).items():
            assert count == large_texts[doc_id].count(pattern)
        for doc_id, offsets in multi_fm_index_large.locate(pattern).items():
            for offset in offsets:
                assert large_texts[doc_id][offset : offset + len(pattern)] == pattern
