# FM Index

[![CI](https://github.com/math-hiyoko/fm-index/actions/workflows/CI.yml/badge.svg)](https://github.com/math-hiyoko/fm-index/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/math-hiyoko/fm-index/graph/badge.svg?token=37GS49DHDH)](https://codecov.io/gh/math-hiyoko/fm-index)
![PyPI - Version](https://img.shields.io/pypi/v/fm-index)
![PyPI - License](https://img.shields.io/pypi/l/fm-index)
![PyPI - PythonVersion](https://img.shields.io/pypi/pyversions/fm-index)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/fm-index)
![PyPI - Types](https://img.shields.io/pypi/types/fm-index)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/fm-index?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=PyPI%20downloads)](https://pepy.tech/projects/fm-index)
![PyPI - Format](https://img.shields.io/pypi/format/fm-index)
![Rust](https://img.shields.io/badge/powered%20by-Rust-orange)


High-performance FM-index powered by Rust, enabling fast substring search and count/locate queries.

- PyPI: https://pypi.org/project/fm-index
- Document: https://math-hiyoko.github.io/fm-index
- Repository: https://github.com/math-hiyoko/fm-index


## Quick Start

```bash
$ pip install fm-index
```

### FMIndex
#### Construction Complexity
The time and space complexity: `O(|data| log σ)`  
where σ is the size of the alphabet (e.g. 2⁸ for UCS-1, 2¹⁶ for UCS-2, etc.).  

```python
>>> from fm_index import FMIndex
>>> 
>>> # Create a FMIndex
>>> genome = "ACGTACGTTGACCTGACTGACTGACTGACGATCGATCGATCGATCGATCG" * 10
>>> fm = FMIndex(data=genome)
>>> fm
FMIndex("ACGTACGTTGACCTGACTGACTGACTGACGATCGATCGATCG...
```

#### Count Substring Occurrences (count)
Counts how many times a pattern appears in the indexed data.  
Time complexity does not depend on the length of the original data.  

```python
>>> fm.count(pattern="GACTGACT")
20
```

#### Locate substring offset (locate)
Finds all starting positions where a pattern occurs.  
Time complexity does not depend on the length of the original data.  

```python
>>> pattern = "GACTGACT"
>>> fm.locate(pattern="GACTGACT")
[468, 418, 368, 318, 268, 218, 168, 118, 68, 18,
 464, 414, 364, 314, 264, 214, 164, 114, 64, 14]
>>> genome[468 : 468 + len(pattern)]
'GACTGACT'
```
⚠️ The order of results is not guaranteed.

### MultiFMIndex
MultiFMIndex extends FMIndex to support multiple documents.

#### Construction Complexity
The time and space complexity: `O(|''.join(data)| log σ)`  
where σ is the size of the alphabet.

```python
>>> from fm_index import MultiFMIndex
>>>
>>> # Create a MultiFMIndex
>>> documents = [
...     "政府はAI研究の支援を強化すると発表した。",
...     "政府は新たなデータ活用方針を発表した。",
...     "政府はサイバーセキュリティ対策を発表した。",
...     "専門家はAI検索技術の進化に注目している。",
...     "研究者は高速な検索アルゴリズムに注目している。",
...     "オープンソース界隈では全文検索ライブラリに注目している。"
... ]
>>> mfm = MultiFMIndex(data=documents)
>>> mfm
MultiFMIndex(["政府はAI研究の支援を強化すると発表した。", "...
```

#### Count Total Substring Occurrences (count_all)
Counts the total number of occurrences of a pattern across all documents.  
Time complexity does not depend on the length of the data or the number of documents.  

```python
>>> mfm.count_all(pattern="検索")
3
```

#### Count Substring Occurrences per Document (count)
Counts how many times a pattern appears in each document.  
Time complexity does not depend on the length of the data or the number of documents.  

```python
>>> mfm.count(pattern="検索")
{3: 1, 4: 1, 5: 1}
```
This means documents[3], documents[4], and documents[5] each contain the pattern once.

#### Locate Substring Offsets (locate)
Finds the starting positions of a pattern in each document.  
Time complexity does not depend on the length of the data or the number of documents.  

```python
>>> pattern = "検索"
>>> mfm.locate(pattern=pattern)
{5: [13], 4: [7], 3: [6]}
>>> documents[5][13 : 13 + len(pattern)]
'検索'
```
⚠️ The order of results is not guaranteed.

#### List Documents That Start With a Prefix (startswith)
Returns document indices whose content starts with the given prefix.  
Time complexity does not depend on the length of the data or the number of documents.  

```python
>>> prefix = "政府は"
>>> mfm.startswith(prefix=prefix)
[0, 2, 1]
>>> documents[0].startswith(prefix)
True
```

#### List Documents That End With a Suffix (endswith)
Returns document indices whose content ends with the given suffix.  
Time complexity does not depend on the length of the data or the number of documents.  

```python
>>> suffix = "注目している。"
>>> mfm.endswith(suffix=suffix)
[5, 4, 3]
>>> documents[5].endswith(suffix)
True
```

### Running Tests

```bash
$ pip install -e ".[test]"

# Cargo test
$ cargo test --all --release

# Run tests
$ pytest --benchmark-skip

# Run benchmarks
$ pytest --benchmark-only
```

### Formating Code
```bash
$ pip install -e ".[dev]"

# Format Rust code
$ cargo fmt --all
$ cargo clippy --all-targets --all-features

# Format Python code
$ ruff format
```

### Generating Docs
```bash
$ pdoc fm_index \
      --output-directory docs \
      --no-search \
      --no-show-source \
      --docformat markdown \
      --footer-text "© 2025 Koki Watanabe"
```

## References

- P. Ferragina and G. Manzini,  
  Opportunistic data structures with applications,  
  Proceedings 41st Annual Symposium on Foundations of Computer Science,
  Redondo Beach, CA, USA,
  2000,
  pp. 390-398,
  https://doi.org/10.1109/SFCS.2000.892127.  
