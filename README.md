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


High-performance FM-index implementation powered by Rust,  
designed for fast substring search on large texts and collections  

- PyPI: https://pypi.org/project/fm-index
- Document: https://math-hiyoko.github.io/fm-index
- Repository: https://github.com/math-hiyoko/fm-index

## Features:
- Fast count / locate substring queries
- Data-parallel optimizations across index construction and queries
- Supports single text and multiple documents

## Installation
```bash
pip install fm-index
```

## FMIndex (Single Document)
### What is FMIndex?
FMIndex builds a compressed index over a single string,  
allowing fast substring search without scanning the original data.  

### Construction Complexity
- Time / Space: `O(|data| log σ)`  
- σ = number of unique characters in the input　　

#### Example

```python
from fm_index import FMIndex

genome = "ACGTACGTTGACCTGACTGACTGACTGACGATCGATCGATCGATCGATCG"
fm = FMIndex(data=genome)
```

### Count Substring Occurrences
Counts how many times a pattern appears.  
Time complexity is independent of data size.  

```python
fm.count(pattern="GACTGACT")
# 2
```

### Locate Substring Positions
Returns all starting offsets where the pattern occurs.  

To improve throughput for high-frequency patterns,  
FMIndex applies parallel execution to parts of the locate pipeline.

```python
fm.locate(pattern="GACTGACT")
# [18, 14]
```

### Iterative Locate (Streaming)
For large result sets, iter_locate provides a memory-efficient  
iterator interface that yields positions lazily.

```python
for pos in fm.iter_locate(pattern="GACTGACT"):
    print(pos)
# 18
# 14
```

- Same results as locate
- Does not allocate a result list
- Suitable for streaming and early termination

## MultiFMIndex (Multiple Documents)
MultiFMIndex extends FMIndex to support multiple documents  
while keeping query time independent of corpus size  

Query processing is internally parallelized where possible,  
making multi-document search efficient in practice.  

### Construction Complexity
- Time / Space: `O(|''.join(data)| log σ)`  
- σ = number of unique characters in the input   

```python
from fm_index import MultiFMIndex

documents = [
    "政府はAI研究の支援を強化すると発表した。",
    "政府は新たなデータ活用方針を発表した。",
    "政府はサイバーセキュリティ対策を発表した。",
    "専門家はAI検索技術の進化に注目している。",
    "研究者は高速な検索アルゴリズムに注目している。",
    "オープンソース界隈では全文検索ライブラリに注目している。"
]

mfm = MultiFMIndex(data=documents)
```

### Count Across All Documents
```python
mfm.count_all(pattern="検索")
# 3
```

### Count Per Document
```python
mfm.count(pattern="検索")
# {3: 1, 4: 1, 5: 1}
```

### Locate Per Document
```python
mfm.locate(pattern="検索")
# {5: [13], 4: [7], 3: [6]}
```

### Iterative Locate (Streaming)
```python
for doc_id, pos in mfm.iter_locate(pattern="検索"):
    print(doc_id, pos)
# 4 7
# 5 13
# 3 6
```

### Prefix / Suffix Search
```python
mfm.startswith(prefix="政府は")
mfm.endswith(suffix="注目している。")
```

## Development & Testing

### Run Tests

```bash
pip install -e ".[test]"
cargo test --all --release
pytest
```

## Formating
```bash
pip install -e ".[dev]"
cargo fmt --all
cargo clippy --all-targets --all-features
ruff format
```

## Generating Docs
```bash
pdoc fm_index \
      --output-directory docs \
      --no-search \
      --no-show-source \
      --docformat markdown \
      --footer-text "© 2026 Koki Watanabe"
```

## References

- P. Ferragina and G. Manzini,  
  Opportunistic data structures with applications,  
  Proceedings 41st Annual Symposium on Foundations of Computer Science,  
  Redondo Beach, CA, USA,  
  2000,  
  pp. 390-398,  
  https://doi.org/10.1109/SFCS.2000.892127.  
