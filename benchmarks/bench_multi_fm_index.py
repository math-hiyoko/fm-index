from typing import Literal

import numpy as np
import pytest

from fm_index import MultiFMIndex


@pytest.fixture
def random_data(size: tuple[int, int], ucs: Literal["ucs1", "ucs2", "ucs4"]) -> str:
    rng = np.random.default_rng(42)
    if ucs == "ucs1":
        codepoints = rng.integers(0x20, 0x80, size=size, dtype=np.uint8)
    elif ucs == "ucs2":
        codepoints = rng.integers(0x20, 0xD800, size=size, dtype=np.uint16)
    elif ucs == "ucs4":
        codepoints = rng.integers(0x20, 0x110000, size=size, dtype=np.uint32)
    else:
        raise ValueError(f"Unknown ucs: {ucs}")
    return ["".join(map(chr, codepoints)) for codepoints in codepoints]


@pytest.fixture
def random_multi_fm_index(random_data: str) -> MultiFMIndex:
    return MultiFMIndex(random_data)


@pytest.mark.parametrize("size", [(100, 100), (1000, 100), (100, 1000)])
@pytest.mark.parametrize("ucs", ["ucs1", "ucs2", "ucs4"])
class BenchMultiFMIndex:
    def bench_multi_construction(self, benchmark, random_data):
        benchmark(MultiFMIndex, random_data)

    def bench_multi_item(self, benchmark, random_multi_fm_index):
        benchmark(random_multi_fm_index.item)

    def bench_multi_contains(self, benchmark, random_multi_fm_index):
        pattern = random_multi_fm_index.item()[0]
        benchmark(random_multi_fm_index.contains, pattern)

    def bench_multi_count_all(self, benchmark, random_multi_fm_index):
        pattern = random_multi_fm_index.item()[0][:50]
        benchmark(random_multi_fm_index.count_all, pattern)

    def bench_multi_count(self, benchmark, random_multi_fm_index):
        pattern = random_multi_fm_index.item()[0][:50]
        benchmark(random_multi_fm_index.count, pattern)

    def bench_multi_locate(self, benchmark, random_multi_fm_index):
        pattern = random_multi_fm_index.item()[0][:50]
        benchmark(random_multi_fm_index.locate, pattern)

    def bench_multi_startswith(self, benchmark, random_multi_fm_index):
        prefix = random_multi_fm_index.item()[0][:50]
        benchmark(random_multi_fm_index.startswith, prefix)

    def bench_multi_endswith(self, benchmark, random_multi_fm_index):
        suffix = random_multi_fm_index.item()[0][-50:]
        benchmark(random_multi_fm_index.endswith, suffix)
