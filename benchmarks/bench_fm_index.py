from typing import Literal

import numpy as np
import pytest

from fm_index import FMIndex


@pytest.fixture
def random_data(size: int, ucs: Literal["ucs1", "ucs2", "ucs4"]) -> str:
    rng = np.random.default_rng(42)
    if ucs == "ucs1":
        codepoints = rng.integers(0x20, 0x80, size=size, dtype=np.uint8)
    elif ucs == "ucs2":
        codepoints = rng.integers(0x20, 0xD800, size=size, dtype=np.uint16)
    elif ucs == "ucs4":
        n1 = size // 2
        n2 = size - n1
        a = rng.integers(0x20, 0xD800, size=n1, dtype=np.uint32)
        b = rng.integers(0xE000, 0x110000, size=n2, dtype=np.uint32)
        codepoints = np.concatenate([a, b])
        rng.shuffle(codepoints)
    else:
        raise ValueError(f"Unknown ucs: {ucs}")
    return "".join(map(chr, codepoints))


@pytest.fixture
def random_fm_index(random_data: str) -> FMIndex:
    return FMIndex(random_data)


@pytest.mark.parametrize("size", [5000, 50000, 500000])
@pytest.mark.parametrize("ucs", ["ucs1", "ucs2", "ucs4"])
class BenchFMIndex:
    def bench_construction(self, benchmark, random_data):
        benchmark(FMIndex, random_data)

    def bench_item(self, benchmark, random_fm_index):
        benchmark(random_fm_index.item)

    def bench_contains(self, benchmark, random_fm_index, random_data):
        pattern = random_data[:5]
        benchmark(random_fm_index.contains, pattern)

    def bench_count(self, benchmark, random_fm_index, random_data):
        pattern = random_data[:5]
        benchmark(random_fm_index.count, pattern)

    def bench_locate(self, benchmark, random_fm_index, random_data):
        pattern = random_data[:5]
        benchmark(random_fm_index.locate, pattern)

    def bench_iter_locate(self, benchmark, random_fm_index, random_data):
        pattern = random_data[:5]
        benchmark(random_fm_index.iter_locate, pattern)

    def bench_startswith(self, benchmark, random_fm_index, random_data):
        prefix = random_data[:50]
        benchmark(random_fm_index.startswith, prefix)

    def bench_endswith(self, benchmark, random_fm_index, random_data):
        suffix = random_data[-50:]
        benchmark(random_fm_index.endswith, suffix)
