"""Cache health checks for local parquet data.

The data fetch notebook used file existence as the cache condition.  That is
not enough here because interrupted or incompatible parquet files can exist
but fail during decoding.  Fetchers should use these helpers before skipping
work.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import pandas as pd


@dataclass(frozen=True)
class CacheStatus:
    path: Path
    exists: bool
    readable: bool
    rows: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exists and self.readable


def parquet_status(path: Path, columns: list[str] | None = None) -> CacheStatus:
    """Return whether a parquet file exists and can be decoded."""
    if not path.exists():
        return CacheStatus(path=path, exists=False, readable=False)
    try:
        df = pd.read_parquet(path, columns=columns)
        return CacheStatus(path=path, exists=True, readable=True, rows=len(df))
    except Exception as exc:  # noqa: BLE001 - report exact cache failure
        return CacheStatus(
            path=path,
            exists=True,
            readable=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def readable_parquet(path: Path, columns: list[str] | None = None) -> bool:
    return parquet_status(path, columns=columns).ok


def parquet_covers(
    path: Path,
    start: str,
    end: str,
    date_col: str = "date",
    code_col: str | None = None,
    codes: set[str] | None = None,
) -> bool:
    """Check that a parquet cache is readable and covers a requested range."""
    if not path.exists():
        return False
    cols = [date_col]
    if code_col:
        cols.append(code_col)
    try:
        df = pd.read_parquet(path, columns=cols)
    except Exception:
        return False
    if df.empty or date_col not in df.columns:
        return False
    dates = pd.to_datetime(df[date_col])
    if dates.min() > pd.Timestamp(start) or dates.max() < pd.Timestamp(end):
        return False
    if codes and code_col:
        have = set(df[code_col].dropna().astype(str))
        return codes.issubset(have)
    return True


def atomic_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet through a temporary file, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
