"""parquet 本地缓存读写工具。"""
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

from .config import PROJECT_ROOT


def data_path(layer: str, name: str) -> Path:
    """layer in {raw, interim, processed}"""
    p = PROJECT_ROOT / "data" / layer / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, layer: str, name: str) -> Path:
    p = data_path(layer, name)
    df.to_parquet(p)
    return p


def load_parquet(layer: str, name: str) -> pd.DataFrame:
    p = data_path(layer, name)
    if not p.exists():
        raise FileNotFoundError(
            f"缺少数据文件 {p}。请先运行 notebooks/01_choice_data_fetch.ipynb 取数。"
        )
    return pd.read_parquet(p)


def exists(layer: str, name: str) -> bool:
    p = data_path(layer, name)
    if not p.exists():
        return False
    try:
        first_col = pq.ParquetFile(p).schema.names[0]
        pd.read_parquet(p, columns=[first_col])
        return True
    except Exception:
        return False
