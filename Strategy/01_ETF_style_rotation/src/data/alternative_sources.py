"""Non-Choice data source adapters.

The public-source layer intentionally emits the same canonical raw schemas as
``src.data.loaders`` expects:

``stock_daily``: [date, code, close_adj, total_mv, float_mv, turnover, amount]
``macro_<key>``: [date, value]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import time

import pandas as pd

from src.data.cache import atomic_to_parquet, parquet_covers, readable_parquet
from src.utils.config import PROJECT_ROOT

LOG = logging.getLogger(__name__)
RAW = PROJECT_ROOT / "data" / "raw"


@dataclass(frozen=True)
class FetchResult:
    name: str
    path: Path
    fetched: bool
    rows: int
    message: str = ""


def _compact_a_code(code: str) -> str:
    return str(code).split(".")[0]


def _choice_suffix_code(symbol: str) -> str:
    prefix = symbol[:3]
    if prefix in {"600", "601", "603", "605", "688", "689"}:
        return f"{symbol}.SH"
    if prefix in {"000", "001", "002", "003", "300", "301"}:
        return f"{symbol}.SZ"
    if prefix in {"430", "831", "832", "833", "834", "835", "836", "837", "838", "839", "870", "871", "872", "873", "920"}:
        return f"{symbol}.BJ"
    return symbol


def fetch_stock_daily_akshare(
    code: str,
    start: str,
    end: str,
    part_dir: Path,
    force: bool = False,
    sleep: float = 0.5,
) -> FetchResult:
    """Fetch one A-share stock from AkShare and cache a canonical parquet part.

    AkShare Eastmoney historical K-line provides adjusted prices, turnover and
    amount.  ``stock_value_em`` provides historical total/free-float market cap
    from 2018 onward, which is enough for the current 2024-2026 gap but not for
    a full 2014 reconstruction.
    """
    import akshare as ak

    symbol = _compact_a_code(code)
    canonical_code = code if "." in str(code) else _choice_suffix_code(symbol)
    part = part_dir / f"{canonical_code.replace('.', '_')}.parquet"
    if not force and parquet_covers(part, start, end, codes={canonical_code}, code_col="code"):
        df = pd.read_parquet(part)
        return FetchResult(canonical_code, part, fetched=False, rows=len(df), message="cache hit")

    start_ak = pd.Timestamp(start).strftime("%Y%m%d")
    end_ak = pd.Timestamp(end).strftime("%Y%m%d")
    hist = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=start_ak,
        end_date=end_ak,
        adjust="hfq",
    )
    if hist.empty:
        return FetchResult(canonical_code, part, fetched=False, rows=0, message="empty history")

    value = ak.stock_value_em(symbol=symbol)
    hist = hist.rename(
        columns={
            "日期": "date",
            "收盘": "close_adj",
            "换手率": "turnover",
            "成交额": "amount",
        }
    )
    out = hist[["date", "close_adj", "turnover", "amount"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["code"] = canonical_code

    if not value.empty:
        value = value.rename(
            columns={
                "数据日期": "date",
                "总市值": "total_mv",
                "流通市值": "float_mv",
            }
        )
        value["date"] = pd.to_datetime(value["date"])
        out = out.merge(value[["date", "total_mv", "float_mv"]], on="date", how="left")
    else:
        out["total_mv"] = pd.NA
        out["float_mv"] = pd.NA

    out = out[["date", "code", "close_adj", "total_mv", "float_mv", "turnover", "amount"]]
    out = out.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    atomic_to_parquet(out, part)
    time.sleep(sleep)
    return FetchResult(canonical_code, part, fetched=True, rows=len(out))


def combine_stock_parts(part_dir: Path, output: Path) -> FetchResult:
    frames = []
    bad = []
    for part in sorted(part_dir.glob("*.parquet")):
        if not readable_parquet(part):
            bad.append(part.name)
            continue
        frames.append(pd.read_parquet(part))
    if not frames:
        return FetchResult("stock_daily", output, fetched=False, rows=0, message="no readable parts")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["close_adj"]).drop_duplicates(["date", "code"])
    df = df.sort_values(["date", "code"])
    atomic_to_parquet(df, output)
    msg = f"combined {len(frames)} parts"
    if bad:
        msg += f"; skipped unreadable parts: {bad[:5]}"
    return FetchResult("stock_daily", output, fetched=True, rows=len(df), message=msg)


def _filter_dates(df: pd.DataFrame, date_col: str, value_col: str, start: str, end: str) -> pd.DataFrame:
    out = df[[date_col, value_col]].rename(columns={date_col: "date", value_col: "value"}).copy()
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))]
    return out.dropna(subset=["value"]).drop_duplicates("date").sort_values("date")


def fetch_macro_akshare(key: str, start: str, end: str, force: bool = False) -> FetchResult:
    """Fetch supported macro indicators from public AkShare sources."""
    import akshare as ak

    output = RAW / f"macro_{key}.parquet"
    if not force and parquet_covers(output, start, end):
        df = pd.read_parquet(output)
        return FetchResult(key, output, fetched=False, rows=len(df), message="cache hit")

    if key == "SHIBOR_1周":
        df = ak.macro_china_shibor_all()
        out = _filter_dates(df, "日期", "1W-定价", start, end)
    elif key == "SHIBOR_1年":
        df = ak.macro_china_shibor_all()
        out = _filter_dates(df, "日期", "1Y-定价", start, end)
    elif key == "DR007":
        # FDR007 is the deposit-institution repo fixing rate.  It is not the
        # volume-weighted DR007 series, but is the closest free ChinaMoney
        # daily series exposed by AkShare.
        frames = []
        for s, e in _month_segments(start, end):
            frames.append(ak.repo_rate_hist(s, e))
            time.sleep(0.2)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        out = _filter_dates(df, "date", "FDR007", start, end)
    elif key == "中债新综合指数":
        df = ak.bond_new_composite_index_cbond(indicator="财富", period="总值")
        date_col, value_col = _pick_date_value_columns(df)
        out = _filter_dates(df, date_col, value_col, start, end)
    elif key == "国债到期收益率_1年":
        out = _fetch_china_bond_yield(ak, start, end, "1年")
    elif key == "国债到期收益率_10年":
        out = _fetch_china_bond_yield(ak, start, end, "10年")
    elif key == "美元兑人民币_中间价":
        try:
            df = ak.macro_china_rmb()
            date_col, value_col = _pick_fx_columns(df)
            out = _filter_dates(df, date_col, value_col, start, end)
        except Exception as exc:  # noqa: BLE001 - try the weaker public fallback
            LOG.warning("macro_china_rmb failed, fallback to currency_boc_sina: %s", exc)
            df = ak.currency_boc_sina(
                symbol="美元",
                start_date=pd.Timestamp(start).strftime("%Y%m%d"),
                end_date=pd.Timestamp(end).strftime("%Y%m%d"),
            )
            out = _filter_dates(df, "日期", "央行中间价", start, end)
    else:
        raise KeyError(f"暂未配置 AkShare 宏观替代源: {key}")

    if out.empty:
        return FetchResult(key, output, fetched=False, rows=0, message="empty result")
    atomic_to_parquet(out, output)
    return FetchResult(key, output, fetched=True, rows=len(out))


def _month_segments(start: str, end: str) -> list[tuple[str, str]]:
    starts = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="MS")
    if starts.empty or starts[0] > pd.Timestamp(start):
        starts = starts.insert(0, pd.Timestamp(start))
    segs = []
    for s in starts:
        e = min(s + pd.offsets.MonthEnd(0), pd.Timestamp(end))
        if s <= e:
            segs.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
    return segs


def _year_segments(start: str, end: str) -> list[tuple[str, str]]:
    out = []
    for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
        s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01"))
        e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31"))
        out.append((s.strftime("%Y%m%d"), e.strftime("%Y%m%d")))
    return out


def _fetch_china_bond_yield(ak, start: str, end: str, tenor: str) -> pd.DataFrame:
    frames = []
    for s, e in _year_segments(start, end):
        frames.append(ak.bond_china_yield(s, e))
        time.sleep(0.2)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["date", "value"])
    date_col = next((c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"), df.columns[0])
    candidates = [c for c in df.columns if tenor in str(c) and ("国债" in str(c) or "Government" in str(c))]
    if not candidates:
        candidates = [c for c in df.columns if tenor in str(c)]
    if not candidates:
        raise KeyError(f"bond_china_yield 返回列中未找到期限 {tenor}: {list(df.columns)}")
    return _filter_dates(df, date_col, candidates[0], start, end)


def _pick_date_value_columns(df: pd.DataFrame) -> tuple[str, str]:
    date_col = next((c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"), df.columns[0])
    numeric_cols = [c for c in df.columns if c != date_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        numeric_cols = [c for c in df.columns if c != date_col]
    return date_col, numeric_cols[0]


def _pick_fx_columns(df: pd.DataFrame) -> tuple[str, str]:
    date_col = next((c for c in df.columns if "日期" in str(c) or str(c).lower() == "date"), df.columns[0])
    preferred = [c for c in df.columns if "美元" in str(c) and ("中间价" in str(c) or "今值" in str(c))]
    if preferred:
        return date_col, preferred[0]
    return _pick_date_value_columns(df)
