"""Fetch non-Choice replacement data with resumable local caches.

Examples:
  python scripts/06_fetch_alt_data.py macro --start 2024-01-01 --end 2026-04-17
  python scripts/06_fetch_alt_data.py stock --codes 000001.SZ 600000.SH --start 2024-01-01 --end 2026-04-17
  python scripts/06_fetch_alt_data.py stock --universe data/manual/needed_codes.txt --start 2024-01-01 --end 2026-04-17
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.alternative_sources import (  # noqa: E402
    RAW,
    combine_stock_parts,
    fetch_macro_akshare,
    fetch_stock_daily_akshare,
)
from src.data.cache import parquet_status  # noqa: E402
from src.utils.config import load_yaml  # noqa: E402


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _codes_from_args(args: argparse.Namespace) -> list[str]:
    codes = list(args.codes or [])
    if args.universe:
        p = Path(args.universe)
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            codes.extend(df["code"].dropna().astype(str).tolist())
        else:
            codes.extend([line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()])
    return sorted(set(codes))


def run_stock(args: argparse.Namespace) -> int:
    codes = _codes_from_args(args)
    if not codes:
        raise SystemExit("stock 模式需要 --codes 或 --universe")
    part_dir = RAW / "stock_daily_akshare_parts"
    ok = failed = skipped = 0
    for code in codes:
        try:
            res = fetch_stock_daily_akshare(
                code=code,
                start=args.start,
                end=args.end,
                part_dir=part_dir,
                force=args.force,
                sleep=args.sleep,
            )
            if res.fetched:
                ok += 1
                logging.info("fetched %s rows=%s", res.name, res.rows)
            else:
                skipped += 1
                logging.info("skip %s %s", res.name, res.message)
        except Exception as exc:  # noqa: BLE001 - continue resumable batch
            failed += 1
            logging.exception("failed %s: %s", code, exc)
    combined = combine_stock_parts(part_dir, RAW / "stock_daily.parquet")
    logging.info("combine stock_daily rows=%s %s", combined.rows, combined.message)
    logging.info("stock done: fetched=%s skipped=%s failed=%s", ok, skipped, failed)
    return 1 if failed else 0


def run_macro(args: argparse.Namespace) -> int:
    cfg = load_yaml("macro_indicators")
    keys = args.indicators
    if not keys:
        keys = [
            ind["name"]
            for spec in cfg["categories"].values()
            for ind in spec["indicators"]
            if ind.get("enabled", True)
        ]
    failed = 0
    for key in keys:
        try:
            res = fetch_macro_akshare(key=key, start=args.start, end=args.end, force=args.force)
            logging.info("%s rows=%s fetched=%s %s", key, res.rows, res.fetched, res.message)
        except KeyError as exc:
            failed += 1
            logging.warning("%s", exc)
        except Exception as exc:  # noqa: BLE001 - keep other indicators running
            failed += 1
            logging.exception("failed %s: %s", key, exc)
    return 1 if failed else 0


def run_check(args: argparse.Namespace) -> int:
    paths = sorted((RAW).glob("*.parquet"))
    paths += sorted((RAW / "stock_daily_parts").glob("*.parquet"))
    paths += sorted((RAW / "etf_daily_parts").glob("*.parquet"))
    paths += sorted((RAW / "index_daily_parts").glob("*.parquet"))
    bad = 0
    for p in paths:
        status = parquet_status(p)
        if status.ok:
            logging.info("OK  %s rows=%s", p.relative_to(RAW.parent.parent), status.rows)
        else:
            bad += 1
            logging.warning("BAD %s %s", p.relative_to(RAW.parent.parent), status.error)
    return 1 if bad else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="cmd", required=True)

    stock = sub.add_parser("stock", help="fetch A-share stock daily data from AkShare")
    stock.add_argument("--start", required=True)
    stock.add_argument("--end", required=True)
    stock.add_argument("--codes", nargs="*")
    stock.add_argument("--universe")
    stock.add_argument("--force", action="store_true")
    stock.add_argument("--sleep", type=float, default=0.5)
    stock.set_defaults(func=run_stock)

    macro = sub.add_parser("macro", help="fetch supported macro indicators from AkShare")
    macro.add_argument("--start", required=True)
    macro.add_argument("--end", required=True)
    macro.add_argument("--indicators", nargs="*")
    macro.add_argument("--force", action="store_true")
    macro.set_defaults(func=run_macro)

    check = sub.add_parser("check-cache", help="verify parquet caches are readable")
    check.set_defaults(func=run_check)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format=LOG_FORMAT)
    raise SystemExit(args.func(args))
