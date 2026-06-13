"""23个描述变量 -> 10个合成风格因子 (按 config/barra_factors.yaml composites 权重)。"""
from __future__ import annotations

import pandas as pd


def synthesize(descriptor_panel: pd.DataFrame, composites_cfg: dict) -> pd.DataFrame:
    """descriptor_panel: index=股票, columns=23个描述变量(已标准化)。
    返回 index=股票, columns=10个合成因子。缺失描述变量按剩余权重重归一。"""
    out = {}
    for name, spec in composites_cfg.items():
        weights = spec["weights"]
        avail = {d: w for d, w in weights.items() if d in descriptor_panel.columns}
        if not avail:
            continue
        sub = descriptor_panel[list(avail.keys())]
        w = pd.Series(avail)
        # 对每只股票, 按非缺失描述变量的权重重归一
        wmat = sub.notna().astype(float).mul(w, axis=1)
        wsum = wmat.sum(axis=1).replace(0, pd.NA)
        out[name] = sub.mul(w, axis=1).sum(axis=1) / wsum
    return pd.DataFrame(out)
