# -*- coding: utf-8 -*-
"""
quantlib.alpha.gp_miner —— 遗传规划因子挖掘器
================================================================
把因子表示成【算子树】（叶=价量字段，内部节点=算子），用遗传算法进化公式：
  选择(锦标赛) → 交叉(换子树) → 变异(换节点/窗口) → 保留精英。

适应度 = 训练期 |ICIR| − 简约性惩罚（树越大越扣分）。
防过拟合三件套：
  1. 训练/样本外分离——名人堂(hall of fame)只收"训练强 且 OOS 同号显著"的公式；
  2. 简约性惩罚——压制过拟合的巨树；
  3. 去相关——新公式与名人堂里已有的相关性过高则不收（保证多样/新颖）。

这正是 WorldQuant 式"自动批量产 alpha"的开源缩影。
"""
from __future__ import annotations
import copy
import numpy as np
import pandas as pd
from . import operators as op
from . import factory
from .. import preprocess, evaluate

FIELDS = ["close", "open", "high", "low", "vwap", "volume", "amount", "returns", "adv20"]
WINDOWS = [3, 5, 10, 20, 60]
UNARY_SIMPLE = ["rank", "neg", "abs", "sign", "log"]
UNARY_WIN = ["delta", "mean", "std", "tsrank", "tsmax", "tsmin", "decay"]
BIN_SIMPLE = ["add", "sub", "mul", "div"]
BIN_EW = ["min", "max"]
BIN_WIN = ["corr"]
MAX_DEPTH = 4


class Node:
    __slots__ = ("kind", "name", "window", "children")
    def __init__(self, kind, name=None, window=None, children=None):
        self.kind, self.name, self.window = kind, name, window
        self.children = children or []


# ---------- 计算算子树 → 宽矩阵 ----------
def eval_tree(node, M):
    if node.kind == "term":
        return getattr(M, node.name)
    c = [eval_tree(ch, M) for ch in node.children]
    n, w = node.name, node.window
    if n == "rank":   return op.rank(c[0])
    if n == "neg":    return -c[0]
    if n == "abs":    return c[0].abs()
    if n == "sign":   return np.sign(c[0])
    if n == "log":    return np.log(c[0].abs() + 1)
    if n == "delta":  return op.delta(c[0], w)
    if n == "mean":   return op.ts_mean(c[0], w)
    if n == "std":    return op.ts_std(c[0], w)
    if n == "tsrank": return op.ts_rank(c[0], w)
    if n == "tsmax":  return op.ts_max(c[0], w)
    if n == "tsmin":  return op.ts_min(c[0], w)
    if n == "decay":  return op.decay_linear(c[0], w)
    if n == "add":    return c[0] + c[1]
    if n == "sub":    return c[0] - c[1]
    if n == "mul":    return c[0] * c[1]
    if n == "div":    return (c[0] / c[1].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    if n == "corr":   return op.ts_corr(c[0], c[1], w)
    if n == "min":    return c[0].where(c[0] <= c[1], c[1])
    if n == "max":    return c[0].where(c[0] >= c[1], c[1])
    raise ValueError(n)


def tree_str(node):
    if node.kind == "term": return node.name
    n, w = node.name, node.window
    if node.kind == "unary":
        tag = f"{n}{w}" if w else n
        return f"{tag}({tree_str(node.children[0])})"
    a, b = tree_str(node.children[0]), tree_str(node.children[1])
    if n in BIN_WIN: return f"{n}{w}({a},{b})"
    sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(n)
    return f"({a} {sym} {b})" if sym else f"{n}({a},{b})"


def size(node):
    return 1 if node.kind == "term" else 1 + sum(size(c) for c in node.children)


# ---------- 随机树 / 变异 / 交叉 ----------
def random_tree(rng, depth=0):
    if depth >= MAX_DEPTH or (depth > 0 and rng.random() < 0.35):
        return Node("term", name=rng.choice(FIELDS))
    r = rng.random()
    if r < 0.45:                                   # 一元
        if rng.random() < 0.6:
            nm = rng.choice(UNARY_WIN); w = int(rng.choice(WINDOWS))
        else:
            nm = rng.choice(UNARY_SIMPLE); w = None
        return Node("unary", name=nm, window=w, children=[random_tree(rng, depth + 1)])
    else:                                          # 二元
        rr = rng.random()
        if rr < 0.5:   nm, w = rng.choice(BIN_SIMPLE), None
        elif rr < 0.7: nm, w = rng.choice(BIN_EW), None
        else:          nm, w = rng.choice(BIN_WIN), int(rng.choice(WINDOWS))
        return Node("binary", name=nm, window=w,
                    children=[random_tree(rng, depth + 1), random_tree(rng, depth + 1)])


def _all_nodes(node, acc):
    acc.append(node)
    for c in node.children: _all_nodes(c, acc)
    return acc


def mutate(tree, rng):
    t = copy.deepcopy(tree)
    nodes = _all_nodes(t, [])
    tgt = rng.choice(nodes)
    if tgt.kind == "term":
        tgt.name = rng.choice(FIELDS)
    elif tgt.window is not None and rng.random() < 0.5:
        tgt.window = int(rng.choice(WINDOWS))           # 微调窗口
    else:                                                # 换整棵子树
        new = random_tree(rng, MAX_DEPTH - 1)
        tgt.kind, tgt.name, tgt.window, tgt.children = new.kind, new.name, new.window, new.children
    return t


def crossover(a, b, rng):
    t = copy.deepcopy(a)
    nodes = _all_nodes(t, [])
    tgt = rng.choice(nodes)
    donor = copy.deepcopy(rng.choice(_all_nodes(b, [])))
    tgt.kind, tgt.name, tgt.window, tgt.children = donor.kind, donor.name, donor.window, donor.children
    return t


# ---------- 适应度 ----------
def factor_icir(tree, M, panel, do_neutralize=False):
    try:
        mat = eval_tree(tree, M)
        raw = factory.sample_to_panel(mat, panel)
        if raw.notna().sum() < 3000:
            return np.nan, np.nan
        f = preprocess.preprocess_factor(panel, raw, do_neutralize=do_neutralize)
        ic = evaluate.compute_ic(panel, f).dropna()
        if len(ic) < 12: return np.nan, np.nan
        return ic.mean() / ic.std(), ic.mean()
    except Exception:
        return np.nan, np.nan


def _factor_values(tree, M, panel):
    raw = factory.sample_to_panel(eval_tree(tree, M), panel)
    return preprocess.preprocess_factor(panel, raw, do_neutralize=False)


def evolve(M, train_panel, oos_panel, pop_size=60, generations=10,
           parsimony=0.002, hof_size=15, max_corr=0.7, seed=42, log=print):
    """进化主循环。返回名人堂 [(formula_str, train_ICIR, oos_ICIR, size)]。"""
    rng = np.random.RandomState(seed)
    pop = [random_tree(rng) for _ in range(pop_size)]

    def fitness(t):
        ic, _ = factor_icir(t, M, train_panel)
        if not np.isfinite(ic): return -1e9
        return abs(ic) - parsimony * size(t)

    fits = [fitness(t) for t in pop]
    for g in range(generations):
        new = []
        # 精英保留
        elite = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)[:max(2, pop_size // 10)]
        new.extend(copy.deepcopy(pop[i]) for i in elite)
        while len(new) < pop_size:
            # 锦标赛选择
            def pick():
                cand = rng.randint(0, len(pop), 4)
                return pop[max(cand, key=lambda i: fits[i])]
            if rng.random() < 0.7:
                child = crossover(pick(), pick(), rng)
            else:
                child = mutate(pick(), rng)
            if rng.random() < 0.3:
                child = mutate(child, rng)
            new.append(child)
        pop = new
        fits = [fitness(t) for t in pop]
        best = max(fits)
        log(f"  gen {g+1}/{generations}  best|ICIR|-pen={best:.3f}")

    # 组装名人堂：训练强 → 去相关 → 记录 OOS
    order = sorted(range(len(pop)), key=lambda i: fits[i], reverse=True)
    hof, hof_vals = [], []
    for i in order:
        if fits[i] < -1e8: continue
        t = pop[i]
        tr_icir, _ = factor_icir(t, M, train_panel)
        if not np.isfinite(tr_icir) or abs(tr_icir) < 0.2: continue
        try:
            vals = _factor_values(t, M, train_panel).values
        except Exception:
            continue
        # 与已入选去相关
        s = pd.Series(vals)
        too_corr = any(abs(np.corrcoef(np.nan_to_num(vals), np.nan_to_num(hv))[0, 1]) > max_corr
                       for hv in hof_vals)
        if too_corr: continue
        oos_icir, _ = factor_icir(t, M, oos_panel)
        hof.append((tree_str(t), round(tr_icir, 3), round(oos_icir, 3), size(t)))
        hof_vals.append(vals)
        if len(hof) >= hof_size: break
    return hof
