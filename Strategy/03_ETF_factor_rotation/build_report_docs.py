"""把 报告_最终版.md 排版成自包含的 Word(.docx) 和 PDF（图片内嵌，单文件可直接发送）。
- 预处理 emoji/罕见符号以兼容 xelatex
- 抽出标题做成 YAML 题头（更干净的标题页/题头 + 目录）
"""
import subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
src = ROOT / "报告_最终版.md"
tmp = ROOT / "_report_build_tmp.md"
pdf = ROOT / "报告_最终版.pdf"
docx = ROOT / "报告_最终版.docx"
DATE = "2026-06-17"

raw = src.read_text(encoding="utf-8").splitlines()
# 抽掉首个 H1 标题行（移到 YAML），其余照旧
title = "全市场 ETF 多因子月度轮动策略 —— 最终研究报告"
body = []
dropped = False
for ln in raw:
    if not dropped and ln.startswith("# "):
        title = ln[2:].strip()
        dropped = True
        continue
    body.append(ln)
text = "\n".join(body)

repl = {
    "✅": "【通过】", "⚠️": "【注意】", "⚠": "【注意】", "️": "",
    "❌": "【否】", "√": "根号", "₆": "6", "₀": "0", "①": "(1)", "②": "(2)",
    "ᵢ": "i", "wᵢ": "w_i", "factorᵢ": "factor_i",
}
for a, b in repl.items():
    text = text.replace(a, b)

yaml = f'---\ntitle: "{title}"\ndate: "{DATE}"\n---\n\n'
tmp.write_text(yaml + text, encoding="utf-8")

common = ["--toc", "-V", "toc-title=目录", "--resource-path", str(ROOT)]

# 1) Word
docx_cmd = ["pandoc", str(tmp), "-o", str(docx)] + common
r1 = subprocess.run(docx_cmd, cwd=ROOT, capture_output=True, text=True)

# 2) PDF
pdf_cmd = ["pandoc", str(tmp), "-o", str(pdf), "--pdf-engine=xelatex",
           "-V", "CJKmainfont=Songti SC", "-V", "mainfont=Arial Unicode MS",
           "-V", "monofont=Arial Unicode MS", "-V", "geometry:margin=2cm",
           "-V", "fontsize=11pt", "-V", "colorlinks=true", "-V", "linkcolor=blue"] + common
r2 = subprocess.run(pdf_cmd, cwd=ROOT, capture_output=True, text=True)

tmp.unlink(missing_ok=True)
for tag, r, f in [("DOCX", r1, docx), ("PDF", r2, pdf)]:
    err = (r.stderr or "")
    bad = [l for l in err.splitlines() if "Missing character" not in l and l.strip()]
    print(f"=== {tag} exit {r.returncode} | {'OK ' + str(f.stat().st_size) + 'B' if f.exists() else 'FAILED'}")
    if r.returncode != 0:
        print("\n".join(bad[-8:]))
