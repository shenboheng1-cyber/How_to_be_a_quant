"""配置加载工具。"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(name: str) -> dict:
    """按文件名加载 config/ 下的 yaml, 例如 load_yaml('strategy')。"""
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all() -> dict:
    return {
        "strategy": load_yaml("strategy"),
        "macro": load_yaml("macro_indicators"),
        "universe": load_yaml("index_universe"),
        "barra": load_yaml("barra_factors"),
    }
