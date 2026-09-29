"""Простой трекинг экспериментов: каждая строка experiments/runs.csv - один прогон обучения."""
import subprocess
from pathlib import Path

import pandas as pd

from customers_prediction.config import PROJECT_ROOT, RUNS_PATH


def git_commit() -> str:
    """Короткий хэш текущего коммита; "+изменения", если в рабочей копии есть незакоммиченные правки."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+изменения" if dirty else commit


def log_run(record: dict, path: Path = RUNS_PATH) -> pd.DataFrame:
    """Дописывает прогон в таблицу и возвращает всю таблицу. Новые поля добавляются как новые колонки."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run = pd.DataFrame([{"run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), "git_commit": git_commit(),
                         **record}])
    runs = pd.concat([pd.read_csv(path), run], ignore_index=True) if path.exists() else run
    runs.to_csv(path, index=False)
    return runs
