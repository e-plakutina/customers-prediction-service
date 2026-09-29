"""Пути проекта, даты разбиения и общие константы."""
import os
from pathlib import Path

import pandas as pd

# корень проекта: по умолчанию - папка репозитория; в Docker пакет установлен в site-packages,
# поэтому корень задается переменной окружения PROJECT_ROOT
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "guest_forecaster.joblib"
RUNS_PATH = PROJECT_ROOT / "experiments" / "runs.csv"

TEST_START = pd.Timestamp("2015-06-20")         # начало отложенного периода (раздел 2 ноутбука 01)
TRAIN_END = TEST_START - pd.Timedelta(days=1)
HOLDOUT_END = pd.Timestamp("2015-07-31")        # последний день с известным числом покупателей
N_LONG = 3                                       # порог длинного закрытия, дней (раздел 4 ноутбука 01)
HORIZON = 7                                      # горизонт прогноза: признаки для дня t - из данных не позже t - 7
SEED = 42
