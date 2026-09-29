"""Получение исходного датасета Rossmann и сборка data/processed для магазинов выборки.

Запуск: python make_dataset.py

1. Если в data/raw нет train.csv, test.csv и store.csv, скрипт пробует скачать их с Kaggle через kagglehub.
   Если скачать не получилось (нет сети или авторизации Kaggle), печатает инструкцию для ручного скачивания.
2. Проверяет сырые файлы: колонки и число строк.
3. Фильтрует их по списку магазинов data/processed/selected_stores.csv (его сохраняет ноутбук 01)
   и сохраняет train_subset.csv, store_subset.csv и plan_subset.csv.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

from customers_prediction.config import PROCESSED_DIR, RAW_DIR
from customers_prediction.data import build_processed

KAGGLE_DATASET = "pratyushakar/rossmann-store-sales"
KAGGLE_URL = f"https://www.kaggle.com/datasets/{KAGGLE_DATASET}"
RAW_FILES = {   # файл: (обязательные колонки, число строк в оригинальном датасете)
    "train.csv": (["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open", "Promo", "StateHoliday",
                   "SchoolHoliday"], 1_017_209),
    "test.csv": (["Id", "Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"], 41_088),
    "store.csv": (["Store", "StoreType", "Assortment", "CompetitionDistance", "CompetitionOpenSinceMonth",
                   "CompetitionOpenSinceYear", "Promo2", "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval"], 1_115),
}


def download_raw(raw_dir: Path) -> None:
    """Скачивает датасет с Kaggle и копирует нужные файлы в raw_dir."""
    try:
        import kagglehub    # группа зависимостей data: uv sync --group data (или uv sync --all-groups)
    except ImportError as error:
        raise ImportError("не установлен kagglehub: uv sync --group data или pip install kagglehub") from error

    downloaded = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in RAW_FILES:
        matches = list(downloaded.rglob(name))
        if not matches:
            raise FileNotFoundError(f"В скачанном датасете нет файла {name}")
        shutil.copy(matches[0], raw_dir / name)


def check_raw(raw_dir: Path) -> list[str]:
    """Проблемы с сырыми файлами; пустой список - все в порядке."""
    problems = []
    for name, (columns, n_rows) in RAW_FILES.items():
        path = raw_dir / name
        if not path.exists():
            problems.append(f"нет файла {path}")
            continue
        frame = pd.read_csv(path, low_memory=False)
        missing = sorted(set(columns) - set(frame.columns))
        if missing:
            problems.append(f"{name}: нет колонок {missing}")
        if len(frame) != n_rows:
            problems.append(f"{name}: {len(frame)} строк, в оригинальном датасете {n_rows}")
    return problems


def main() -> int:
    if any(not (RAW_DIR / name).exists() for name in RAW_FILES):
        print(f"В {RAW_DIR} нет исходных файлов, скачиваю {KAGGLE_DATASET} с Kaggle...")
        try:
            download_raw(RAW_DIR)
        except Exception as error:   # сеть, авторизация Kaggle, отсутствие kagglehub - любая причина ведет к инструкции
            print(f"Не удалось скачать датасет: {error}\n\n"
                  f"Скачайте его вручную: {KAGGLE_URL}\n"
                  f"и положите train.csv, test.csv и store.csv в {RAW_DIR}, затем запустите скрипт снова.\n"
                  f"Для скачивания через API нужен токен Kaggle: https://www.kaggle.com/settings -> API -> "
                  f"Create New Token.", file=sys.stderr)
            return 1

    problems = check_raw(RAW_DIR)
    if problems:
        print("Исходные файлы не совпадают с оригинальным датасетом:\n- " + "\n- ".join(problems), file=sys.stderr)
        return 1

    stores = pd.read_csv(PROCESSED_DIR / "selected_stores.csv", index_col="Store").index
    for name, path in build_processed(RAW_DIR, stores, PROCESSED_DIR).items():
        print(f"{path}: {len(pd.read_csv(path))} строк, {path.stat().st_size / 1024:.0f} КБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
