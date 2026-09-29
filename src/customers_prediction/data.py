"""Загрузка данных, восстановление календаря "магазин x день" и сборка data/processed."""
from pathlib import Path

import numpy as np
import pandas as pd

from customers_prediction.config import N_LONG, PROCESSED_DIR

DAY_STATUSES = ["open", "closed", "absent", "missing_export", "anomaly"]
READ_DTYPES = {"StateHoliday": "category", "Store": "int16"}
PLAN_COLUMNS = ["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
HISTORY_COLUMNS = PLAN_COLUMNS + ["Sales", "Customers"]
STATE_HOLIDAYS = {"0", "a", "b", "c"}


class DataValidationError(ValueError):
    """Входная таблица не соответствует ожидаемому формату."""


def validate_frame(df: pd.DataFrame, columns: list[str], name: str, allow_missing_open: bool = False) -> pd.DataFrame:
    """Векторная проверка таблицы истории или плана; возвращает ее без изменений или бросает DataValidationError.

    Проверяются: обязательные колонки, даты, одна строка на магазин и день, согласованность дня недели с датой,
    допустимые значения флагов (0/1) и праздника (0/a/b/c), неотрицательные покупатели и выручка.
    """
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise DataValidationError(f"{name}: нет колонок {missing}")
    problems = []
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]) or df["Date"].isna().any():
        problems.append("Date должна быть датой без пропусков")
    else:
        duplicated = df.duplicated(["Store", "Date"])
        if duplicated.any():
            problems.append(f"{duplicated.sum()} повторов пары Store + Date")
        wrong_weekday = df["DayOfWeek"] != df["Date"].dt.dayofweek + 1
        if wrong_weekday.any():
            problems.append(f"{wrong_weekday.sum()} строк, где DayOfWeek не соответствует Date")
    for flag in ["Open", "Promo", "SchoolHoliday"]:
        values = df[flag]
        allowed = values.isin([0, 1])
        if flag == "Open" and allow_missing_open:
            allowed |= values.isna()
        bad = ~allowed
        if bad.any():
            problems.append(f"{flag}: недопустимые значения {sorted(values[bad].astype(str).unique())[:5]} (нужно 0 или 1)")
    bad_holiday = ~df["StateHoliday"].astype("string").isin(STATE_HOLIDAYS)
    if bad_holiday.any():
        problems.append(f"StateHoliday: недопустимые значения {sorted(df.loc[bad_holiday, 'StateHoliday'].astype(str).unique())[:5]}")
    for value in ["Sales", "Customers"]:
        if value in columns and (df[value].isna() | (df[value] < 0)).any():
            problems.append(f"{value}: пропуски или отрицательные значения")
    if problems:
        raise DataValidationError(f"{name}: " + "; ".join(problems))
    return df


def load_history(path: Path = PROCESSED_DIR / "train_subset.csv") -> pd.DataFrame:
    """История магазинов выборки: одна строка на магазин и день, как в train.csv Rossmann."""
    history = pd.read_csv(path, parse_dates=["Date"], dtype=READ_DTYPES)
    return validate_frame(history, HISTORY_COLUMNS, f"История {path.name}")


def load_stores(processed_dir: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Характеристики магазинов (store.csv) вместе с сегментом и флагами отбора, индекс - Store."""
    stores = pd.read_csv(processed_dir / "store_subset.csv", index_col="Store")
    selected = pd.read_csv(processed_dir / "selected_stores.csv", index_col="Store")
    return stores.join(selected.drop(columns="StoreType"))


def load_plan(path: Path = PROCESSED_DIR / "plan_subset.csv") -> pd.DataFrame:
    """План работы на 01.08-17.09.2015 из test.csv Rossmann: расписание, промоакции, праздники, без покупателей.

    Open может быть пропущен (в test.csv так бывает): такой день считается днем без плана.
    """
    plan = pd.read_csv(path, parse_dates=["Date"], dtype=READ_DTYPES)
    return validate_frame(plan, PLAN_COLUMNS, f"План {path.name}", allow_missing_open=True)


def restore_calendar(df: pd.DataFrame, end: pd.Timestamp, n_long: int = N_LONG) -> pd.DataFrame:
    """Восстанавливает полный календарь "магазин x дата" до даты end включительно
    и присваивает каждому дню статус.

    Статусы:
    - open - магазин открыт, число покупателей - реальный спрос;
    - closed - регулярное закрытие: серия Open = 0 короче n_long дней (воскресенье, праздник), покупателей 0;
    - absent - точки нет в этот период: серия Open = 0 от n_long дней (ремонт или магазин появился позже);
    - missing_export - строки нет в выгрузке, значение неизвестно;
    - anomaly - магазин отмечен открытым, но покупателей 0.

    Исходные Customers и Sales не меняются, очищенные значения - в customers_clean и sales_clean.
    """
    dates = pd.date_range(df["Date"].min(), end, freq="D")
    grid = pd.MultiIndex.from_product([df["Store"].unique(), dates], names=["Store", "Date"])

    calendar = (df[df["Date"] <= end]
                .set_index(["Store", "Date"])
                .reindex(grid)
                .reset_index()
                # целые типы с поддержкой пропусков: у восстановленных строк значений нет
                .astype({"Open": "Int8", "Promo": "Int8", "SchoolHoliday": "Int8",
                         "Customers": "Int32", "Sales": "Int32"})
                # день недели известен и для отсутствующих строк
                .assign(DayOfWeek=lambda d: d["Date"].dt.dayofweek + 1))

    # Серии закрытых подряд дней. Отсутствующая строка (Open = NA) серию разрывает:
    # закрытия до и после пропуска выгрузки не склеиваются
    is_closed = calendar["Open"].eq(0).fillna(False).astype(bool)
    run_id = is_closed.ne(is_closed.groupby(calendar["Store"]).shift()).cumsum()
    run_length = is_closed.groupby(run_id).transform("size")

    is_open = calendar["Open"].eq(1).fillna(False).astype(bool)
    zero_customers = calendar["Customers"].eq(0).fillna(False).astype(bool)
    conditions = [
        calendar["Open"].isna().to_numpy(),                 # строки нет в выгрузке
        (is_closed & run_length.ge(n_long)).to_numpy(),     # длинное закрытие
        is_closed.to_numpy(),                               # регулярное закрытие
        (is_open & zero_customers).to_numpy(),              # открыт, но 0 покупателей
    ]
    calendar["day_status"] = pd.Categorical(
        np.select(conditions, ["missing_export", "absent", "closed", "anomaly"], default="open"),
        categories=DAY_STATUSES,
    )
    calendar["closure_run_days"] = run_length.where(is_closed)   # длина серии закрытия, для анализа

    # Очищенные значения: реальный спрос в открытые дни, 0 в регулярно закрытые, NaN во всех остальных
    known = calendar["day_status"].isin(["open", "closed"])
    calendar["customers_clean"] = calendar["Customers"].where(known).astype("Float64")
    calendar["sales_clean"] = calendar["Sales"].where(known).astype("Float64")
    return calendar


def period_days(calendar: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Открытые дни периода [start, end] - дни, по которым считаются метрики."""
    return calendar[(calendar["day_status"] == "open") & calendar["Date"].between(start, end)]


def build_processed(raw_dir: Path, stores: pd.Index, out_dir: Path) -> dict[str, Path]:
    """Собирает data/processed из сырых файлов Rossmann для магазинов выборки.

    train_subset.csv и store_subset.csv совпадают с файлами, которые сохраняет ноутбук 01,
    plan_subset.csv - план на август-сентябрь 2015 года из test.csv (есть не у всех магазинов выборки).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"], dtype=READ_DTYPES)
    store = pd.read_csv(raw_dir / "store.csv")
    plan = pd.read_csv(raw_dir / "test.csv", parse_dates=["Date"], dtype=READ_DTYPES)
    outputs = {
        "train_subset.csv": train[train["Store"].isin(stores)],
        "store_subset.csv": store[store["Store"].isin(stores)],
        "plan_subset.csv": plan.loc[plan["Store"].isin(stores), PLAN_COLUMNS].sort_values(["Store", "Date"]),
    }
    paths = {}
    for name, frame in outputs.items():
        paths[name] = out_dir / name
        frame.to_csv(paths[name], index=False)
    return paths
