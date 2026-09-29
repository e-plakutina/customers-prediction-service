"""Генерация признаков. Одни и те же функции используются при обучении и при инференсе.

Правило для всех признаков: для дня t используются либо история не позже t - HORIZON,
либо заранее известный план на день t (расписание работы, промоакции, праздники, каникулы).
"""
import numpy as np
import pandas as pd
from dateutil.easter import easter

from customers_prediction.config import HORIZON

LEVEL_DAYS = 24          # открытых дней в уровне магазина
MIN_LEVEL_DAYS = 12      # меньше - уровень не считается
LEVEL_AGE_CAP = 28       # верхняя граница level_age, дней (раздел 3.1 ноутбука 03)
HOLIDAY_WINDOW = 3       # дней до и после праздника
LAG_WEEKS = (1, 2, 3, 4)
NO_EVENT = "нет"         # значение признака события вне его окна

LAGS = [f"lag_{7 * w}" for w in LAG_WEEKS]
PROMO_LAGS = [f"promo_lag_{7 * w}" for w in LAG_WEEKS]

# набор признаков раздела 3 ноутбука 03
BASE_NUMERIC = ["Promo", "level_age", "lag_filled", "lag_filled_weeks", "lag_median", "rel_364"] + LAGS + PROMO_LAGS
BASE_CATEGORICAL = ["DayOfWeek", "month", "segment", "next_holiday", "prev_holiday", "easter_day",
                    "christmas_day", "carnival_day", "days_since_reopen", "days_to_long_closure"]
BASE_FEATURES = BASE_NUMERIC + BASE_CATEGORICAL

# кандидаты раздела 7.1 ноутбука 03
EXTRA_NUMERIC = ["day_of_month", "SchoolHoliday", "competition_open", "competition_distance_log", "promo2_active"]
EXTRA_CATEGORICAL = ["Assortment", "store_id"]
NUMERIC = BASE_NUMERIC + EXTRA_NUMERIC
CATEGORICAL = BASE_CATEGORICAL + EXTRA_CATEGORICAL

# итоговый набор (раздел 7.1): база + принятые кандидаты, в том же порядке, что в ноутбуке
FINAL_FEATURES = BASE_FEATURES + ["day_of_month", "SchoolHoliday", "Assortment", "competition_open",
                                  "competition_distance_log", "store_id"]


def window_label(days: pd.Series, low: int, high: int) -> pd.Series:
    """Номер дня внутри окна события [low, high] как строка, вне окна - NO_EVENT."""
    inside = days.between(low, high)
    return days.where(inside).astype("Int64").astype("string").where(inside, NO_EVENT)


def nearest_holiday(df: pd.DataFrame, holidays: pd.DataFrame, direction: str) -> pd.Series:
    """Тип и расстояние до ближайшего праздника магазина в пределах HOLIDAY_WINDOW дней: "a_1", "c_3" или NO_EVENT.

    direction="forward" - ближайший праздник впереди (включая сам день), "backward" - предыдущий праздник.
    """
    matched = pd.merge_asof(
        df[["Store", "Date"]].reset_index().sort_values("Date"),
        holidays.rename(columns={"Date": "holiday_date"}).sort_values("holiday_date"),
        left_on="Date", right_on="holiday_date", by="Store", direction=direction,
        tolerance=pd.Timedelta(days=HOLIDAY_WINDOW), allow_exact_matches=(direction == "forward"),
    ).set_index("index").sort_index()
    days = (matched["holiday_date"] - matched["Date"]).dt.days.abs()
    label = matched["kind"].astype("string") + "_" + days.astype("Int64").astype("string")
    return label.where(matched["holiday_date"].notna(), NO_EVENT)


def _history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Уровень магазина, целевая и лаги - только из истории не позже t - HORIZON."""
    log_y = np.log(df["customers_clean"].astype(float).where(df["day_status"] == "open"))
    log_by_store = log_y.groupby(df["Store"])

    # уровень магазина: последние LEVEL_DAYS открытых дней не позже t - HORIZON
    known = log_by_store.shift(HORIZON)
    df["level"] = known.groupby(df["Store"]).transform(
        lambda s: s.dropna().rolling(LEVEL_DAYS, min_periods=MIN_LEVEL_DAYS).mean().reindex(s.index).ffill())
    last_known = df["Date"].where(log_y.notna()).groupby(df["Store"]).shift(HORIZON).groupby(df["Store"]).ffill()
    df["level_age"] = (df["Date"] - last_known).dt.days.astype(float).clip(upper=LEVEL_AGE_CAP)
    df["target"] = log_y - df["level"]

    # лаги того же дня недели относительно уровня и промоакция в дни лагов
    promo = df["Promo"].astype(float)
    for week, lag, promo_lag in zip(LAG_WEEKS, LAGS, PROMO_LAGS):
        df[lag] = log_by_store.shift(7 * week) - df["level"]
        df[promo_lag] = promo.groupby(df["Store"]).shift(7 * week)
    lags = df[LAGS]
    df["lag_filled"] = lags.bfill(axis=1).iloc[:, 0]
    df["lag_filled_weeks"] = (lags.notna().to_numpy().argmax(axis=1) + 1).astype(float)
    df.loc[lags.isna().all(axis=1), "lag_filled_weeks"] = np.nan
    df["lag_median"] = lags.median(axis=1)
    df["rel_364"] = df["target"].groupby(df["Store"]).shift(364)
    df["Promo"] = promo
    return df


def _calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Календарь, праздники, Пасха, Рождество, карнавал и длинные закрытия - по плану на день t."""
    df["month"] = df["Date"].dt.month
    df["day_of_month"] = df["Date"].dt.day.astype(float)

    state_holiday = df["StateHoliday"].astype("string").fillna("0")
    is_holiday = state_holiday.ne("0")
    holidays = df.loc[is_holiday, ["Store", "Date"]].assign(kind=state_holiday[is_holiday])
    df["next_holiday"] = nearest_holiday(df, holidays, "forward")
    df["prev_holiday"] = nearest_holiday(df, holidays, "backward")

    easter_dates = df["Date"].dt.year.map({y: pd.Timestamp(easter(y)) for y in df["Date"].dt.year.unique()})
    df["easter_day"] = window_label((df["Date"] - easter_dates).dt.days, -7, 1)
    # карнавальный понедельник - Пасха минус 48 дней (раздел 6.4 ноутбука 02)
    df["carnival_day"] = window_label((df["Date"] - (easter_dates - pd.Timedelta(days=48))).dt.days, -4, 1)
    month_day = df["Date"].dt.strftime("%m-%d")
    df["christmas_day"] = month_day.where((month_day >= "12-01") | (month_day <= "01-06"), NO_EVENT)

    # длинные закрытия: день открытия после закрытия и дни до следующего (по плану работы)
    long_closed = df["day_status"].isin(["absent", "anomaly"])
    last_closed = df["Date"].where(long_closed).groupby(df["Store"]).ffill()
    next_closed = df["Date"].where(df["day_status"] == "absent").groupby(df["Store"]).bfill()
    df["days_since_reopen"] = window_label((df["Date"] - last_closed).dt.days - 1, 0, 13)
    df["days_to_long_closure"] = window_label((next_closed - df["Date"]).dt.days, 1, 7)
    df["SchoolHoliday"] = df["SchoolHoliday"].astype(float)
    return df


def _store_features(df: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Характеристики магазина: сегмент, ассортимент, конкурент и Promo2 на дату t, номер магазина."""
    store = df["Store"]
    df["segment"] = store.map(stores["segment"])
    df["Assortment"] = store.map(stores["Assortment"])
    df["store_id"] = store.astype("string")

    competition_since = pd.to_datetime(
        stores[["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"]]
        .rename(columns={"CompetitionOpenSinceYear": "year", "CompetitionOpenSinceMonth": "month"}).assign(day=1),
        errors="coerce")
    distance = store.map(stores["CompetitionDistance"])
    since = store.map(competition_since)
    # дата открытия неизвестна, а расстояние известно - считаем, что конкурент уже был
    competition_open = distance.notna() & (since.isna() | (df["Date"] >= since))
    df["competition_open"] = competition_open.astype(float)
    df["competition_distance_log"] = np.log(distance).where(competition_open)

    promo2_since = stores.apply(
        lambda s: pd.to_datetime(f"{int(s['Promo2SinceYear'])}-W{int(s['Promo2SinceWeek']):02d}-1", format="%G-W%V-%u")
        if s["Promo2"] == 1 else pd.NaT, axis=1)
    df["promo2_active"] = (df["Date"] >= store.map(promo2_since)).astype(float)
    return df


def build_features(calendar: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Признаки для каждой строки календаря (результат restore_calendar), индекс совпадает с календарем.

    Календарь должен быть полной сеткой "магазин x день", отсортированной по магазину и дате:
    сдвиг на k строк внутри магазина равен сдвигу на k дней.
    Кроме признаков возвращает level (уровень магазина) и target = log(покупатели) - level.
    Категориальные признаки возвращаются как есть; к фиксированным категориям их приводит модель.
    """
    ordered = calendar.sort_values(["Store", "Date"])
    if not ordered.index.equals(calendar.index):
        raise ValueError("Календарь должен быть отсортирован по Store и Date")
    unknown = set(calendar["Store"].unique()) - set(stores.index)
    if unknown:
        raise ValueError(f"Нет характеристик магазинов: {sorted(unknown)}")

    df = calendar[["Store", "Date", "DayOfWeek", "Promo", "StateHoliday", "SchoolHoliday", "day_status",
                   "customers_clean"]].copy()
    df = _history_features(df)
    df = _calendar_features(df)
    df = _store_features(df, stores)
    return df.drop(columns=["StateHoliday", "customers_clean"])


def category_levels(stores: pd.DataFrame) -> dict[str, list[str]]:
    """Все возможные значения каждого категориального признака.

    Список не зависит от данных, поэтому обучение и инференс кодируют категории одинаково,
    даже если в неделе прогноза встречается только часть значений.
    """
    def days(low: int, high: int) -> list[str]:
        return [str(d) for d in range(low, high + 1)] + [NO_EVENT]

    holiday_kinds = ["a", "b", "c"]
    christmas = pd.date_range("2000-12-01", "2001-01-06").strftime("%m-%d").tolist()
    levels = {
        "DayOfWeek": [str(d) for d in range(1, 8)],
        "month": [str(m) for m in range(1, 13)],
        "segment": stores["segment"].unique().tolist(),
        "next_holiday": [f"{k}_{d}" for k in holiday_kinds for d in range(0, HOLIDAY_WINDOW + 1)] + [NO_EVENT],
        "prev_holiday": [f"{k}_{d}" for k in holiday_kinds for d in range(1, HOLIDAY_WINDOW + 1)] + [NO_EVENT],
        "easter_day": days(-7, 1),
        "christmas_day": christmas + [NO_EVENT],
        "carnival_day": days(-4, 1),
        "days_since_reopen": days(0, 13),
        "days_to_long_closure": days(1, 7),
        "Assortment": ["a", "b", "c"],
        "store_id": [str(s) for s in stores.index],
    }
    return {name: sorted(values) for name, values in levels.items()}
