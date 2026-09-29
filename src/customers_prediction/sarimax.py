"""SARIMAX по каждому магазину - альтернативный подход к сезонности (раздел 8 ноутбука 03).

Регрессия с ошибками ARIMA на логарифме покупателей: сезонность и события задаются регрессорами,
ARIMA-часть описывает, как отклонения от них сохраняются во времени.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from customers_prediction.config import HORIZON
from customers_prediction.features import NO_EVENT

SARIMAX_SPECS = {"ARMA(1, 1)": (1, 0, 1), "ARIMA(1, 1, 1)": (1, 1, 1)}
SELECTED_SPEC = "ARIMA(1, 1, 1)"          # выбрана в разделе 8 по средней MAPE на фолдах


def sarimax_exog(features: pd.DataFrame) -> pd.DataFrame:
    """Регрессоры: день недели, промоакция, дни у праздников, Пасха, рождественский период,
    карнавал, закрытия, годовые гармоники."""
    X = pd.DataFrame(index=features.index)
    for day in range(1, 8):
        X[f"dow_{day}"] = (features["DayOfWeek"].astype(int) == day).astype(float)
    X["promo"] = features["Promo"].fillna(0).astype(float)
    X["before_holiday"] = features["next_holiday"].astype("string").str.match(r"^\w_[123]$").astype(float)
    X["after_holiday"] = features["prev_holiday"].ne(NO_EVENT).astype(float)
    easter_day = pd.to_numeric(features["easter_day"].astype("string").replace(NO_EVENT, pd.NA), errors="coerce")
    X["easter_week"] = easter_day.between(-6, -3).astype(float)
    X["easter_saturday"] = easter_day.eq(-1).astype(float)
    day = features["christmas_day"].astype("string")
    X["pre_christmas"] = day.between("12-17", "12-23").astype(float)
    X["short_day"] = day.isin(["12-24", "12-31"]).astype(float)
    X["between_holidays"] = day.between("12-27", "12-30").astype(float)
    X["early_january"] = day.between("01-02", "01-06").astype(float)
    X["carnival"] = features["carnival_day"].ne(NO_EVENT).astype(float)
    X["after_reopen"] = features["days_since_reopen"].ne(NO_EVENT).astype(float)
    X["before_closure"] = features["days_to_long_closure"].ne(NO_EVENT).astype(float)
    day_of_year = features["Date"].dt.dayofyear
    for k in (1, 2):
        X[f"sin_{k}"] = np.sin(2 * np.pi * k * day_of_year / 365.25)
        X[f"cos_{k}"] = np.cos(2 * np.pi * k * day_of_year / 365.25)
    return X.fillna(0.0)


def forecast_store(store: int, order: tuple[int, int, int], start: pd.Timestamp, end: pd.Timestamp,
                   features: pd.DataFrame, calendar: pd.DataFrame, exog: pd.DataFrame | None = None) -> pd.Series:
    """Прогноз магазина на дни [start, end] недельными шагами.

    Параметры оцениваются один раз по данным до start. Для каждой недели модель с этими параметрами
    фильтрует всю историю до начала недели и прогнозирует HORIZON дней вперед.
    Закрытые дни, пробелы выгрузки и периоды без точки - пропуски ряда (фильтр Калмана их пропускает).
    """
    exog = sarimax_exog(features) if exog is None else exog
    rows = features.index[features["Store"] == store]
    dates = features.loc[rows, "Date"]
    endog = np.log(calendar.loc[rows, "customers_clean"].astype(float).where(calendar.loc[rows, "day_status"] == "open"))
    y = pd.Series(endog.to_numpy(), index=pd.DatetimeIndex(dates, freq="D"))
    X = pd.DataFrame(exog.loc[rows].to_numpy(), index=y.index, columns=exog.columns)
    train_y, train_X = y[y.index < start], X[X.index < start]
    observed = train_y.notna()
    # регрессоры без вариации на наблюдаемых днях (например, воскресенье у закрытого по воскресеньям магазина) не оцениваются
    columns = [c for c in X.columns if train_X.loc[observed, c].nunique() > 1 or c.startswith("dow_")]
    columns = [c for c in columns if not c.startswith("dow_") or train_X.loc[observed, c].sum() > 0]
    if order[1] == 1:    # при разности уровень задается случайным блужданием, один день недели - база
        columns = [c for c in columns if c != "dow_1"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(train_y, exog=train_X[columns], order=order, trend="n").fit(disp=False, maxiter=200)
        parts = []
        for week_start in pd.date_range(start, end, freq=f"{HORIZON}D"):
            history = fitted.apply(y[y.index < week_start], exog=X.loc[X.index < week_start, columns])
            horizon = X.loc[week_start:week_start + pd.Timedelta(days=HORIZON - 1), columns]
            parts.append(np.exp(history.forecast(len(horizon), exog=horizon)))
    return pd.Series(pd.concat(parts).to_numpy(), index=rows[dates.between(start, end).to_numpy()])


def forecast_period(order: tuple[int, int, int], start: pd.Timestamp, end: pd.Timestamp,
                    features: pd.DataFrame, calendar: pd.DataFrame) -> pd.Series:
    """Прогноз всех магазинов с открытыми днями в [start, end] на открытые дни периода."""
    open_days = calendar[(calendar["day_status"] == "open") & calendar["Date"].between(start, end)]
    exog = sarimax_exog(features)
    pred = pd.concat([forecast_store(store, order, start, end, features, calendar, exog)
                      for store in open_days["Store"].unique()])
    return pred.loc[open_days.index]
