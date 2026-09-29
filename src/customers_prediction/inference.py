"""Прогноз на 7 дней вперед для одного магазина"""
import pandas as pd
from pydantic import ValidationError

from customers_prediction.config import HORIZON
from customers_prediction.data import PLAN_COLUMNS, restore_calendar
from customers_prediction.features import build_features
from customers_prediction.model import GuestForecaster
from customers_prediction.schemas import ForecastRequest, validate_plan
from customers_prediction.validation import make_baselines, metrics


class ForecastError(ValueError):
    """Ошибка входных данных прогноза с понятным пользователю сообщением."""


class UnknownRestaurantError(ForecastError):
    """Магазина нет в справочнике."""


def _week_plan(store: int, week: pd.DatetimeIndex, history: pd.DataFrame, plan: pd.DataFrame) -> pd.DataFrame:
    """План на дни недели прогноза: для прошлых дат - из истории, для будущих - из файла плана."""
    sources = [history.loc[history["Store"] == store, PLAN_COLUMNS], plan.loc[plan["Store"] == store, PLAN_COLUMNS]]
    week_plan = (pd.concat(sources, ignore_index=True)
                 .drop_duplicates("Date")                  # если дата есть и в истории, и в плане - берем историю
                 .set_index("Date")
                 .reindex(week))
    missing = week_plan.index[week_plan["Open"].isna()]
    if len(missing):
        dates = ", ".join(d.strftime("%d.%m.%Y") for d in missing)
        raise ForecastError(f"Нет плана работы магазина {store} на даты: {dates}. "
                            f"Для прогноза нужны расписание работы и промоакции на все {HORIZON} дней")
    week_plan = week_plan.rename_axis("Date").reset_index().assign(Store=store)
    try:
        validate_plan(week_plan)
    except ValueError as error:
        raise ForecastError(str(error)) from None
    return week_plan


def forecast_week(store: int, date: pd.Timestamp, history: pd.DataFrame, plan: pd.DataFrame,
                  stores: pd.DataFrame, model: GuestForecaster) -> pd.DataFrame:
    """Прогноз числа покупателей магазина store на дни date, ..., date + 6.

    Модель видит только историю до date - 1 включительно; план работы (расписание, промоакции,
    праздники, каникулы) на 7 дней берется из истории, если даты прошедшие, или из файла плана.
    Закрытые по плану дни прогнозируются нулем. Если фактическое число покупателей за эти дни известно,
    оно возвращается в колонке actual (только для сравнения, в прогнозе не участвует).
    """
    try:
        request = ForecastRequest(restaurant=store, date=pd.Timestamp(date).date())
    except ValidationError as error:
        fields = "; ".join(f"{e['loc'][0]}={e['input']!r}: {e['msg']}" for e in error.errors())
        raise ForecastError(f"Некорректный запрос: {fields}") from None
    except ValueError:
        raise ForecastError(f"Некорректная дата прогноза: {date!r}") from None
    store, date = request.restaurant, pd.Timestamp(request.date)
    if store not in stores.index:
        known = ", ".join(str(s) for s in sorted(stores.index))
        raise UnknownRestaurantError(f"Неизвестный магазин {store}. Доступные магазины: {known}")
    store_history = history[history["Store"] == store]
    first_day, last_day = store_history["Date"].min(), store_history["Date"].max()
    if date <= first_day:
        raise ForecastError(f"История магазина {store} начинается {first_day:%d.%m.%Y}: прогноз на более раннюю дату невозможен")
    if date > last_day + pd.Timedelta(days=1):
        raise ForecastError(f"История магазина {store} известна до {last_day:%d.%m.%Y}, поэтому прогноз можно "
                            f"начать не позже {last_day + pd.Timedelta(days=1):%d.%m.%Y}: для более поздней даты "
                            f"нужно фактическое число покупателей за пропущенные дни")

    week = pd.date_range(date, periods=HORIZON, freq="D")
    week_plan = _week_plan(store, week, store_history, plan)
    known_history = store_history[store_history["Date"] < date]          # будущее модели не показываем
    frame = pd.concat([known_history, week_plan], ignore_index=True)
    calendar = restore_calendar(frame, end=week[-1]).sort_values(["Store", "Date"], ignore_index=True)
    features = build_features(calendar, stores)

    in_week = features["Date"].isin(week).to_numpy()
    week_features = features[in_week]
    is_open = (calendar.loc[in_week, "Open"] == 1).to_numpy()
    if week_features.loc[is_open, "level"].isna().any():
        raise ForecastError(f"Слишком короткая история магазина {store} перед {date:%d.%m.%Y}: "
                            f"для уровня магазина нужно хотя бы 12 открытых дней не позже чем за 7 дней до прогноза")

    result = calendar.loc[in_week, ["Date", "DayOfWeek", "Open", "Promo", "StateHoliday", "SchoolHoliday"]].copy()
    result["forecast"] = 0.0
    result.loc[is_open, "forecast"] = model.forecast(week_features[is_open]).to_numpy()
    result["baseline_lag_7"] = make_baselines(calendar).loc[in_week, "lag_7"].where(is_open, 0.0)
    actual = store_history.set_index("Date")["Customers"].reindex(week)
    result["actual"] = actual.to_numpy()
    return result.reset_index(drop=True)


def week_metrics(result: pd.DataFrame) -> pd.DataFrame | None:
    """MAE, MAPE и WAPE прогноза и бейзлайна lag_7 по открытым дням недели, если факт известен для всех этих дней."""
    open_days = result[(result["Open"] == 1) & result["actual"].notna() & (result["actual"] > 0)]
    if open_days.empty or len(open_days) < (result["Open"] == 1).sum():
        return None
    return pd.DataFrame({
        "модель": metrics(open_days["actual"], open_days["forecast"]),
        "бейзлайн lag_7": metrics(open_days["actual"], open_days["baseline_lag_7"]),
    }).T
