"""Проверка входа прогноза через pydantic: запрос и план работы на дни прогноза.

Эти объекты маленькие (один запрос, 7 дней плана), поэтому их удобно проверять построчно.
Большие таблицы истории проверяются векторно в data.validate_frame.
"""
import datetime
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

Flag = Annotated[int, Field(ge=0, le=1)]


class ForecastRequest(BaseModel):
    """Запрос прогноза: номер магазина и первый день прогноза."""
    model_config = ConfigDict(frozen=True)

    restaurant: int = Field(gt=0, description="номер магазина (Store)")
    date: datetime.date = Field(description="первый день прогноза")


class PlanDay(BaseModel):
    """План на один день прогноза: известен заранее и подается модели на вход."""
    date: datetime.date
    open: Flag = Field(description="1 - магазин работает, 0 - закрыт по расписанию")
    promo: Flag = Field(description="1 - в этот день промоакция")
    state_holiday: Literal["0", "a", "b", "c"] = Field(description="государственный праздник: 0 - нет, a/b/c - тип")
    school_holiday: Flag = Field(description="1 - школьные каникулы")


class ForecastDay(BaseModel):
    """Прогноз на один день."""
    date: datetime.date
    weekday: int = Field(ge=1, le=7, description="день недели, 1 - понедельник")
    open: bool = Field(description="работает ли точка по плану")
    promo: bool = Field(description="промоакция по плану")
    forecast: int = Field(ge=0, description="прогноз числа гостей (0 - закрыто по плану)")
    baseline_lag_7: int | None = Field(description="бейзлайн: гостей в тот же день неделю назад")
    actual: int | None = Field(description="фактическое число гостей, если уже известно")


class ForecastResponse(BaseModel):
    """Прогноз на 7 дней и, если факт известен, качество прогноза за эту неделю."""
    restaurant: int
    start: datetime.date
    end: datetime.date
    days: list[ForecastDay]
    metrics: dict[str, dict[str, float]] | None = Field(
        default=None, description="MAE, MAPE и WAPE модели и бейзлайна по открытым дням, если факт известен")


class RestaurantInfo(BaseModel):
    restaurant: int
    segment: str
    history_until: datetime.date = Field(description="последний день с известным числом гостей")
    plan_until: datetime.date | None = Field(description="последний день, на который есть план работы")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_trained_until: str
    restaurants: int


def validate_plan(week_plan: pd.DataFrame) -> list[PlanDay]:
    """Проверяет каждую строку плана; ошибка содержит дату и поле с недопустимым значением."""
    def plain(value):
        return value.item() if hasattr(value, "item") else value      # numpy-скаляр -> обычное число Python

    days = []
    for row in week_plan.itertuples(index=False):     # 7 строк: построчная проверка здесь уместна
        record = {"date": row.Date.date(), "open": plain(row.Open), "promo": plain(row.Promo),
                  "state_holiday": str(row.StateHoliday), "school_holiday": plain(row.SchoolHoliday)}
        try:
            days.append(PlanDay.model_validate(record))
        except ValidationError as error:
            fields = "; ".join(f"{e['loc'][0]}={e['input']!r}: {e['msg']}" for e in error.errors())
            raise ValueError(f"Некорректный план на {row.Date:%d.%m.%Y}: {fields}") from None
    return days
