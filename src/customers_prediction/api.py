"""HTTP API прогноза числа гостей (FastAPI).

Запуск: uvicorn customers_prediction.api:app --port 8000, документация - http://localhost:8000/docs.
Прогноз считает та же функция forecast_week, что и predict.py.
"""
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pandas as pd
from fastapi import FastAPI, HTTPException

from customers_prediction.config import MODEL_PATH, PROCESSED_DIR
from customers_prediction.data import load_history, load_plan, load_stores
from customers_prediction.features import FINAL_FEATURES
from customers_prediction.inference import ForecastError, UnknownRestaurantError, forecast_week, week_metrics
from customers_prediction.model import GuestForecaster
from customers_prediction.schemas import (ForecastDay, ForecastRequest, ForecastResponse, HealthResponse,
                                          RestaurantInfo)


@dataclass(frozen=True)
class Resources:
    """Все, что нужно для прогноза: загружается один раз при старте сервиса."""
    model: GuestForecaster
    metadata: dict
    history: pd.DataFrame
    plan: pd.DataFrame
    stores: pd.DataFrame


def load_resources() -> Resources:
    plan_path = PROCESSED_DIR / "plan_subset.csv"
    history = load_history()
    return Resources(
        model=GuestForecaster.load(MODEL_PATH, expected_features=FINAL_FEATURES),
        metadata=json.loads(MODEL_PATH.with_suffix(".json").read_text(encoding="utf-8")),
        history=history,
        plan=load_plan(plan_path) if plan_path.exists() else history.iloc[0:0],
        stores=load_stores(),
    )


resources: Resources | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global resources
    resources = load_resources()
    yield


app = FastAPI(title="Прогноз числа гостей", version="1.0",
              description="Прогноз числа гостей точки на 7 дней вперед (LightGBM, данные Rossmann).",
              lifespan=lifespan)


def to_int(value) -> int | None:
    return None if pd.isna(value) else int(round(float(value)))


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_trained_until=resources.metadata["train_end"],
                          restaurants=len(resources.stores))


@app.get("/restaurants", response_model=list[RestaurantInfo])
def restaurants() -> list[RestaurantInfo]:
    history_until = resources.history.groupby("Store")["Date"].max()
    plan_until = resources.plan.groupby("Store")["Date"].max()
    return [RestaurantInfo(restaurant=int(store), segment=resources.stores.loc[store, "segment"],
                           history_until=history_until[store].date(),
                           plan_until=plan_until[store].date() if store in plan_until.index else None)
            for store in resources.stores.index]


@app.post("/forecast", response_model=ForecastResponse,
          responses={404: {"description": "Неизвестная точка"},
                     422: {"description": "Некорректный запрос, нет плана или слишком короткая история"}})
def forecast(request: ForecastRequest) -> ForecastResponse:
    """Прогноз на 7 дней начиная с request.date; модель видит историю только до предыдущего дня."""
    try:
        result = forecast_week(request.restaurant, pd.Timestamp(request.date), resources.history, resources.plan,
                               resources.stores, resources.model)
    except UnknownRestaurantError as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except ForecastError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None

    days = [ForecastDay(date=row.Date.date(), weekday=int(row.DayOfWeek), open=bool(row.Open == 1),
                        promo=bool(row.Promo == 1), forecast=to_int(row.forecast),
                        baseline_lag_7=to_int(row.baseline_lag_7), actual=to_int(row.actual))
            for row in result.itertuples(index=False)]
    scores = week_metrics(result)
    return ForecastResponse(
        restaurant=request.restaurant, start=days[0].date, end=days[-1].date, days=days,
        metrics=None if scores is None else scores.round(2).to_dict(orient="index"),
    )
