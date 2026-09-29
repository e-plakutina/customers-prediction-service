"""HTTP API: прогноз совпадает с forecast_week, ошибки возвращаются с понятными кодами."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from customers_prediction.api import app
from customers_prediction.inference import forecast_week


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:     # with - чтобы выполнилась загрузка модели и данных при старте
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_trained_until": "2015-06-19", "restaurants": 40}


def test_restaurants(client):
    response = client.get("/restaurants")
    assert response.status_code == 200
    restaurants = {r["restaurant"]: r for r in response.json()}
    assert len(restaurants) == 40
    assert restaurants[75]["history_until"] == "2015-07-31"


def test_forecast_matches_cli(client, real_history, real_stores, saved_model):
    response = client.post("/forecast", json={"restaurant": 65, "date": "2015-07-11"})
    assert response.status_code == 200
    body = response.json()
    expected = forecast_week(65, pd.Timestamp("2015-07-11"), real_history, real_history.iloc[0:0], real_stores,
                             saved_model)
    assert [d["forecast"] for d in body["days"]] == expected["forecast"].round().astype(int).tolist()
    assert body["start"] == "2015-07-11" and body["end"] == "2015-07-17"
    assert body["metrics"]["модель"]["MAPE, %"] < body["metrics"]["бейзлайн lag_7"]["MAPE, %"]


def test_future_forecast_has_no_actuals(client):
    response = client.post("/forecast", json={"restaurant": 75, "date": "2015-08-01"})
    assert response.status_code == 200
    body = response.json()
    assert body["metrics"] is None
    assert all(day["actual"] is None for day in body["days"])


@pytest.mark.parametrize("payload, status", [
    ({"restaurant": 1, "date": "2015-07-11"}, 404),          # неизвестная точка
    ({"restaurant": 75, "date": "2026-10-01"}, 422),         # после конца истории
    ({"restaurant": 65, "date": "2015-08-01"}, 422),         # нет плана на август
    ({"restaurant": 75, "date": "2015-13-01"}, 422),         # некорректная дата
    ({"restaurant": 0, "date": "2015-07-11"}, 422),          # некорректный номер
    ({"date": "2015-07-11"}, 422),                           # нет поля
])
def test_errors(client, payload, status):
    response = client.post("/forecast", json=payload)
    assert response.status_code == status
    assert response.json()["detail"]
