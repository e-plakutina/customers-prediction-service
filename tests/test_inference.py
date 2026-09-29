"""Прогноз predict.py: те же признаки и модель, что при обучении, будущее не используется, понятные ошибки."""
import numpy as np
import pandas as pd
import pytest

from customers_prediction.config import TEST_START
from customers_prediction.features import FINAL_FEATURES, build_features
from customers_prediction.inference import ForecastError, forecast_week
from customers_prediction.model import GuestForecaster, ModelCompatibilityError

STORE = 75
EMPTY_PLAN = pd.DataFrame(columns=["Store", "DayOfWeek", "Date", "Open", "Promo", "StateHoliday", "SchoolHoliday"])


def test_forecast_matches_training_pipeline(real_history, real_stores, real_calendar, saved_model):
    """Недельный прогноз совпадает с прогнозом по календарю, построенному так же, как при обучении."""
    result = forecast_week(STORE, TEST_START, real_history, EMPTY_PLAN, real_stores, saved_model)
    features = build_features(real_calendar, real_stores)
    week = features[(features["Store"] == STORE) & features["Date"].isin(result["Date"])]
    is_open = (result["Open"] == 1).to_numpy()
    expected = saved_model.forecast(week[is_open]).to_numpy()
    np.testing.assert_allclose(result.loc[is_open, "forecast"], expected)
    assert (result.loc[~is_open, "forecast"] == 0).all()        # закрытые по плану дни - 0


def test_history_after_date_is_not_used(real_history, real_stores, saved_model):
    """Изменение истории начиная с даты прогноза не меняет прогноз."""
    changed = real_history.copy()
    after = (changed["Store"] == STORE) & (changed["Date"] >= TEST_START)
    changed.loc[after, "Customers"] = changed.loc[after, "Customers"] * 3
    before = forecast_week(STORE, TEST_START, real_history, EMPTY_PLAN, real_stores, saved_model)
    after_change = forecast_week(STORE, TEST_START, changed, EMPTY_PLAN, real_stores, saved_model)
    pd.testing.assert_series_equal(before["forecast"], after_change["forecast"])


@pytest.mark.parametrize("store, date, message", [
    (1, "2015-07-01", "Неизвестный магазин"),
    (STORE, "2026-10-01", "не позже"),
    (STORE, "2012-01-01", "начинается"),
    (STORE, "2013-01-05", "короткая история"),
    (STORE, "2015-08-01", "Нет плана"),                          # план не передан
    (0, "2015-07-01", "Некорректный запрос"),
])
def test_errors_are_readable(real_history, real_stores, saved_model, store, date, message):
    with pytest.raises(ForecastError, match=message):
        forecast_week(store, pd.Timestamp(date), real_history, EMPTY_PLAN, real_stores, saved_model)


def test_model_rejects_other_feature_set(saved_model, tmp_path):
    path = tmp_path / "model.joblib"
    saved_model.save(path, metadata={})
    with pytest.raises(ModelCompatibilityError, match="другом наборе признаков"):
        GuestForecaster.load(path, expected_features=FINAL_FEATURES[:-1])


def test_unknown_category_is_rejected(real_calendar, real_stores, saved_model):
    features = build_features(real_calendar, real_stores).head(5).copy()
    features["christmas_day"] = "13-45"
    with pytest.raises(ValueError, match="Неизвестные значения признака christmas_day"):
        saved_model.predict(features)
