"""Признаки не заглядывают в будущее: для дня t используется только история не позже t - 7 и план на день t."""
import numpy as np
import pandas as pd
import pytest

from customers_prediction.config import HORIZON, TEST_START
from customers_prediction.data import restore_calendar
from customers_prediction.features import (BASE_NUMERIC, CATEGORICAL, FINAL_FEATURES, build_features,
                                           category_levels)
from customers_prediction.validation import FOLDS

HISTORY_COLUMNS = BASE_NUMERIC + ["level"]


def features_of(history: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    calendar = restore_calendar(history, end=history["Date"].max()).sort_values(["Store", "Date"], ignore_index=True)
    return build_features(calendar, stores).set_index("Date")


def test_lag_7_is_same_weekday_week_ago(history, synthetic_stores):
    features = features_of(history, synthetic_stores)
    customers = history.set_index("Date")["Customers"].astype(float)
    day = pd.Timestamp("2015-04-01")                     # среда, неделю назад магазин был открыт
    expected = np.log(customers[day - pd.Timedelta(days=7)]) - features.loc[day, "level"]
    assert features.loc[day, "lag_7"] == pytest.approx(expected)


def test_future_values_do_not_change_features(history, synthetic_stores):
    """Изменение покупателей в дни t-6..t (и позже) не меняет признаки дня t."""
    day = pd.Timestamp("2015-04-01")
    changed = history.copy()
    recent = changed["Date"].between(day - pd.Timedelta(days=HORIZON - 1), changed["Date"].max())
    changed.loc[recent & (changed["Open"] == 1), "Customers"] = 10_000
    before, after = features_of(history, synthetic_stores), features_of(changed, synthetic_stores)
    pd.testing.assert_series_equal(before.loc[day, HISTORY_COLUMNS], after.loc[day, HISTORY_COLUMNS])


def test_value_seven_days_ago_changes_features(history, synthetic_stores):
    """Контроль чувствительности: изменение дня t-7 должно менять лаг и уровень дня t."""
    day = pd.Timestamp("2015-04-01")
    changed = history.copy()
    changed.loc[changed["Date"] == day - pd.Timedelta(days=HORIZON), "Customers"] = 10_000
    before, after = features_of(history, synthetic_stores), features_of(changed, synthetic_stores)
    assert before.loc[day, "lag_7"] != after.loc[day, "lag_7"]
    assert before.loc[day, "level"] != after.loc[day, "level"]


def test_target_is_not_a_feature(history, synthetic_stores):
    """Целевая дня t не участвует в признаках дня t напрямую."""
    assert not {"target", "Customers", "customers_clean", "Sales"} & set(FINAL_FEATURES)
    day = pd.Timestamp("2015-04-01")
    changed = history.copy()
    changed.loc[changed["Date"] == day, "Customers"] = 10_000
    before, after = features_of(history, synthetic_stores), features_of(changed, synthetic_stores)
    numeric = [c for c in FINAL_FEATURES if c not in CATEGORICAL]
    pd.testing.assert_series_equal(before.loc[day, numeric], after.loc[day, numeric])
    assert before.loc[day, "target"] != after.loc[day, "target"]


def test_closed_day_is_not_used_as_lag(history, synthetic_stores):
    """Если неделю назад магазин был закрыт, lag_7 - пропуск, а lag_filled берется двумя неделями раньше."""
    day = pd.Timestamp("2015-04-01")
    changed = history.copy()
    changed.loc[changed["Date"] == day - pd.Timedelta(days=7), ["Open", "Customers", "Sales"]] = 0
    features = features_of(changed, synthetic_stores)
    assert np.isnan(features.loc[day, "lag_7"])
    assert features.loc[day, "lag_filled"] == pytest.approx(features.loc[day, "lag_14"])
    assert features.loc[day, "lag_filled_weeks"] == 2


@pytest.mark.parametrize("origin", [fold.start for fold in FOLDS] + [TEST_START])
def test_no_leak_on_real_data(real_calendar, real_stores, origin):
    """Стираем всех покупателей начиная с даты прогноза: признаки 7 дней горизонта не меняются (раздел 3.2 ноутбука 03)."""
    features = build_features(real_calendar, real_stores)
    wiped = real_calendar.copy()
    wiped.loc[wiped["Date"] >= origin, "customers_clean"] = pd.NA
    rebuilt = build_features(wiped, real_stores)
    week = features["Date"].between(origin, origin + pd.Timedelta(days=HORIZON - 1))
    pd.testing.assert_frame_equal(features.loc[week, HISTORY_COLUMNS], rebuilt.loc[week, HISTORY_COLUMNS])


def test_category_levels_cover_real_data(real_calendar, real_stores):
    features = build_features(real_calendar, real_stores)
    levels = category_levels(real_stores)
    for column in CATEGORICAL:
        values = set(features[column].astype("string").dropna())
        assert values <= set(levels[column]), column


def test_unsorted_calendar_is_rejected(history, synthetic_stores):
    calendar = restore_calendar(history, end=history["Date"].max()).sort_values("Date", ascending=False)
    with pytest.raises(ValueError, match="отсортирован"):
        build_features(calendar, synthetic_stores)
