"""Общие данные для тестов: маленькая синтетическая история и реальная выборка из data/processed."""
import numpy as np
import pandas as pd
import pytest

from customers_prediction.config import HOLDOUT_END, MODEL_PATH
from customers_prediction.data import load_history, load_stores, restore_calendar
from customers_prediction.features import FINAL_FEATURES
from customers_prediction.model import GuestForecaster


def make_history(days: int = 120, start: str = "2015-01-01", store: int = 1, seed: int = 0) -> pd.DataFrame:
    """Синтетическая история одного магазина в формате train.csv Rossmann: открыт каждый день, кроме воскресенья."""
    dates = pd.date_range(start, periods=days, freq="D")
    rng = np.random.default_rng(seed)
    open_ = (dates.dayofweek != 6).astype(int)
    customers = np.where(open_ == 1, rng.integers(400, 600, size=days), 0)
    return pd.DataFrame({
        "Store": np.int16(store), "DayOfWeek": dates.dayofweek + 1, "Date": dates,
        "Sales": customers * 10, "Customers": customers, "Open": open_,
        "Promo": ((dates.isocalendar().week.to_numpy() % 2 == 0) & (dates.dayofweek < 5)).astype(int),
        "StateHoliday": pd.Categorical(["0"] * days), "SchoolHoliday": 0,
    })


@pytest.fixture
def history() -> pd.DataFrame:
    return make_history()


@pytest.fixture
def synthetic_stores() -> pd.DataFrame:
    """Характеристики синтетического магазина 1 в формате load_stores."""
    return pd.DataFrame({
        "StoreType": ["a"], "Assortment": ["a"], "CompetitionDistance": [500.0],
        "CompetitionOpenSinceMonth": [1.0], "CompetitionOpenSinceYear": [2010.0], "Promo2": [0],
        "Promo2SinceWeek": [np.nan], "Promo2SinceYear": [np.nan], "PromoInterval": [np.nan], "segment": ["mid"],
    }, index=pd.Index([1], name="Store"))


@pytest.fixture(scope="session")
def real_history() -> pd.DataFrame:
    return load_history()


@pytest.fixture(scope="session")
def real_stores() -> pd.DataFrame:
    return load_stores()


@pytest.fixture(scope="session")
def real_calendar(real_history) -> pd.DataFrame:
    return restore_calendar(real_history, end=HOLDOUT_END).sort_values(["Store", "Date"], ignore_index=True)


@pytest.fixture(scope="session")
def saved_model() -> GuestForecaster:
    return GuestForecaster.load(MODEL_PATH, expected_features=FINAL_FEATURES)
