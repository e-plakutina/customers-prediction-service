"""Восстановление календаря и проверка входных таблиц."""
import pandas as pd
import pytest

from customers_prediction.data import HISTORY_COLUMNS, DataValidationError, restore_calendar, validate_frame


def status_of(calendar: pd.DataFrame, date: str) -> str:
    return calendar.loc[calendar["Date"] == pd.Timestamp(date), "day_status"].iloc[0]


def test_three_kinds_of_missing_days(history):
    """Закрытый день, сбой выгрузки и отсутствие точки - разные статусы с разными значениями."""
    history = history.copy()
    history.loc[history["Date"].between("2015-02-10", "2015-02-14"), ["Open", "Customers", "Sales"]] = 0  # ремонт
    history.loc[history["Date"] == "2015-02-16", "Customers"] = 0                                          # аномалия
    history = history[history["Date"] != "2015-03-03"]                                                     # нет в выгрузке
    calendar = restore_calendar(history, end=history["Date"].max())

    assert status_of(calendar, "2015-01-04") == "closed"             # воскресенье
    assert status_of(calendar, "2015-02-12") == "absent"
    assert status_of(calendar, "2015-02-16") == "anomaly"
    assert status_of(calendar, "2015-03-03") == "missing_export"
    assert status_of(calendar, "2015-01-05") == "open"

    clean = calendar.set_index("Date")["customers_clean"]
    assert clean[pd.Timestamp("2015-01-04")] == 0                    # закрытый день - 0 покупателей
    for date in ["2015-02-12", "2015-02-16", "2015-03-03"]:          # спрос неизвестен - пропуск, а не 0
        assert pd.isna(clean[pd.Timestamp(date)])


def test_calendar_is_full_grid(history):
    history = history[history["Date"] != "2015-03-03"]
    calendar = restore_calendar(history, end=pd.Timestamp("2015-04-30"))
    assert len(calendar) == calendar["Date"].nunique()
    assert calendar["Date"].diff().dropna().eq(pd.Timedelta(days=1)).all()


def test_valid_history_passes(history):
    assert validate_frame(history, HISTORY_COLUMNS, "история") is history


@pytest.mark.parametrize("corrupt, message", [
    (lambda h: h.assign(Open=h["Open"].replace(1, 2)), "Open"),
    (lambda h: pd.concat([h, h.iloc[[0]]]), "повторов"),
    (lambda h: h.assign(Customers=-h["Customers"]), "Customers"),
    (lambda h: h.assign(DayOfWeek=1), "DayOfWeek"),
    (lambda h: h.drop(columns="Promo"), "нет колонок"),
])
def test_invalid_history_is_rejected(history, corrupt, message):
    with pytest.raises(DataValidationError, match=message):
        validate_frame(corrupt(history), HISTORY_COLUMNS, "история")
