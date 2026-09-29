"""Разбиение по времени, метрики, бейзлайны и сравнение моделей с бейзлайнами."""
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from customers_prediction.config import SEED
from customers_prediction.data import period_days


@dataclass(frozen=True)
class Fold:
    """Период валидации: модель обучается на всех днях до start и проверяется на открытых днях [start, end]."""
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


# фолды раздела 1.2 ноутбука 03: 6 недель, расширяющееся окно
FOLDS = (
    Fold("зима", pd.Timestamp("2014-12-06"), pd.Timestamp("2015-01-16")),
    Fold("весна", pd.Timestamp("2015-03-28"), pd.Timestamp("2015-05-08")),
    Fold("лето", pd.Timestamp("2015-05-09"), pd.Timestamp("2015-06-19")),
)
BASELINES = ["lag_7", "lag_14", "median_7_28"]


def metrics(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """MAE, MAPE и WAPE по открытым дням (у открытых дней покупателей всегда больше 0)."""
    y_true, y_pred = y_true.astype(float), y_pred.astype(float)
    if y_pred.isna().any():
        raise ValueError("Прогноз должен быть на каждый открытый день")
    if (y_true <= 0).any():
        raise ValueError("В метрики попал день без покупателей")
    error = (y_true - y_pred).abs()
    return pd.Series({"MAE": error.mean(),
                      "MAPE, %": (error / y_true).mean() * 100,
                      "WAPE, %": error.sum() / y_true.sum() * 100})


def business_summary(days: pd.DataFrame, pred: pd.Series, start: pd.Timestamp) -> pd.Series:
    """Ошибка прогноза в понятных для бизнеса величинах (по открытым дням days, прогноз pred).

    Кроме средних ошибок - доля дней с большой ошибкой и ошибка недельной суммы гостей магазина
    (от нее зависят закупки на неделю).
    """
    y = days["customers_clean"].astype(float)
    error = (pred - y).abs()
    ape = error / y * 100
    week = (days["Date"] - start).dt.days // 7
    totals = pd.DataFrame({"y": y, "pred": pred, "store": days["Store"], "week": week}).groupby(["store", "week"]).sum()
    week_error = (totals["pred"] - totals["y"]).abs() / totals["y"] * 100
    return pd.Series({
        "гостей в день, в среднем": y.mean(),
        "ошибка в день, медиана": error.median(),
        "ошибка в день, 90% дней не больше": error.quantile(0.9),
        "дней с ошибкой до 5%, %": (ape <= 5).mean() * 100,
        "дней с ошибкой больше 10%, %": (ape > 10).mean() * 100,
        "дней с ошибкой больше 20%, %": (ape > 20).mean() * 100,
        "ошибка недельной суммы магазина, медиана, %": week_error.median(),
        "ошибка недельной суммы магазина, 90% недель не больше, %": week_error.quantile(0.9),
        "смещение (прогноз - факт) от суммы факта, %": (pred - y).sum() / y.sum() * 100,
    })


def same_weekday_history(calendar: pd.DataFrame, max_weeks: int = 52) -> pd.DataFrame:
    """Покупатели в тот же день недели k недель назад (k = 1..max_weeks), только открытые дни, иначе NaN.

    Календарь - полная сетка "магазин x день", поэтому сдвиг на 7k строк внутри магазина = 7k дней.
    """
    open_value = calendar["customers_clean"].astype(float).where(calendar["day_status"] == "open")
    by_store = open_value.groupby(calendar["Store"])
    return pd.DataFrame({k: by_store.shift(7 * k) for k in range(1, max_weeks + 1)}, index=calendar.index)


def first_available(history: pd.DataFrame) -> pd.Series:
    """Первое непустое значение слева направо - ближайшая доступная неделя."""
    return history.bfill(axis=1).iloc[:, 0]


def make_baselines(calendar: pd.DataFrame) -> pd.DataFrame:
    """Бейзлайны раздела 2 ноутбука 03: тот же день недели в прошлом, не ближе 7 дней назад.

    - lag_7 - тот же день неделю назад (если день был закрыт - следующая по давности неделя);
    - lag_14 - то же, но только четное число недель назад (чаще та же фаза промоакции);
    - median_7_28 - медиана того же дня 1-4 недели назад.
    """
    history = same_weekday_history(calendar)
    return pd.DataFrame({
        "lag_7": first_available(history),
        "lag_14": first_available(history[history.columns[1::2]]),
        "median_7_28": history[[1, 2, 3, 4]].median(axis=1).fillna(first_available(history)),
    })


def training_rows(features: pd.DataFrame, before: pd.Timestamp, target: pd.Series | None = None) -> pd.DataFrame:
    """Строки для обучения: дни до даты before, для которых известна целевая."""
    target = features["target"] if target is None else target
    return features[(features["Date"] < before) & target.notna()]


def run_folds(features: pd.DataFrame, calendar: pd.DataFrame, make_model: Callable[[], object],
              columns: list[str], folds: tuple[Fold, ...] = FOLDS,
              train_target: Callable[[Fold], pd.Series] | None = None,
              rows: Callable[[pd.DataFrame], pd.Series] | None = None) -> pd.DataFrame:
    """Прогноз на открытые дни фолдов; для каждого фолда новая модель обучается на всех днях до его начала.

    make_model() возвращает модель с методами fit(X, y) и predict(X), которая прогнозирует target.
    train_target(fold) - целевая для обучения (по умолчанию features["target"]),
    rows(frame) - отбор строк для обучения и прогноза (например, один сегмент).
    Возвращает прогноз в покупателях: exp(прогноз + level), с колонкой fold.
    """
    parts = []
    for fold in folds:
        target = features["target"] if train_target is None else train_target(fold)
        train = training_rows(features, fold.start, target)
        test = features.loc[period_days(calendar, fold.start, fold.end).index]
        if rows is not None:
            train, test = train[rows(train)], test[rows(test)]
        if test["level"].isna().any():
            raise ValueError(f"Фолд {fold.name}: нет уровня магазина для части дней")
        model = make_model()
        model.fit(train[columns], target.loc[train.index])
        relative = np.asarray(model.predict(test[columns]))
        parts.append(pd.DataFrame({"fold": fold.name, "pred": np.exp(relative + test["level"])}, index=test.index))
    return pd.concat(parts)


def baseline_folds(baselines: pd.DataFrame, calendar: pd.DataFrame, name: str,
                   folds: tuple[Fold, ...] = FOLDS) -> pd.DataFrame:
    """Прогноз бейзлайна на открытые дни фолдов в том же формате, что run_folds."""
    return pd.concat([pd.DataFrame({"fold": fold.name,
                                    "pred": baselines.loc[period_days(calendar, fold.start, fold.end).index, name]})
                      for fold in folds])


def fold_scores(pred: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Метрики по фолдам и строка "среднее" - среднее по фолдам."""
    y = calendar.loc[pred.index, "customers_clean"]
    scores = pd.DataFrame({fold: metrics(y.loc[part.index], part["pred"])
                           for fold, part in pred.groupby("fold", sort=False)}).T
    scores.loc["среднее"] = scores.mean()
    return scores


def compare_to_base(variant_scores: pd.DataFrame, base_scores: pd.DataFrame,
                    folds: tuple[Fold, ...] = FOLDS) -> dict:
    """Изменение MAPE варианта относительно базы и решение по правилу принятия раздела 7:
    средняя MAPE уменьшается и MAPE улучшается хотя бы в двух фолдах."""
    names = [fold.name for fold in folds]
    delta = variant_scores["MAPE, %"] - base_scores["MAPE, %"]
    improved = int((delta.loc[names] < 0).sum())
    return {**{f"изменение MAPE {name}, п.п.": delta.loc[name] for name in names},
            "изменение MAPE среднее, п.п.": delta.loc["среднее"],
            "фолдов с улучшением": improved,
            "принять": bool(delta.loc["среднее"] < 0 and improved >= 2)}


def block_bootstrap(y_true: pd.Series, predictions: dict[str, pd.Series], blocks: pd.Series,
                    n_boot: int = 2000, seed: int = SEED) -> dict[str, dict[str, np.ndarray]]:
    """MAE и MAPE каждой модели на n_boot бутстреп-выборках блоков (например, "магазин x неделя").

    Блоки выбираются с возвращением; выборки одинаковы для всех моделей, поэтому их можно сравнивать попарно.
    """
    codes, names = pd.factorize(blocks)
    sample = np.random.default_rng(seed).integers(0, len(names), size=(n_boot, len(names)))

    def sums(values) -> np.ndarray:
        per_block = np.bincount(codes, weights=np.asarray(values, dtype=float), minlength=len(names))
        return per_block[sample].sum(axis=1)

    count = sums(np.ones(len(y_true)))
    result = {}
    for name, pred in predictions.items():
        error = (pred - y_true).abs()
        result[name] = {"MAE": sums(error) / count, "MAPE, %": sums(error / y_true) / count * 100}
    return result
