"""Модели: Ridge и LightGBM в виде Pipeline и итоговая модель GuestForecaster.

Все модели прогнозируют target = log(покупатели) - level; прогноз в покупателях - exp(прогноз + level).
"""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from customers_prediction.config import SEED

RIDGE_ALPHA = 10.0                                                       # раздел 4 ноутбука 03
LGBM_PARAMS = {"num_leaves": 63, "min_child_samples": 200, "n_estimators": 150}   # раздел 5
SEGMENT_MODELS = ("high_check",)                                          # раздел 7.3


class ModelCompatibilityError(ValueError):
    """Данные или код не соответствуют сохраненной модели."""


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Приводит категориальные колонки к фиксированному списку значений (features.category_levels).

    Одинаковые категории на обучении и инференсе дают одинаковое кодирование;
    значение вне списка - ошибка, а не тихий пропуск.
    """

    def __init__(self, levels: dict[str, list[str]]):
        self.levels = levels

    def fit(self, X: pd.DataFrame, y=None) -> "CategoryCaster":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column in X.columns.intersection(list(self.levels)):
            values = X[column].astype("string")
            unknown = values.notna() & ~values.isin(self.levels[column])
            if unknown.any():
                raise ValueError(f"Неизвестные значения признака {column}: {sorted(values[unknown].unique())[:5]}")
            X[column] = values.astype(pd.CategoricalDtype(self.levels[column]))
        return X


def make_ridge(levels: dict[str, list[str]], columns: list[str], alpha: float = RIDGE_ALPHA) -> Pipeline:
    """Ridge на тех же признаках, что LightGBM: пропуски - 0 и индикатор пропуска,
    масштабирование числовых признаков, one-hot для категориальных. Все шаги настраиваются на обучении.

    Категориальные признаки - те, для которых есть список значений в levels, остальные - числовые.
    """
    numeric = [c for c in columns if c not in levels]
    categorical = [c for c in columns if c in levels]
    return Pipeline([
        ("categories", CategoryCaster(levels)),
        ("columns", ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
                                  StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])),
        ("model", Ridge(alpha=alpha)),
    ])


def make_lgbm(levels: dict[str, list[str]], num_leaves: int = LGBM_PARAMS["num_leaves"],
              min_child_samples: int = LGBM_PARAMS["min_child_samples"],
              n_estimators: int = LGBM_PARAMS["n_estimators"]) -> Pipeline:
    """LightGBM: пропуски обрабатывает сам, категориальные признаки получает как категории."""
    return Pipeline([
        ("categories", CategoryCaster(levels)),
        ("model", lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=0.05, num_leaves=num_leaves,
                                    min_child_samples=min_child_samples, random_state=SEED, verbose=-1)),
    ])


class GuestForecaster:
    """Итоговая модель (раздел 7.4 ноутбука 03): общий LightGBM и отдельные модели для сегментов segment_models.

    fit/predict работают с target (отклонение от уровня магазина в логарифме), как обычная модель sklearn;
    forecast возвращает прогноз в покупателях.
    """

    def __init__(self, features: list[str], levels: dict[str, list[str]],
                 params: dict | None = None, segment_models: tuple[str, ...] = SEGMENT_MODELS):
        if segment_models and "segment" not in features:
            raise ValueError("Для моделей по сегментам среди признаков нужен segment")
        self.features = features
        self.levels = levels
        self.params = dict(LGBM_PARAMS if params is None else params)
        self.segment_models = segment_models
        self.models_: dict[str, Pipeline] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GuestForecaster":
        X = self.check_features(X)
        self.models_ = {"all": make_lgbm(self.levels, **self.params).fit(X, y)}
        for segment in self.segment_models:
            rows = X["segment"] == segment
            self.models_[segment] = make_lgbm(self.levels, **self.params).fit(X[rows], y[rows])
        return self

    def check_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Проверяет, что в X есть все признаки модели, числовые признаки числовые и конечные.

        Пропуски в числовых признаках допустимы: LightGBM обрабатывает их сам (например, лаг закрытого дня).
        """
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            raise ModelCompatibilityError(f"Нет признаков, на которых обучена модель: {missing}")
        numeric = [c for c in self.features if c not in self.levels]
        not_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(X[c])]
        if not_numeric:
            raise ModelCompatibilityError(f"Признаки должны быть числовыми: {not_numeric}")
        infinite = [c for c in numeric if np.isinf(X[c].to_numpy(dtype=float)).any()]
        if infinite:
            raise ModelCompatibilityError(f"Бесконечные значения в признаках: {infinite}")
        return X[self.features]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Прогноз target: общая модель, для сегментов из segment_models - их отдельные модели."""
        if not self.models_:
            raise RuntimeError("Модель не обучена")
        X = self.check_features(X)
        pred = pd.Series(self.models_["all"].predict(X), index=X.index)
        for segment in self.segment_models:
            rows = X["segment"] == segment
            if rows.any():
                pred[rows] = self.models_[segment].predict(X[rows])
        return pred.to_numpy()

    def forecast(self, frame: pd.DataFrame) -> pd.Series:
        """Прогноз числа покупателей: exp(прогноз target + level)."""
        if frame["level"].isna().any():
            raise ValueError("Нет уровня магазина: слишком короткая история")
        return pd.Series(np.exp(self.predict(frame) + frame["level"].to_numpy()), index=frame.index)

    def feature_importance(self) -> pd.Series:
        """Важность признаков общей модели (gain), в процентах."""
        gain = pd.Series(self.models_["all"].named_steps["model"].booster_.feature_importance("gain"),
                         index=self.features)
        return (gain / gain.sum() * 100).sort_values(ascending=False)

    def save(self, path: Path, metadata: dict) -> None:
        """Сохраняет модель (joblib) и рядом метаданные (json): дата обучения, признаки, параметры, метрики."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, default=str),
                                             encoding="utf-8")

    @staticmethod
    def load(path: Path, expected_features: list[str] | None = None) -> "GuestForecaster":
        """Загружает модель; если передан expected_features, сверяет его с признаками модели."""
        if not path.exists():
            raise FileNotFoundError(f"Нет обученной модели {path}: сначала запустите train.py")
        model = joblib.load(path)
        if not isinstance(model, GuestForecaster):
            raise ModelCompatibilityError(f"{path} не содержит GuestForecaster")
        if expected_features is not None and list(expected_features) != model.features:
            added = sorted(set(expected_features) - set(model.features))
            removed = sorted(set(model.features) - set(expected_features))
            raise ModelCompatibilityError(
                f"Модель {path.name} обучена на другом наборе признаков (в коде добавлены {added}, "
                f"удалены {removed}, или изменен порядок): переобучите ее командой python train.py")
        return model
