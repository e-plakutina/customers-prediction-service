"""Обучение итоговой модели, оценка на фолдах и отложенном периоде, сохранение модели и запись прогона.

Запуск: python train.py
"""
import argparse
import json
import platform
from importlib.metadata import version

import pandas as pd

from customers_prediction.config import HOLDOUT_END, MODEL_PATH, RUNS_PATH, TEST_START, TRAIN_END
from customers_prediction.data import load_history, load_stores, period_days, restore_calendar
from customers_prediction.features import FINAL_FEATURES, build_features, category_levels
from customers_prediction.model import LGBM_PARAMS, SEGMENT_MODELS, GuestForecaster
from customers_prediction.tracking import git_commit, log_run
from customers_prediction.validation import (BASELINES, FOLDS, business_summary, fold_scores, make_baselines, metrics,
                                             run_folds, training_rows)



def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение модели прогноза числа покупателей")
    parser.add_argument("--no-save", action="store_true", help="не сохранять модель, только посчитать метрики")
    args = parser.parse_args()

    stores = load_stores()
    calendar = restore_calendar(load_history(), end=HOLDOUT_END).sort_values(["Store", "Date"], ignore_index=True)
    features = build_features(calendar, stores)
    levels = category_levels(stores)

    def make_model() -> GuestForecaster:
        return GuestForecaster(FINAL_FEATURES, levels, LGBM_PARAMS, SEGMENT_MODELS)

    # 1. валидация на фолдах внутри трейна
    folds = fold_scores(run_folds(features, calendar, make_model, FINAL_FEATURES), calendar)
    print("MAPE на фолдах внутри трейна, %:")
    print(folds["MAPE, %"].round(2).to_string(), end="\n\n")

    # 2. итоговая модель: обучение на всех днях до начала отложенного периода
    train = training_rows(features, TEST_START)
    model = make_model().fit(train, train["target"])

    # 3. отложенный период: модель и бейзлайны на одних и тех же открытых днях
    holdout = period_days(calendar, TEST_START, HOLDOUT_END)
    y = holdout["customers_clean"]
    baselines = make_baselines(calendar)
    scores = pd.DataFrame({name: metrics(y, baselines.loc[holdout.index, name]) for name in BASELINES}).T
    forecast = model.forecast(features.loc[holdout.index])
    scores.loc["LightGBM"] = metrics(y, forecast)
    print(f"Отложенный период {TEST_START:%d.%m.%Y}-{HOLDOUT_END:%d.%m.%Y}, {len(holdout)} открытых дней:")
    print(scores.round(2).to_string(), end="\n\n")
    business = pd.DataFrame({"LightGBM": business_summary(holdout, forecast, TEST_START),
                             "lag_7": business_summary(holdout, baselines.loc[holdout.index, "lag_7"], TEST_START)})
    print("Что это значит для бизнеса (отложенный период):")
    print(business.round(1).to_string(), end="\n\n")

    record = {
        "model": "LightGBM (общая + по сегментам)",
        "params": json.dumps(LGBM_PARAMS),
        "segment_models": ",".join(SEGMENT_MODELS),
        "n_features": len(FINAL_FEATURES),
        "train_end": f"{TRAIN_END:%Y-%m-%d}",
        **{f"fold_{fold.name}_mape": round(folds.loc[fold.name, "MAPE, %"], 3) for fold in FOLDS},
        "folds_mape": round(folds.loc["среднее", "MAPE, %"], 3),
        "holdout_mae": round(scores.loc["LightGBM", "MAE"], 3),
        "holdout_mape": round(scores.loc["LightGBM", "MAPE, %"], 3),
        "holdout_wape": round(scores.loc["LightGBM", "WAPE, %"], 3),
        **{f"holdout_mape_{name}": round(scores.loc[name, "MAPE, %"], 3) for name in BASELINES},
    }
    log_run(record)
    print(f"Прогон записан в {RUNS_PATH}")

    if not args.no_save:
        metadata = {
            "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_commit": git_commit(),
            "train_end": f"{TRAIN_END:%Y-%m-%d}",
            "features": FINAL_FEATURES,
            "params": LGBM_PARAMS,
            "segment_models": list(SEGMENT_MODELS),
            "folds_mape": folds["MAPE, %"].round(3).to_dict(),
            "holdout": scores.round(3).to_dict(orient="index"),
            "business": business["LightGBM"].round(2).to_dict(),
            "versions": {"python": platform.python_version(),
                         **{lib: version(lib) for lib in ["pandas", "numpy", "scikit-learn", "lightgbm"]}},
        }
        model.save(MODEL_PATH, metadata)
        print(f"Модель сохранена: {MODEL_PATH} (метаданные - {MODEL_PATH.with_suffix('.json').name})")


if __name__ == "__main__":
    main()
