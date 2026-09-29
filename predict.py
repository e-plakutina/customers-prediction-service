"""Прогноз числа покупателей магазина на 7 дней вперед.

Пример: python predict.py --date 2015-07-11 --restaurant 65 [--output forecast.csv]

--date - первый день прогноза. Модель видит историю только до предыдущего дня.
Если фактические значения за эти 7 дней известны (дата внутри истории), печатаются и метрики
прогноза в сравнении с бейзлайном "тот же день неделю назад".
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from customers_prediction.config import MODEL_PATH, PROCESSED_DIR
from customers_prediction.data import DataValidationError, load_history, load_plan, load_stores
from customers_prediction.features import FINAL_FEATURES
from customers_prediction.inference import ForecastError, forecast_week, week_metrics
from customers_prediction.model import GuestForecaster, ModelCompatibilityError

DAY_NAMES = {1: "Пн", 2: "Вт", 3: "Ср", 4: "Чт", 5: "Пт", 6: "Сб", 7: "Вс"}


def parse_date(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(pd.to_datetime(value, format="%Y-%m-%d"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"некорректная дата {value!r}, нужен формат ГГГГ-ММ-ДД") from None


def main() -> int:
    parser = argparse.ArgumentParser(description="Прогноз числа покупателей на 7 дней вперед")
    parser.add_argument("--date", type=parse_date, required=True, help="первый день прогноза, ГГГГ-ММ-ДД")
    parser.add_argument("--restaurant", type=int, required=True, help="номер магазина (Store)")
    parser.add_argument("--output", type=Path, help="сохранить прогноз в CSV")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="путь к обученной модели")
    args = parser.parse_args()

    plan_path = PROCESSED_DIR / "plan_subset.csv"
    try:
        model = GuestForecaster.load(args.model, expected_features=FINAL_FEATURES)
        history = load_history()
        plan = load_plan(plan_path) if plan_path.exists() else history.iloc[0:0]
        result = forecast_week(args.restaurant, args.date, history, plan, load_stores(), model)
    except (ForecastError, DataValidationError, ModelCompatibilityError, FileNotFoundError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        if isinstance(error, ForecastError) and "Нет плана" in str(error) and not plan_path.exists():
            print(f"План на даты после истории берется из {plan_path}; он создается командой python make_dataset.py",
                  file=sys.stderr)
        return 2

    table = pd.DataFrame({
        "дата": result["Date"].dt.strftime("%d.%m.%Y"),
        "день": result["DayOfWeek"].map(DAY_NAMES),
        "открыт": result["Open"].map({1: "да", 0: "нет"}),
        "промоакция": result["Promo"].map({1: "да", 0: "нет"}),
        "прогноз": result["forecast"].round().astype(int),
        "бейзлайн lag_7": result["baseline_lag_7"].round().astype("Int64"),
        "факт": result["actual"].round().astype("Int64"),
    })
    if table["факт"].isna().all():
        table = table.drop(columns="факт")
    else:
        table["факт"] = table["факт"].astype("string").fillna("-")     # будущие дни: факт еще неизвестен
    print(f"Прогноз числа покупателей, магазин {args.restaurant}, "
          f"{result['Date'].iloc[0]:%d.%m.%Y}-{result['Date'].iloc[-1]:%d.%m.%Y}:")
    print(table.to_string(index=False))

    scores = week_metrics(result)
    if scores is not None:
        print("\nКачество на этой неделе (открытые дни):")
        print(scores.round(2).to_string())

    if args.output:
        result.to_csv(args.output, index=False)
        print(f"\nПрогноз сохранен: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
