"""Отбор выборки магазинов и проверка ее баланса (ноутбук 01_data_research).

Правила отбора (обязательные группы, минимум на сегмент, размер выборки) выведены из анализа в ноутбуке 01
и передаются параметрами; здесь - только механика отбора и метрики сравнения выборки с сетью.
"""
import numpy as np
import pandas as pd
from scipy.stats import kruskal


def epsilon_squared(frame: pd.DataFrame, group_col: str, value_col: str) -> float:
    """Размер эффекта для критерия Краскела-Уоллиса: доля различий value_col, объясняемая группами group_col.

    Используется, чтобы выбрать характеристики магазина, связанные со спросом, для стратификации.
    """
    groups = [values.dropna() for _, values in frame.groupby(group_col, observed=True)[value_col]]
    groups = [g for g in groups if len(g) > 0]
    h_stat, _ = kruskal(*groups)
    return h_stat / (sum(len(g) for g in groups) - 1)


def proportional_quota(sizes: pd.Series, total: int, minimum: int = 0) -> pd.Series:
    """Квоты пропорционально размерам групп (метод наибольших остатков).

    Группы, чья пропорциональная доля меньше minimum, поднимаются до minimum за счет остальных.
    """
    raised = sizes.index[sizes / sizes.sum() * total < minimum]
    quota = pd.Series(0, index=sizes.index)
    quota[raised] = np.minimum(minimum, sizes[raised])
    rest = sizes.drop(raised)
    exact = rest / rest.sum() * (total - quota.sum())
    quota[rest.index] = np.floor(exact).astype(int)
    leftover = int(total - quota.sum())
    quota[(exact - np.floor(exact)).sort_values(ascending=False).index[:leftover]] += 1
    return quota


def select_stores(pool: pd.DataFrame, size: int, seed: int, mandatory_groups: dict[str, int],
                  min_per_segment: int) -> pd.Series:
    """Отбирает size магазинов из pool и возвращает причину включения каждого.

    1. Обязательные группы (флаги pool): в выборке должно быть не меньше mandatory_groups[flag] магазинов с флагом;
       квоты накопительные - магазин, уже попавший в выборку, засчитывается во все свои группы.
    2. Стратифицированный добор: квоты по сегментам пропорционально размеру (не меньше min_per_segment),
       внутри сегмента - пропорционально терцилям роста (growth_tercile), с учетом уже выбранных магазинов.
    Случайность засеяна seed, поэтому отбор воспроизводим.
    """
    rng = np.random.default_rng(seed)
    reasons = pd.Series(dtype=str)

    for flag, needed in mandatory_groups.items():
        already = pool.loc[pool.index.isin(reasons.index), flag].sum()
        candidates = pool.index[pool[flag] & ~pool.index.isin(reasons.index)].sort_values()
        picked = rng.choice(candidates, size=max(needed - already, 0), replace=False)
        reasons = pd.concat([reasons, pd.Series(f"обязательный: {flag}", index=picked)])

    segment_quota = proportional_quota(pool["segment"].value_counts(), size, minimum=min_per_segment)
    for segment, seg_quota in segment_quota.items():
        seg_pool = pool[pool["segment"] == segment]
        chosen = seg_pool.index.isin(reasons.index)
        tercile_target = proportional_quota(seg_pool["growth_tercile"].value_counts(), seg_quota)
        deficit = (tercile_target - seg_pool.loc[chosen, "growth_tercile"].value_counts()
                   .reindex(tercile_target.index, fill_value=0)).clip(lower=0)
        while deficit.sum() > max(seg_quota - chosen.sum(), 0):     # обязательные уже заняли часть квоты
            deficit[deficit.idxmax()] -= 1
        for tercile, needed in deficit[deficit > 0].items():
            candidates = seg_pool.index[(seg_pool["growth_tercile"] == tercile) & ~chosen].sort_values()
            picked = rng.choice(candidates, size=needed, replace=False)
            reasons = pd.concat([reasons, pd.Series("стратифицированный добор", index=picked)])
    return reasons.rename("reason")


def smd(sample: pd.Series, population: pd.Series) -> float:
    """Стандартизированная разница средних выборки и сети (для флагов - разница долей)."""
    sample, population = sample.dropna().astype(float), population.dropna().astype(float)
    pooled_std = np.sqrt((sample.var() + population.var()) / 2)
    return (sample.mean() - population.mean()) / pooled_std if pooled_std > 0 else 0.0


def balance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Характеристики магазинов, по которым сравнивается выборка и сеть: масштаб, средний чек, вариативность,
    рост, конкуренция, флаги проблемных групп, тип магазина и ассортимент (one-hot)."""
    return pd.concat([
        frame[["guests_median", "avg_check", "guests_cv", "growth_yoy"]],
        np.log10(frame["CompetitionDistance"]).rename("log10_competition_distance"),
        frame[["Promo2", "competitor_opened_in_train", "has_export_gap",
               "has_long_closure", "sunday_open"]].astype(int),
        pd.get_dummies(frame["StoreType"], prefix="StoreType").astype(int),
        pd.get_dummies(frame["Assortment"], prefix="Assortment").astype(int),
    ], axis=1)
