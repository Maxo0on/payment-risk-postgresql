"""Обучение моделей риска и экономическая оценка политики с двумя порогами."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class PolicyCosts:
    """Бизнес-ограничения и параметры сценария ручной проверки.

    `review_capacity` задаёт предельную долю ручных проверок во всём потоке.
    `max_fraud_loss_ratio` ограничивает остаточные потери относительно
    сценария, в котором одобряются все операции. `review_cost` и
    `false_decline_cost` задают стоимость одной проверки и одного ложного
    отклонения в единицах суммы транзакции.

    `review_fraud_catch_rate` задаёт ожидаемую долю предотвращённого ущерба
    по мошенническим операциям на проверке. `review_false_decline_rate` задаёт
    долю легитимных операций на проверке, которые будут ошибочно отклонены. `fraud_lgd` задаёт долю суммы, которая
    теряется при одобрении мошеннической операции.
    """

    review_capacity: float = 0.06
    max_fraud_loss_ratio: float = 0.35
    review_cost: float = 2.0
    false_decline_cost: float = 12.0
    review_fraud_catch_rate: float = 0.8
    review_false_decline_rate: float = 0.02
    fraud_lgd: float = 1.0

    def __post_init__(self) -> None:
        """Проверяет диапазон долей и допустимость денежных ставок.

        Доли лежат в диапазоне [0, 1]. Денежные ставки должны быть конечными
        и неотрицательными. Некорректный параметр вызывает `ValueError`.
        """
        for key in (
            "review_capacity",
            "max_fraud_loss_ratio",
            "review_fraud_catch_rate",
            "review_false_decline_rate",
            "fraud_lgd",
        ):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be between 0 and 1")
        for key in ("review_cost", "false_decline_cost"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and non-negative")


def _mature_rows(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Возвращает маску доступных меток мошенничества на момент `cutoff`.

    Строка участвует в оценке по исходу, когда `is_fraud` заполнен, а
    `label_available_at` наступил не позже среза. Индекс исходной таблицы
    сохраняется для согласованного отбора строк и оценок модели.
    """
    return frame["is_fraud"].notna() & frame["label_available_at"].le(cutoff)


def _model_view(
    frame: pd.DataFrame,
    features: list[str],
    categories: dict[str, list[str]],
    *,
    native_categories: bool,
) -> pd.DataFrame:
    """Готовит копию признаков в порядке, который используется при обучении.

    `features` задаёт порядок столбцов. В `categories` перечислены
    категориальные столбцы и значения, встречавшиеся на обучении.
    При `native_categories=True` LightGBM получает тип `Categorical`
    с фиксированным словарём. Незнакомая категория кодируется как пропуск.
    CatBoost получает строки, включая маркер `__MISSING__` для пропусков.
    Числовые признаки передаются в формате `float` с сохранением NaN.
    """
    result = frame[features].copy()
    for column in features:
        if column in categories:
            values = result[column].astype("string").fillna("__MISSING__")
            if native_categories:
                result[column] = pd.Categorical(values, categories=categories[column])
            else:
                result[column] = values.astype(str)
        else:
            result[column] = pd.to_numeric(result[column]).astype(float)
    return result


def _scores_at_top(
    observed: pd.DataFrame, scores: np.ndarray, top_share: float
) -> dict[str, float | int | None]:
    """Оценивает ранжирование на строках с доступной бинарной меткой.

    `scores` выровнен по порядку строк `observed`. ROC-AUC сравнивает
    позиции классов, а `pr_auc_ap` хранит average precision (AP). Размер
    верхнего сегмента K округляется вверх от доли `top_share` с минимумом
    в одну строку для непустой выборки. Recall@K показывает долю всех
    мошеннических операций, попавших в сегмент, а precision@K показывает
    долю мошеннических операций внутри сегмента. Для неопределённых метрик
    возвращается `None`.
    """
    labels = observed["is_fraud"].to_numpy(dtype=int)
    positive_count = int(labels.sum())
    top_k = min(len(labels), max(1, int(np.ceil(top_share * len(labels)))))
    # При равных оценках стабильная сортировка сохраняет входной порядок.
    top_positive = int(labels[np.argsort(-scores, kind="stable")[:top_k]].sum())
    return {
        "n": len(labels),
        "fraud_count": positive_count,
        "fraud_rate": float(labels.mean()) if len(labels) else None,
        "roc_auc": (
            float(roc_auc_score(labels, scores))
            if 0 < positive_count < len(labels)
            else None
        ),
        "pr_auc_ap": (
            float(average_precision_score(labels, scores)) if positive_count else None
        ),
        "top_k": top_k,
        "recall_at_top_k": top_positive / positive_count if positive_count else None,
        "precision_at_top_k": top_positive / top_k if top_k else None,
    }


def _actions(
    scores: np.ndarray, review_threshold: float, reject_threshold: float
) -> np.ndarray:
    """Возвращает действия в том же порядке, что и оценки модели.

    При `score <= review_threshold` применяется `approve`, при
    `review_threshold < score <= reject_threshold` применяется `review`,
    при `score > reject_threshold` применяется `reject`.
    """
    return np.select(
        [scores > reject_threshold, scores > review_threshold],
        ["reject", "review"],
        default="approve",
    )


def _fraud_and_legitimate(
    observed: pd.DataFrame, costs: PolicyCosts
) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает потенциальный ущерб и маску легитимных операций.

    Для каждой строки с доступной меткой ущерб при одобрении равен
    `amount * fraud_lgd` для мошеннических операций и нулю для остальных.
    Оба массива выровнены по строкам `observed`.
    """
    target = observed["is_fraud"].to_numpy(dtype=int)
    fraud_amounts = observed["amount"].to_numpy(dtype=float) * target * costs.fraud_lgd
    return fraud_amounts, target == 0


def _range_minimum(
    values: np.ndarray, left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Находит индекс минимума на каждом включительном интервале.

    `left` и `right` содержат парные границы непустых интервалов в
    `values`. Разреженная таблица хранит индексы минимумов на отрезках
    длиной в степень двойки. При равном значении возвращается больший
    индекс. Порядок ответов соответствует порядку запросов.
    """
    spans = right - left + 1
    powers = np.floor(np.log2(spans)).astype(int)
    table = [np.arange(len(values))]
    # Отрезок длиной 2**level состоит из двух соседних половин из предыдущего уровня.
    for level in range(1, int(powers.max()) + 1):
        half = 1 << (level - 1)
        previous = table[-1]
        first, second = previous[:-half], previous[half:]
        table.append(np.where(values[first] < values[second], first, second))
    best = np.empty_like(left)
    for level, indices in enumerate(table):
        use = powers == level
        # Два отрезка такой длины от левой и правой границ покрывают весь запрос.
        first = indices[left[use]]
        second = indices[right[use] - (1 << level) + 1]
        best[use] = np.where(values[first] < values[second], first, second)
    return best


def _optimize_thresholds(
    cohort: pd.DataFrame, mature: pd.Series, scores: np.ndarray, costs: PolicyCosts
) -> tuple[float, float]:
    """Возвращает пару порогов с минимальной стоимостью на валидации.

    Учитываются все варианты маршрутизации, достижимые двумя порогами.
    Операции с одинаковым `score` получают одно действие. Лимит ручной
    проверки считается на всём потоке `cohort`. Ограничение ущерба и
    денежная функция используют строки из маски `mature`. При равной
    стоимости выбирается больший `reject_threshold`, а внутри одной
    границы отклонения — больший `review_threshold`.
    """
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    # Границы соответствуют концам групп равных оценок. Порог под
    # минимальной оценкой позволяет задать пустой префикс.
    ends = np.r_[np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    count = np.r_[0, ends]
    thresholds = np.r_[np.nextafter(ordered_scores[0], -np.inf), ordered_scores[ends - 1]]
    known = mature.to_numpy()
    fraud_amount, is_legitimate = _fraud_and_legitimate(cohort.loc[mature], costs)
    fraud_values = np.zeros(len(cohort))
    legitimate = np.zeros(len(cohort), dtype=bool)
    fraud_values[known] = fraud_amount
    legitimate[known] = is_legitimate
    # Префиксы дают суммы и количества по произвольному диапазону
    # операций как разность значений в его границах.
    fraud = np.r_[0.0, np.cumsum(fraud_values[order])[ends - 1]]
    legit = np.r_[0, np.cumsum(legitimate[order])[ends - 1]]
    mature_count = np.r_[0, np.cumsum(known[order])[ends - 1]]

    # Для заданной верхней границы j число проверок count[j] − count[i]
    # задаёт нижнюю границу i. Остаточный ущерб равен сумме ущерба при
    # одобрении fraud[i] и неперехваченной доли (1 − catch) в интервале
    # fraud[j] − fraud[i]. Он задаёт верхнюю границу i.
    max_review_count = int(np.floor(costs.review_capacity * len(scores) + 1e-9))
    loss_limit = costs.max_fraud_loss_ratio * fraud[-1]
    catch = costs.review_fraud_catch_rate
    missed_review_fraud = (1 - catch) * fraud
    left = np.searchsorted(count, count - max_review_count, side="left")
    right = np.minimum(
        np.arange(len(count)),
        np.searchsorted(
            catch * fraud, loss_limit - missed_review_fraud + 1e-9, side="right"
        ) - 1,
    )
    feasible = left <= right
    if not feasible.any():
        raise RuntimeError("No threshold policy satisfies the validation constraints")

    # Стоимость разбивается на независимые слагаемые от i и j. Запрос
    # минимума на допустимом интервале даёт лучшую нижнюю границу для
    # каждой верхней. Это позволяет обойтись без перебора всех пар.
    review_term = costs.review_cost * mature_count
    false_decline_review_term = (
        costs.false_decline_cost * costs.review_false_decline_rate * legit
    )
    cost_i = catch * fraud - review_term - false_decline_review_term
    cost_j = (
        missed_review_fraud
        + review_term
        + false_decline_review_term
        + costs.false_decline_cost * (legit[-1] - legit)
    )
    feasible_j = np.flatnonzero(feasible)
    best_i = _range_minimum(cost_i, left[feasible], right[feasible])
    losses = cost_i[best_i] + cost_j[feasible]
    # При точном совпадении стоимости берём правую границу отклонения.
    best_offset = len(losses) - 1 - int(np.argmin(losses[::-1]))
    return float(thresholds[best_i[best_offset]]), float(
        thresholds[feasible_j[best_offset]]
    )


def _policy_outcomes(
    cohort: pd.DataFrame,
    mature: pd.Series,
    actions: np.ndarray,
    costs: PolicyCosts,
) -> dict[str, float | int | bool | None]:
    """Считает операционную загрузку и смоделированную экономику политики.

    `actions` выровнен по всему `cohort`, поэтому назначения и доля
    проверок учитывают все входящие операции. Денежные результаты,
    включая плату за проверку, считаются на единой выборке `mature`.
    Остаточный ущерб складывается из потерь на одобренных операциях
    и доли мошеннических операций, пропущенных ручной проверкой.
    Стоимость ложных отклонений учитывает все отклонённые легитимные
    операции и ожидаемую долю ошибок при ручной проверке. Результат
    содержит количества, доли, затраты и флаги соблюдения лимитов.
    """
    observed = cohort.loc[mature]
    observed_actions = actions[mature.to_numpy()]
    fraud_amounts, legitimate = _fraud_and_legitimate(observed, costs)
    reviewed = observed_actions == "review"
    rejected = observed_actions == "reject"
    approved = observed_actions == "approve"

    gross_fraud_loss = float(fraud_amounts.sum())
    fraud_loss = float(
        fraud_amounts[approved].sum()
        + (1 - costs.review_fraud_catch_rate) * fraud_amounts[reviewed].sum()
    )
    auto_false_declines = int(legitimate[rejected].sum())
    expected_review_false_declines = float(
        costs.review_false_decline_rate * legitimate[reviewed].sum()
    )
    expected_false_declines = auto_false_declines + expected_review_false_declines
    review_cost = float(reviewed.sum() * costs.review_cost)
    false_decline_cost = expected_false_declines * costs.false_decline_cost
    review_count = int(np.count_nonzero(actions == "review"))
    review_rate = review_count / len(cohort)
    fraud_loss_ratio = fraud_loss / gross_fraud_loss if gross_fraud_loss else None
    legitimate_count = int(legitimate.sum())
    return {
        "transactions": len(cohort),
        "mature_transactions": len(observed),
        "mature_review_count": int(reviewed.sum()),
        "approve_count": int(np.count_nonzero(actions == "approve")),
        "review_count": review_count,
        "reject_count": int(np.count_nonzero(actions == "reject")),
        "review_rate": review_rate,
        "gross_fraud_loss": gross_fraud_loss,
        "fraud_loss": fraud_loss,
        "prevented_fraud_loss": gross_fraud_loss - fraud_loss,
        "fraud_loss_ratio": fraud_loss_ratio,
        "fraud_amount_approved": float(fraud_amounts[approved].sum()),
        "fraud_amount_reviewed": float(fraud_amounts[reviewed].sum()),
        "fraud_amount_rejected": float(fraud_amounts[rejected].sum()),
        "legitimate_mature_count": legitimate_count,
        "auto_false_decline_count": auto_false_declines,
        "review_false_decline_expected": expected_review_false_declines,
        "false_decline_count_expected": expected_false_declines,
        "false_decline_rate_expected": (
            expected_false_declines / legitimate_count if legitimate_count else None
        ),
        "review_cost": review_cost,
        "false_decline_cost": false_decline_cost,
        "total_cost": fraud_loss + review_cost + false_decline_cost,
        "review_limit_met": review_rate <= costs.review_capacity + 1e-12,
        "fraud_loss_limit_met": fraud_loss
        <= costs.max_fraud_loss_ratio * gross_fraud_loss + 1e-9,
    }


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    """Возвращает временную отметку в UTC для сравнения со срезами витрины."""
    return pd.to_datetime(value, utc=True)


def run_risk(
    transactions: pd.DataFrame,
    out: Path,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    observation_end: pd.Timestamp,
    review_capacity: float = 0.06,
    max_fraud_loss_ratio: float = 0.35,
    review_cost: float = 2.0,
    false_decline_cost: float = 12.0,
    review_fraud_catch_rate: float = 0.8,
    review_false_decline_rate: float = 0.02,
    fraud_lgd: float = 1.0,
    seed: int = 42,
) -> dict:
    """Обучает две модели на прошлом и проверяет выбранную политику на тесте.

    Витрина `transactions` содержит `transaction_id`, `event_ts`,
    `amount`, `is_fraud`, `label_available_at` и признаки `f_*`.
    `train_end` и `validation_end` задают исключающие правые границы
    периодов по времени операции. На тесте берутся строки витрины с временем
    операции до `observation_end`. Метка учитывается, если она известна
    к этому срезу, даже если сам платёж произошёл гораздо раньше. Параметры сценария
    описаны в `PolicyCosts`. `seed` фиксирует случайность обучения.

    Сохраняет в `out` обе модели со схемой признаков, таблицы метрик,
    тестовые значения score и действия, а также JSON со сводкой.
    Возвращает эту сводку с моделью, порогами, метриками и валидационной
    долей проверок для прогноза нагрузки.
    """
    costs = PolicyCosts(
        review_capacity=review_capacity,
        max_fraud_loss_ratio=max_fraud_loss_ratio,
        review_cost=review_cost,
        false_decline_cost=false_decline_cost,
        review_fraud_catch_rate=review_fraud_catch_rate,
        review_false_decline_rate=review_false_decline_rate,
        fraud_lgd=fraud_lgd,
    )
    train_end, validation_end, observation_end = map(
        _utc, (train_end, validation_end, observation_end)
    )
    if not train_end < validation_end <= observation_end:
        raise ValueError("Dates must satisfy train_end < validation_end <= observation_end")
    data = transactions.copy()
    data["event_ts"] = pd.to_datetime(data["event_ts"], utc=True)
    data["label_available_at"] = pd.to_datetime(data["label_available_at"], utc=True)
    # Хронологический порядок нужен CatBoost с has_time и фиксирует
    # воспроизводимый порядок строк с равными оценками в метриках.
    data.sort_values(["event_ts", "transaction_id"], inplace=True, ignore_index=True)
    amounts = data["amount"].to_numpy(dtype=float)
    if not np.isfinite(amounts).all() or (amounts < 0).any():
        raise ValueError("Risk policy amounts must be finite and non-negative")
    if not data["is_fraud"].dropna().isin([0, 1]).all():
        raise ValueError("is_fraud must contain binary labels or missing values")

    train_window = data.loc[data["event_ts"].lt(train_end)]
    validation_window = data.loc[
        data["event_ts"].ge(train_end) & data["event_ts"].lt(validation_end)
    ]
    test = data.loc[
        data["event_ts"].ge(validation_end) & data["event_ts"].lt(observation_end)
    ]
    # Проверяем доступность метки на правой границе каждого периода.
    # Все валидационные и тестовые строки остаются в расчёте загрузки.
    train = train_window.loc[_mature_rows(train_window, train_end)]
    validation_mature = _mature_rows(validation_window, validation_end)
    validation = validation_window.loc[validation_mature]
    test_mature = _mature_rows(test, observation_end)
    if train["is_fraud"].nunique() < 2 or validation["is_fraud"].nunique() < 2:
        raise ValueError("Train and validation each require mature fraud and legitimate labels")
    if test.empty:
        raise ValueError("The test date range contains no transactions")

    # Только входы, размеченные SQL-витриной как признаки. Словари
    # категорий фиксируются по обучающей выборке.
    features = [column for column in data if column.startswith("f_")]
    if not features:
        raise ValueError("The input mart must provide features prefixed with f_")
    categories = {
        column: sorted(set(train[column].astype("string").fillna("__MISSING__")))
        for column in features
        if not is_numeric_dtype(data[column])
    }
    target = train["is_fraud"].astype(int)
    models = {
        "CatBoost": CatBoostClassifier(
            iterations=350,
            depth=7,
            learning_rate=0.06,
            l2_leaf_reg=5,
            random_seed=seed,
            has_time=True,
            thread_count=2,
            allow_writing_files=False,
            verbose=False,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=350,
            num_leaves=31,
            learning_rate=0.05,
            min_child_samples=35,
            reg_lambda=3.0,
            deterministic=True,
            force_col_wise=True,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
        ),
    }

    metrics = []
    candidates = {}
    for name, model in models.items():
        native_categories = name == "LightGBM"
        train_view = _model_view(
            train, features, categories, native_categories=native_categories
        )
        validation_view = _model_view(
            validation_window, features, categories, native_categories=native_categories
        )
        fit_options = {"cat_features": list(categories)} if name == "CatBoost" else {}
        model.fit(train_view, target, **fit_options)
        # Оценка положительного класса задаёт порядок риска. Пороговая
        # стоимость считается по наблюдаемым исходам валидации.
        val_score = model.predict_proba(validation_view)[:, 1]
        val_metrics = _scores_at_top(
            validation, val_score[validation_mature.to_numpy()], costs.review_capacity
        )
        metrics.append({"model": name, "split": "validation", **val_metrics})
        thresholds = _optimize_thresholds(
            validation_window, validation_mature, val_score, costs
        )
        val_actions = _actions(val_score, *thresholds)
        candidates[name] = {
            "review_threshold": thresholds[0],
            "reject_threshold": thresholds[1],
            "pr_auc_ap": val_metrics["pr_auc_ap"],
            **_policy_outcomes(validation_window, validation_mature, val_actions, costs),
        }

    # Сначала минимизируем стоимость на валидации. При равной стоимости
    # выбираем более высокую AP. Тестовые оценки считаются после выбора.
    selected = min(
        candidates,
        key=lambda name: (
            candidates[name]["total_cost"],
            -candidates[name]["pr_auc_ap"],
        ),
    )
    chosen = candidates[selected]
    test_scores = {}
    for name, model in models.items():
        test_view = _model_view(
            test, features, categories, native_categories=name == "LightGBM"
        )
        test_scores[name] = model.predict_proba(test_view)[:, 1]
        metrics.append(
            {
                "model": name,
                "split": "test",
                **_scores_at_top(
                    test.loc[test_mature],
                    test_scores[name][test_mature.to_numpy()],
                    costs.review_capacity,
                ),
            }
        )
    test_score = test_scores[selected]
    test_actions = _actions(
        test_score, chosen["review_threshold"], chosen["reject_threshold"]
    )
    # Сравниваем итоговую политику с одобрением всех операций на тех же
    # данных. Флаги теста показывают, переносится ли соблюдение лимитов.
    policy_metrics = [
        {
            "split": "validation",
            "policy": "approve_all",
            **_policy_outcomes(
                validation_window,
                validation_mature,
                np.full(len(validation_window), "approve"),
                costs,
            ),
        },
        {"split": "validation", "policy": "optimized", **chosen},
        {
            "split": "test",
            "policy": "approve_all",
            **_policy_outcomes(test, test_mature, np.full(len(test), "approve"), costs),
        },
        {
            "split": "test",
            "policy": "optimized",
            **_policy_outcomes(test, test_mature, test_actions, costs),
        },
    ]
    passthrough = [
        column
        for column in ("merchant_id", "merchant_category", "f_merchant_category")
        if column in test
    ]
    scored = test[["transaction_id", "event_ts", "amount", "is_fraud", *passthrough]].copy()
    # Операционное действие требуется каждой строке. Исход скрывается
    # для меток, которые ещё не были доступны на момент среза.
    scored["is_fraud"] = scored["is_fraud"].where(test_mature).astype("Int8")
    scored["label_mature"] = test_mature
    scored["score"] = test_score
    scored["action"] = test_actions

    out.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out / "test_scored.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "risk_metrics.csv", index=False)
    pd.DataFrame(policy_metrics).to_csv(out / "policy_metrics.csv", index=False)
    pd.DataFrame(
        [{"model": name, **outcomes} for name, outcomes in candidates.items()]
    ).to_csv(out / "validation_policy_candidates.csv", index=False)

    # Для CatBoost используем штатную важность признака, для LightGBM —
    # суммарный выигрыш от разбиений по признаку.
    importance = (
        models[selected].feature_importances_
        if selected == "CatBoost"
        else models[selected].booster_.feature_importance(importance_type="gain")
    )
    pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(out / "feature_importance.csv", index=False)

    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    models["CatBoost"].save_model(str(model_dir / "catboost.cbm"))
    models["LightGBM"].booster_.save_model(str(model_dir / "lightgbm.txt"))
    (model_dir / "feature_schema.json").write_text(
        json.dumps({"features": features, "categories": categories}, indent=2),
        encoding="utf-8",
    )
    summary = {
        "selected_model": selected,
        "thresholds": {
            "review_threshold": chosen["review_threshold"],
            "reject_threshold": chosen["reject_threshold"],
        },
        "validation_reference_review_rate": chosen["review_rate"],
        "costs": asdict(costs),
        "cutoffs": {
            "train_end": train_end.isoformat(),
            "validation_end": validation_end.isoformat(),
            "observation_end": observation_end.isoformat(),
        },
        "label_counts": {
            "train": {
                "transactions": len(train_window),
                "mature": len(train),
                "fraud": int(train["is_fraud"].sum()),
            },
            "validation": {
                "transactions": len(validation_window),
                "mature": len(validation),
                "fraud": int(validation["is_fraud"].sum()),
            },
            "test": {
                "transactions": len(test),
                "mature": int(test_mature.sum()),
                "fraud": int(test.loc[test_mature, "is_fraud"].sum()),
            },
        },
        "model_selection": "minimum validation policy cost, then maximum validation AP",
        "metrics": metrics,
        "policy_metrics": policy_metrics,
        "validation_policy_candidates": candidates,
    }
    (out / "risk_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    return summary
