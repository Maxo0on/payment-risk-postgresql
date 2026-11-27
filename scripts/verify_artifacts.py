"""Проверяет результаты проекта по витрине Parquet и сохранённым моделям.

Укажите каталог результатов, содержащий CSV-таблицы, витрину и models/.
Скрипт загружает обе модели риска, пересчитывает метрики, решения и расходы.
Затем он сверяет качество почасовых прогнозов с витриной фактов.
Расхождение с сохранёнными результатами приводит к ненулевому коду выхода.
"""

import argparse
import json
import math
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def read_json(path):
    """Читает параметры запуска или модели из JSON в UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def predict(frame, native, feature_schema):
    """Восстанавливает вход модели по сохранённой схеме и возвращает score."""
    columns = {}
    categories = feature_schema["categories"]
    for name in feature_schema["features"]:
        column = frame[name]
        if name not in categories:
            columns[name] = pd.to_numeric(column).astype(float)
            continue
        strings = column.astype("string").fillna("__MISSING__")
        columns[name] = (
            strings.astype(str)
            if isinstance(native, CatBoostClassifier)
            else pd.Categorical(strings, categories=categories[name])
        )
    features = pd.DataFrame(columns, index=frame.index)
    if isinstance(native, CatBoostClassifier):
        return native.predict_proba(features, thread_count=2)[:, 1]
    return native.predict(features, num_threads=2)


def label_mask(frame, cutoff):
    """Отбирает только метки, известные к моменту оценки."""
    return frame.is_fraud.notna() & frame.label_available_at.le(cutoff)


def check_values(actual, published, context):
    """Сверяет значения пересчитанных показателей с одной строкой CSV."""
    for name, value in actual.items():
        expected = published[name]
        if value is None:
            if not pd.isna(expected):
                raise AssertionError(f"{context}/{name}: ожидался пропуск, получено {expected}")
        elif isinstance(value, (int, bool, np.integer, np.bool_)):
            if value != expected:
                raise AssertionError(f"{context}/{name}: пересчитано {value}, сохранено {expected}")
        else:
            np.testing.assert_allclose(
                value, expected, rtol=1e-10, atol=1e-9, err_msg=f"{context}/{name}"
            )


def metrics(labels, scores, review_share):
    """Пересчитывает метрики ранжирования по строкам с доступными метками."""
    labels = labels.astype(int)
    positives = int(labels.sum())
    k = min(len(labels), max(1, math.ceil(review_share * len(labels))))
    # При равном score сохраняется хронологический порядок строк.
    ranked = pd.DataFrame({"score": scores, "label": labels.to_numpy()})
    top = ranked.sort_values("score", ascending=False, kind="stable").head(k)
    top_positives = int(top.label.sum())
    return {
        "n": len(labels),
        "fraud_count": positives,
        "fraud_rate": labels.mean() if len(labels) else None,
        "roc_auc": roc_auc_score(labels, scores) if 0 < positives < len(labels) else None,
        "pr_auc_ap": average_precision_score(labels, scores) if positives else None,
        "top_k": k,
        "recall_at_top_k": top_positives / positives if positives else None,
        "precision_at_top_k": top_positives / k if k else None,
    }


def decisions(scores, thresholds):
    """Применяет пороги с теми же включёнными границами, что и в отчёте."""
    review, reject = thresholds["review_threshold"], thresholds["reject_threshold"]
    if review > reject:
        raise AssertionError("Порог ручной проверки выше порога отклонения")
    actions = np.full(len(scores), "approve", dtype="<U7")
    actions[(scores > review) & (scores <= reject)] = "review"
    actions[scores > reject] = "reject"
    return actions


def money_and_routing(frame, mature, actions, terms):
    """Считает назначения по всему потоку, а расходы по созревшим меткам."""
    case = frame.loc[mature, ["amount", "is_fraud"]].assign(action=actions[mature.to_numpy()])
    fraud = case.loc[case.is_fraud.eq(1)]
    legit = case.loc[case.is_fraud.eq(0)]
    fraud_by_action = fraud.groupby("action").amount.sum().mul(terms["fraud_lgd"])
    after_review = 1 - terms["review_fraud_catch_rate"]
    residual = (
        fraud_by_action.get("approve", 0.0)
        + after_review * fraud_by_action.get("review", 0.0)
    )
    gross = float(fraud_by_action.sum())
    auto_false = int(legit.action.eq("reject").sum())
    review_false = float(legit.action.eq("review").sum() * terms["review_false_decline_rate"])
    false_cost = (auto_false + review_false) * terms["false_decline_cost"]
    mature_reviews = int(case.action.eq("review").sum())
    review_cost = mature_reviews * terms["review_cost"]
    n_reviews = int((actions == "review").sum())
    review_rate = n_reviews / len(frame)
    return {
        "transactions": len(frame),
        "mature_transactions": len(case),
        "mature_review_count": mature_reviews,
        "approve_count": int((actions == "approve").sum()),
        "review_count": n_reviews,
        "reject_count": int((actions == "reject").sum()),
        "review_rate": review_rate,
        "gross_fraud_loss": gross,
        "fraud_loss": residual,
        "prevented_fraud_loss": gross - residual,
        "fraud_loss_ratio": residual / gross if gross else None,
        "fraud_amount_approved": fraud_by_action.get("approve", 0.0),
        "fraud_amount_reviewed": fraud_by_action.get("review", 0.0),
        "fraud_amount_rejected": fraud_by_action.get("reject", 0.0),
        "legitimate_mature_count": len(legit),
        "auto_false_decline_count": auto_false,
        "review_false_decline_expected": review_false,
        "false_decline_count_expected": auto_false + review_false,
        "false_decline_rate_expected": (auto_false + review_false) / len(legit) if len(legit) else None,
        "review_cost": review_cost,
        "false_decline_cost": false_cost,
        "total_cost": residual + review_cost + false_cost,
        "review_limit_met": review_rate <= terms["review_capacity"] + 1e-12,
        "fraud_loss_limit_met": residual <= terms["max_fraud_loss_ratio"] * gross + 1e-9,
    }


def verify_risk(directory):
    """Сверяет таблицы метрик и решения по каждой тестовой операции."""
    summary = read_json(directory / "risk_summary.json")
    model_dir = directory / "models"
    schema = read_json(model_dir / "feature_schema.json")
    catboost = CatBoostClassifier().load_model(str(model_dir / "catboost.cbm"))
    lightgbm = lgb.Booster(model_file=str(model_dir / "lightgbm.txt"))
    if catboost.feature_names_ != schema["features"]:
        raise AssertionError("Признаки CatBoost отличаются от сохранённой схемы")
    if lightgbm.feature_name() != schema["features"]:
        raise AssertionError("Признаки LightGBM отличаются от сохранённой схемы")
    models = {"CatBoost": catboost, "LightGBM": lightgbm}

    data = pd.read_parquet(directory / "transaction_mart.parquet")
    if not data.transaction_id.is_unique:
        raise AssertionError("Витрина содержит повторяющиеся transaction_id")
    data = data.sort_values(["event_ts", "transaction_id"]).reset_index(drop=True)
    train_end, validation_end, observation_end = (
        pd.Timestamp(summary["cutoffs"][key])
        for key in ("train_end", "validation_end", "observation_end")
    )
    periods = {
        "validation": (train_end, validation_end),
        "test": (validation_end, observation_end),
    }
    risk_csv = pd.read_csv(directory / "risk_metrics.csv").set_index(
        ["model", "split"], verify_integrity=True
    )
    policy_csv = pd.read_csv(directory / "policy_metrics.csv").set_index(
        ["split", "policy"], verify_integrity=True
    )
    if len(risk_csv) != len(models) * len(periods):
        raise AssertionError("Число строк в risk_metrics.csv отличается от ожидаемого")
    if len(policy_csv) != 2 * len(periods):
        raise AssertionError("Число строк в policy_metrics.csv отличается от ожидаемого")
    terms = summary["costs"]
    selected = summary["selected_model"]
    for split, (start, end) in periods.items():
        frame = data.loc[data.event_ts.ge(start) & data.event_ts.lt(end)]
        mature = label_mask(frame, end)
        labels = frame.loc[mature, "is_fraud"]
        predictions = {name: predict(frame, model, schema) for name, model in models.items()}
        for name, scores in predictions.items():
            evaluated = metrics(labels, scores[mature.to_numpy()], terms["review_capacity"])
            check_values(evaluated, risk_csv.loc[(name, split)], f"{split}/{name}")
            print(
                f"{split:10s} {name:8s} "
                f"ROC-AUC={evaluated['roc_auc']!r}, AP={evaluated['pr_auc_ap']!r}"
            )
        scores = predictions[selected]
        actions = decisions(scores, summary["thresholds"])
        policies = {"approve_all": np.repeat("approve", len(frame)), "optimized": actions}
        for policy, routing in policies.items():
            checked = money_and_routing(frame, mature, routing, terms)
            check_values(checked, policy_csv.loc[(split, policy)], f"{split}/{policy}")
            print(
                f"{split:10s} {policy:11s} approve/review/reject="
                f"{checked['approve_count']}/{checked['review_count']}/{checked['reject_count']}, "
                f"потери от мошенничества={checked['fraud_loss']:.2f}, "
                f"общие расходы={checked['total_cost']:.2f}"
            )
        if split == "test":
            scored = pd.read_csv(
                directory / "test_scored.csv", dtype={"transaction_id": "string"},
                float_precision="round_trip", keep_default_na=False, na_values={"is_fraud": [""]},
            )
            pd.testing.assert_series_equal(
                frame.transaction_id.astype("string").reset_index(drop=True), scored.transaction_id
            )
            np.testing.assert_allclose(scores, scored.score, rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(actions, scored.action)
            np.testing.assert_array_equal(mature, scored.label_mature)
            np.testing.assert_array_equal(
                frame.is_fraud.where(mature).to_numpy(dtype=float, na_value=np.nan), scored.is_fraud
            )
            print(f"{selected}. Сверены score, действия и доступность меток для {len(scored)} операций.")
    print(f"Сверены строки метрик ранжирования ({len(risk_csv)}) и политики ({len(policy_csv)}).")


def verify_forecast_metrics(directory: Path) -> None:
    """Пересчитывает метрики двух методов, проверяет часовые факты и выбор модели."""
    targets = ("n_transactions", "total_amount", "risk_events")
    methods = ("sarimax", "lag_boosting")
    summary = read_json(directory / "forecast_summary.json")
    info = read_json(directory / "run_info.json")
    marts = pd.read_parquet(directory / "hourly_mart.parquet")
    totals = marts.groupby(pd.to_datetime(marts["hour"], utc=True))[list(targets)].sum()
    parts = {}
    cutoffs = {"validation": ("train_end", "validation_end"), "test": ("validation_end", "end")}
    for split, (start, end) in cutoffs.items():
        rows = pd.read_csv(directory / f"forecast_{split}.csv")
        hours = pd.date_range(info[start], info[end], inclusive="left", freq="h")
        forecast_hours = pd.DatetimeIndex(pd.to_datetime(rows["hour"], utc=True))
        if not forecast_hours.equals(hours):
            raise AssertionError(f"{split}: пропущены или смещены часы прогноза")
        facts = rows[[f"actual_{target}" for target in targets]].to_numpy(float)
        np.testing.assert_allclose(facts, totals.reindex(hours, fill_value=0), rtol=1e-10, atol=1e-9)
        parts[split] = rows
    parts["validation_selection"] = parts["validation"].iloc[:summary["selection_hours"]]

    metrics = pd.read_csv(directory / "forecast_metrics.csv")
    keys = ["target", "method", "split"]
    expected = {(target, method, split) for target in targets for method in methods for split in parts}
    combinations = set(metrics[keys].itertuples(index=False, name=None))
    if len(metrics) != len(expected) or combinations != expected:
        raise AssertionError("В таблице метрик отсутствуют или повторяются результаты прогноза")
    selection_mae = {}
    for row in metrics.itertuples(index=False):
        rows = parts[row.split]
        fact = rows[f"actual_{row.target}"].to_numpy(float)
        error = fact - rows[f"{row.method}_{row.target}"].to_numpy(float)
        abs_error = np.abs(error)
        volume = float(np.abs(fact).sum())
        mae = float(abs_error.mean())
        wape = float(abs_error.sum()) / volume if volume else np.nan
        calculated = [mae, np.sqrt(np.mean(error**2)), wape]
        np.testing.assert_allclose(
            [row.mae, row.rmse, row.wape], calculated, rtol=1e-10, atol=1e-9, equal_nan=True,
            err_msg=f"{row.target}/{row.method}/{row.split}",
        )
        if row.n_hours != len(rows):
            raise AssertionError(f"{row.split}: число часов отличается от таблицы прогнозов")
        selected = summary["selected_models"][row.target] == row.method
        if row.selected != selected:
            raise AssertionError(f"{row.target}: расходится выбор метода в CSV и JSON")
        if row.split == "validation_selection":
            selection_mae[row.target, row.method] = mae
    for target in targets:
        winner = min(methods, key=lambda method: selection_mae[target, method])
        if summary["selected_models"][target] != winner:
            raise AssertionError(f"{target}: выбранный метод отличается от лучшего по MAE")
    print(f"Сверены часовые факты и все {len(metrics)} строк метрик прогноза.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Каталог с результатами проекта")
    directory = parser.parse_args().directory
    verify_risk(directory)
    verify_forecast_metrics(directory)
