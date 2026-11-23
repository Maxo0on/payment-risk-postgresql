"""Почасовой прогноз нагрузки, поиск аномалий и оценка ручной проверки.

Часовая витрина по категориям поставляет фактические ряды числа операций,
оборота и операционных риск-сигналов. SARIMAX и бустинг обучаются на одном
отрезке истории и выпускают прогнозы блоками до 24 часов. На валидации
выбирается модель для каждой цели и фиксируются пороги тревоги. На тесте
оцениваются прогнозы, а аномалии получают пояснения из истории категорий.
Число ручных проверок по решениям риск-политики на тестовых операциях
сопоставляется с прогнозом нагрузки.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from threadpoolctl import threadpool_limits


TARGETS = ("n_transactions", "total_amount", "risk_events")
METHODS = ("sarimax", "lag_boosting")
HORIZON = 24
# Лаг 168 часов соответствует тому же часу предыдущей недели.
LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)
ROLLING_MEANS = (3, 6, 24, 168)
ROLLING_STDS = (6, 24, 168)
ALERT_COLUMNS = (
    "hour",
    "primary_metric",
    "flagged_metrics",
    "max_robust_z",
    "observed",
    "forecast",
    "residual",
    "alert_residual_threshold",
    "merchant_category",
    "category_observed",
    "category_same_hour_weekly_mean",
    "category_excess",
    "category_risk_event_rate",
    "category_decline_rate",
    "possible_explanation",
)


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    """Приводит границу выборки к UTC для сравнения с часовым индексом.

    Метка с часовым поясом переводится в UTC с сохранением момента времени.
    Метка без пояса интерпретируется как UTC.
    """
    return pd.to_datetime(value, utc=True)


def _calendar(index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает матрицы календарных признаков для двух моделей.

    Строка соответствует часу из `index`. SARIMAX получает флаг выходного
    и пары синуса и косинуса для четырёх суточных и трёх недельных гармоник.
    Бустинг дополнительно получает числовые час суток и день недели.
    Календарь вычислим для любого будущего часа в момент выпуска прогноза.
    """
    elapsed = np.asarray(
        (index - pd.Timestamp("2000-01-03", tz="UTC")) / pd.Timedelta(hours=1)
    )
    weekend = np.asarray(index.dayofweek >= 5, dtype=float)
    harmonics = [weekend]
    for period, count in ((24, 4), (168, 3)):
        for harmonic in range(1, count + 1):
            phase = 2 * np.pi * harmonic * (elapsed % period) / period
            harmonics.extend((np.sin(phase), np.cos(phase)))
    classic = np.column_stack(harmonics)
    machine_learning = np.column_stack([index.hour, index.dayofweek, classic])
    return classic, machine_learning


def _blocks(n_train: int, n_validation_end: int, n_observed: int) -> list[tuple[int, int]]:
    """Задаёт полуоткрытые интервалы позиций `[start, end)` для выпусков прогноза.

    Блоки валидации и теста нарезаются отдельно, поэтому граничный блок
    каждого периода может быть короче 24 часов. Последний блок соответствует
    24 будущим часам, начиная с первой позиции после наблюдаемого ряда.
    """
    blocks = []
    for phase_start, phase_end in ((n_train, n_validation_end), (n_validation_end, n_observed)):
        for start in range(phase_start, phase_end, HORIZON):
            blocks.append((start, min(start + HORIZON, phase_end)))
    blocks.append((n_observed, n_observed + HORIZON))
    return blocks


def _classic(
    observed: np.ndarray,
    calendar: np.ndarray,
    n_train: int,
    blocks: list[tuple[int, int]],
) -> tuple[np.ndarray, str]:
    """Строит прогноз SARIMAX для одного наблюдаемого ряда.

    Модель AR(1) с константой и календарными регрессорами оценивает параметры
    на первых `n_train` точках. Масштабирование на среднее обучения улучшает
    численную устойчивость для рядов разных порядков величины. Факты блока
    обновляют отфильтрованное состояние после выпуска всего блока с уже
    оценёнными параметрами.

    Возвращает массив длины календаря с прогнозом в исходных единицах и
    строковый статус сходимости. Для обучающего периода в массиве стоит
    `NaN`, для постоянного обучающего ряда выдаётся его константа.
    """
    training = observed[:n_train]
    predictions = np.full(len(calendar), np.nan)
    if np.ptp(training) == 0:
        predictions[n_train:] = training[0]
        return predictions, "constant_training_series"

    scale = max(float(training.mean()), 1.0)
    model = SARIMAX(
        training / scale,
        exog=calendar[:n_train],
        order=(1, 0, 0),
        trend="c",
        enforce_stationarity=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        result = model.fit(disp=False, maxiter=120)

    fit_status = "converged"
    if any(isinstance(warning.message, ConvergenceWarning) for warning in caught):
        fit_status = "optimizer_did_not_converge"
    for start, end in blocks:
        exog = calendar[start:end]
        values = scale * np.asarray(result.forecast(steps=end - start, exog=exog))
        predictions[start:end] = np.maximum(0, values)
        # Обновляем состояние по фактам только после прогноза всего блока.
        if end <= len(observed):
            result = result.extend(observed[start:end] / scale, exog=exog)
    return predictions, fit_status


def _lag_features(history: list[float], calendar: np.ndarray) -> list[float]:
    """Собирает одну строку признаков для очередного прогнозируемого часа.

    Последнее значение `history` относится к предыдущему часу. Порядок
    признаков фиксирован: календарь, заданные лаги, средние по окнам и
    стандартные отклонения по окнам. В обучении история состоит из фактов,
    при прогнозировании в неё по мере продвижения добавляются оценки модели.
    """
    recent = np.asarray(history[-max(LAGS):], dtype=float)
    lags = [float(recent[-lag]) for lag in LAGS]
    means = [float(recent[-width:].mean()) for width in ROLLING_MEANS]
    stds = [float(recent[-width:].std()) for width in ROLLING_STDS]
    return [*calendar, *lags, *means, *stds]


def _boosting(
    observed: np.ndarray,
    calendar: np.ndarray,
    n_train: int,
    blocks: list[tuple[int, int]],
) -> np.ndarray:
    """Обучает гистограммный градиентный бустинг и прогнозирует ряды рекурсивно.

    При обучении каждая строка признаков использует факты строго ранее
    целевой позиции, поэтому первые 168 наблюдений служат историей для лагов.
    На старте блока доступна фактическая история до `start`. Каждая оценка
    становится частью входной истории для следующего часа этого блока.

    Возвращает массив длины календаря в единицах исходной цели. Позиции
    обучения остаются `NaN`, отрицательные прогнозы ограничиваются нулём.
    """
    training = observed[:n_train].tolist()
    first = max(LAGS)
    features = np.asarray(
        [_lag_features(training[position - first:position], calendar[position])
         for position in range(first, n_train)]
    )
    scale = max(float(np.mean(training)), 1.0)
    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.07,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )
    model.fit(features, observed[first:n_train] / scale)
    predictions = np.full(len(calendar), np.nan)
    for start, end in blocks:
        history = observed[start - first:start].tolist()
        for position in range(start, end):
            features = _lag_features(history, calendar[position])
            forecast = max(0.0, float(model.predict([features])[0]) * scale)
            predictions[position] = forecast
            history.append(forecast)
    return predictions


def _metrics(observed: np.ndarray, forecast: np.ndarray) -> dict[str, float]:
    """Считает MAE, RMSE и WAPE на совпадающих по часам массивах.

    MAE и RMSE выражены в единицах цели. WAPE равен сумме абсолютных
    ошибок, делённой на сумму абсолютных фактов, и хранится как доля.
    При нулевой сумме фактов WAPE имеет значение `NaN`.
    """
    error = observed - forecast
    total = float(np.abs(observed).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "wape": float(np.abs(error).sum()) / total if total else np.nan,
    }


def _alert_calibration(residuals: np.ndarray) -> tuple[float, float, float]:
    """Возвращает центр, масштаб и односторонний порог для z-оценки остатков.

    Остаток — факт минус прогноз на калибровочной части валидации.
    Медиана и масштаб 1,4826 × MAD устойчивы к отдельным всплескам. При
    нулевом MAD используется стандартное отклонение с нижней границей 1.
    Порог берётся как максимум 3,5 и эмпирического квантиля уровня
    1 − 0,05 / число прогнозируемых целей.
    """
    center = float(np.median(residuals))
    scale = float(1.4826 * np.median(np.abs(residuals - center)))
    if scale == 0:
        scale = max(float(np.std(residuals)), 1.0)
    z_score = (residuals - center) / scale
    percentile = 1 - 0.05 / len(TARGETS)
    threshold = max(3.5, float(np.quantile(z_score, percentile)))
    return center, scale, threshold


def _category_history(
    hourly: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Возвращает факты и сезонные ориентиры в разрезе часа и категории.

    В каждом из двух словарей ключ обозначает показатель, строки таблицы
    — часы полного индекса, столбцы — категории мерчантов. Ориентир для
    ячейки — среднее того же часа и дня недели за предыдущие четыре недели.
    До накопления четырёх недель усредняются доступные недели, а при
    отсутствии предыдущей недели ячейка ориентира остаётся `NaN`.
    """
    observed = {}
    seasonal = {}
    for metric in (*TARGETS, "n_declined"):
        matrix = hourly.pivot_table(
            index="hour",
            columns="merchant_category",
            values=metric,
            aggfunc="sum",
            observed=True,
        ).reindex(index).sort_index(axis=1).fillna(0.0)
        observed[metric] = matrix
        # Сдвиг ровно на неделю даёт сезонный ориентир только из прошлых часов.
        previous = [matrix.shift(168 * week) for week in range(1, 5)]
        history_count = sum(frame.notna().astype(int) for frame in previous)
        history_sum = sum(frame.fillna(0.0) for frame in previous)
        seasonal[metric] = history_sum / history_count.replace(0, np.nan)
    return observed, seasonal


def _explanation(
    target: str,
    category: str,
    hour: pd.Timestamp,
    excess: float,
    observed: dict[str, pd.DataFrame],
    seasonal: dict[str, pd.DataFrame],
) -> str:
    """Формулирует гипотезу о пике по главной метрике и выбранной категории.

    Для оборота сравнивает средний чек категории с её сезонным ориентиром.
    Превышение в 1,4 раза выделяет гипотезу о повышенном среднем чеке.
    В остальных случаях формулировка описывает вид наблюдаемого роста.
    """
    if excess <= 0:
        return (
            "Значения по категориям укладываются в сезонную базу. "
            "Возможное объяснение — заниженный общий прогноз."
        )
    if target == "n_transactions":
        return f"Возможный источник пика — скачок числа операций в категории {category}."
    if target == "risk_events":
        return f"Возможный источник пика — скопление операционных риск-сигналов в категории {category}."
    current_count = float(observed["n_transactions"].at[hour, category])
    past_count = float(seasonal["n_transactions"].at[hour, category])
    current_amount = float(observed["total_amount"].at[hour, category])
    past_amount = float(seasonal["total_amount"].at[hour, category])
    if (
        current_count > 0 and past_count > 0 and past_amount > 0
        and current_amount / current_count > 1.4 * past_amount / past_count
    ):
        return f"Возможный источник пика — повышенный средний чек в категории {category}."
    return f"Возможный источник пика — рост оборота в категории {category}."


def _anomalies(
    hourly: pd.DataFrame,
    test: pd.DataFrame,
    selected: dict[str, str],
    calibration: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Находит превышения почасовых прогнозов на тесте и объясняет их.

    Для каждой выбранной модели параметры нормирования и порог берутся
    из калибровочной части валидации. Тестовый час попадает в таблицу, если
    хотя бы одна z-оценка выше своего верхнего порога. Главная метрика
    — с максимальной z-оценкой среди сработавших. Категория выбирается
    по наибольшему превышению недельного ориентира этой метрики. При
    отсутствии ориентира выбирается категория с максимальным фактом.

    Возвращает строки тревог от самой высокой z-оценки и словарь
    калибровочных параметров для каждой цели. Порог сохраняется и в
    единицах z-оценки, и в исходных единицах остатка.
    """
    z_scores = {}
    thresholds = {}
    for target in selected:
        residuals = calibration[f"actual_{target}"] - calibration[f"pred_{target}"]
        center, scale, threshold = _alert_calibration(residuals.to_numpy())
        test_residuals = test[f"actual_{target}"] - test[f"pred_{target}"]
        z_scores[target] = (test_residuals - center) / scale
        thresholds[target] = {
            "residual_median": center,
            "residual_scale": scale,
            "robust_z_threshold": threshold,
            "residual_threshold": center + scale * threshold,
        }

    score = pd.DataFrame(z_scores, index=test.index)
    flags = score.gt({target: values["robust_z_threshold"] for target, values in thresholds.items()})
    category_observed, category_seasonal = _category_history(hourly, index)
    alerts = []
    for hour in flags.index[flags.any(axis=1)]:
        flagged = [target for target in TARGETS if flags.at[hour, target]]
        target = max(flagged, key=lambda name: score.at[hour, name])
        actual = float(test.at[hour, f"actual_{target}"])
        forecast = float(test.at[hour, f"pred_{target}"])
        category_values = category_observed[target].loc[hour]
        category_baseline = category_seasonal[target].loc[hour]
        excess = category_values - category_baseline
        category = str(excess.idxmax()) if excess.notna().any() else str(category_values.idxmax())
        volume = float(category_observed["n_transactions"].at[hour, category])
        alerts.append({
            "hour": hour,
            "primary_metric": target,
            "flagged_metrics": "|".join(flagged),
            "max_robust_z": float(score.at[hour, target]),
            "observed": actual,
            "forecast": forecast,
            "residual": actual - forecast,
            "alert_residual_threshold": thresholds[target]["residual_threshold"],
            "merchant_category": category,
            "category_observed": float(category_values[category]),
            "category_same_hour_weekly_mean": float(category_baseline[category]),
            "category_excess": float(excess[category]),
            "category_risk_event_rate": (
                float(category_observed["risk_events"].at[hour, category]) / volume if volume else np.nan
            ),
            "category_decline_rate": (
                float(category_observed["n_declined"].at[hour, category]) / volume if volume else np.nan
            ),
            "possible_explanation": _explanation(
                target, category, hour, excess[category], category_observed, category_seasonal
            ),
        })
    alert_table = pd.DataFrame(alerts, columns=ALERT_COLUMNS)
    return alert_table.sort_values("max_robust_z", ascending=False), thresholds


def _review_load(
    hourly_forecasts: pd.DataFrame,
    scored: pd.DataFrame,
    reference_review_rate: float,
) -> pd.DataFrame:
    """Сопоставляет ожидаемое и наблюдаемое в ретроспективе число проверок.

    Ожидаемое число — прогноз операций, умноженный на фиксированную долю
    ручных проверок на валидации. Для наблюдаемого значения суммируются
    действия `review` риск-политики по часу UTC. Индекс результата совпадает
    с индексом прогнозов. Часам без таких действий присваивается ноль.
    """
    reviews = scored.loc[scored["action"].eq("review"), "event_ts"]
    review_counts = pd.to_datetime(reviews, utc=True).dt.floor("h").value_counts()
    output = hourly_forecasts[[
        "forecast_origin", "lead_hours", "actual_n_transactions",
        "pred_n_transactions", "pred_risk_events",
    ]].copy()
    output["expected_review_count"] = output["pred_n_transactions"] * reference_review_rate
    output["observed_review_count"] = review_counts.reindex(output.index, fill_value=0).astype(int)
    return output


def run_forecast(
    hourly: pd.DataFrame,
    scored: pd.DataFrame,
    out: Path,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    reference_review_rate: float = 0.05,
    risk_events_definition: str = "observable operational risk events",
) -> dict:
    """Запускает обучение, сравнение прогнозов, поиск аномалий и расчёт нагрузки.

    `hourly` содержит строки часовой витрины по категориям. Для общего
    прогноза значения суммируются по часам UTC. Разрез категорий сохраняется
    для объяснения аномалий. `scored` содержит время операции и действие
    риск-политики для проверки прогноза нагрузки на тесте.

    `train_end` и `validation_end` задают первые часы валидации и теста.
    Каждый прогнозный блок использует факты только до своего первого часа.
    Первая часть валидации выбирает метод по MAE отдельно для каждой цели.
    Остаток валидации определяет пороги тревоги. Тест показывает ошибки
    обоих методов и работу выбранных моделей с зафиксированными порогами.

    Записывает в `out` CSV с метриками, рядами прогноза, тревогами и
    нагрузкой. Возвращает сводку выбора моделей, порогов, пиков и
    ошибки оценки ручных проверок и сохраняет её в JSON. Параметр
    `risk_events_definition` описывает происхождение риск-сигналов в сводке.
    """
    if not 0 <= reference_review_rate <= 1:
        raise ValueError("reference_review_rate must be between 0 and 1.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = hourly[["hour", "merchant_category", *TARGETS, "n_declined"]].copy()
    data["hour"] = pd.to_datetime(data["hour"], utc=True)
    if data.empty or data["hour"].isna().any() or not data["hour"].eq(data["hour"].dt.floor("h")).all():
        raise ValueError("Forecasting requires nonempty hourly observations aligned to UTC hours.")
    if data["merchant_category"].isna().any():
        raise ValueError("Forecasting requires a populated merchant category for each hourly row.")
    data["merchant_category"] = data["merchant_category"].astype(str)
    values = data[[*TARGETS, "n_declined"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Hourly count and amount targets must be finite and nonnegative.")

    # Сплошная сетка сохраняет смысл лагов в часах даже при отсутствии операций.
    index = pd.date_range(data["hour"].min(), data["hour"].max(), freq="h", name="hour")
    totals = data.groupby("hour")[list(TARGETS)].sum().reindex(index, fill_value=0.0)
    n_observed = len(totals)
    n_train = int(index.searchsorted(_utc_timestamp(train_end)))
    n_validation_end = int(index.searchsorted(_utc_timestamp(validation_end)))
    n_validation = n_validation_end - n_train
    if n_train < 336 or n_validation < 48 or n_validation_end >= n_observed:
        raise ValueError(
            "Hourly forecasting requires at least 336 training hours, "
            "48 validation hours, and at least one observed test hour."
        )
    # Выбор метода заканчивается на границе полного прогнозного блока.
    # Оставшиеся часы валидации идут на оценку остатков для порогов тревоги.
    n_selection_end = n_train + HORIZON * max(1, (n_validation // 2) // HORIZON)
    full_index = pd.date_range(index[0], freq="h", periods=n_observed + HORIZON, name="hour")
    blocks = _blocks(n_train, n_validation_end, n_observed)
    classic_calendar, boost_calendar = _calendar(full_index)
    forecasts = pd.DataFrame(index=full_index)
    forecasts["forecast_origin"] = pd.Series(pd.NaT, index=full_index, dtype="datetime64[ns, UTC]")
    forecasts["lead_hours"] = 0
    for start, end in blocks:
        # Первый час блока имеет горизонт 1: последний факт ему предшествует.
        forecasts.loc[full_index[start:end], "forecast_origin"] = full_index[start]
        forecasts.loc[full_index[start:end], "lead_hours"] = np.arange(1, end - start + 1)

    selection_slice = slice(n_train, n_selection_end)
    evaluation = {
        "validation_selection": selection_slice,
        "validation": slice(n_train, n_validation_end),
        "test": slice(n_validation_end, n_observed),
    }
    metric_rows = []
    selected = {}
    classic_fit = {}
    with threadpool_limits(limits=1):
        for target in TARGETS:
            actual = totals[target].to_numpy(dtype=float)
            forecasts[f"actual_{target}"] = totals[target]
            classic, classic_fit[target] = _classic(actual, classic_calendar, n_train, blocks)
            predictions = {
                "sarimax": classic,
                "lag_boosting": _boosting(actual, boost_calendar, n_train, blocks),
            }
            for method in METHODS:
                forecast = predictions[method]
                forecasts[f"{method}_{target}"] = forecast
                for split, window in evaluation.items():
                    metric_rows.append({
                        "target": target,
                        "method": method,
                        "split": split,
                        "n_hours": len(actual[window]),
                        **_metrics(actual[window], forecast[window]),
                    })
            selected[target] = min(
                METHODS,
                key=lambda method: _metrics(actual[selection_slice], predictions[method][selection_slice])["mae"],
            )
            forecasts[f"pred_{target}"] = predictions[selected[target]]

    # Для оценки сохраняем прогнозы обоих методов и отдельно прогноз победителя.
    validation = forecasts.iloc[n_train:n_validation_end]
    calibration = forecasts.iloc[n_selection_end:n_validation_end]
    test = forecasts.iloc[n_validation_end:n_observed]
    alerts, alert_thresholds = _anomalies(data, test, selected, calibration, index)
    review_load = _review_load(test, scored, reference_review_rate)
    next_hours = forecasts.iloc[n_observed:].copy()
    next_hours = next_hours.drop(columns=[f"actual_{target}" for target in TARGETS])
    next_hours["expected_review_count"] = next_hours["pred_n_transactions"] * reference_review_rate

    metrics = pd.DataFrame(metric_rows)
    metrics["selected"] = metrics["method"].eq(metrics["target"].map(selected))
    metrics.to_csv(out / "forecast_metrics.csv", index=False)
    validation.to_csv(out / "forecast_validation.csv")
    test.to_csv(out / "forecast_test.csv")
    next_hours.to_csv(out / "forecast_next_24h.csv")
    alerts.to_csv(out / "anomaly_hours.csv", index=False)
    review_load.to_csv(out / "review_load.csv")
    summary = {
        "selected_models": selected,
        "classic_fit_status": classic_fit,
        "selection_hours": len(forecasts.iloc[selection_slice]),
        "alert_calibration_hours": len(calibration),
        "test_hours": len(test),
        "anomaly_hours": len(alerts),
        "anomaly_thresholds": alert_thresholds,
        "reference_review_rate": reference_review_rate,
        "review_count_mae": float(
            (review_load["expected_review_count"] - review_load["observed_review_count"]).abs().mean()
        ),
        "peak_test_expected_review_hour": review_load["expected_review_count"].idxmax().isoformat(),
        "peak_test_expected_reviews": float(review_load["expected_review_count"].max()),
        "peak_test_actual_review_hour": review_load["observed_review_count"].idxmax().isoformat(),
        "peak_test_actual_reviews": int(review_load["observed_review_count"].max()),
        "peak_next_expected_review_hour": next_hours["expected_review_count"].idxmax().isoformat(),
        "peak_next_expected_reviews": float(next_hours["expected_review_count"].max()),
        "peak_next_expected_risk_hour": next_hours["pred_risk_events"].idxmax().isoformat(),
        "peak_next_expected_risk_events": float(next_hours["pred_risk_events"].max()),
        "risk_events_definition": risk_events_definition,
        "origin_convention": (
            "The UTC timestamp is the first forecasted hour. "
            "The last observation available precedes that hour."
        ),
        "review_formula": "expected_review_count = pred_n_transactions * reference_review_rate",
    }
    (out / "forecast_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
