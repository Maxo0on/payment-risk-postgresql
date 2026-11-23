"""Проверки временных границ SQL-витрин и правил выбора порогов риска.

Запускайте через ``python -m unittest discover -s tests -v``. Тест витрин
использует PostgreSQL из ``PAYMENT_RISK_PG_DSN`` и небольшой набор из пяти
платежей, поэтому ожидаемые признаки можно посчитать вручную.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

from payment_risk.marts import build_marts
from payment_risk.risk import PolicyCosts, _actions, _optimize_thresholds, _policy_outcomes


class MartTimingTests(unittest.TestCase):
    """Проверяет, какие операции и статусы доступны на момент расчёта признака."""

    def test_rolling_windows_use_strictly_earlier_events_and_known_statuses(self):
        """Проверяет одновременные платежи, запаздывающие статусы и пустой час.

        Для обоих платежей в 00:10 в часовую историю входит только платёж
        на 10 единиц из 00:00, отстоящий на 600 секунд. К 00:20 видны два
        статуса, один из них отказ. В 01:00 в истории четыре операции.
        За три часа ожидаются 4, 1 и 0 операций, а также 1, 0 и 0 отказов.
        """

        # Общие идентификаторы дают одну историю для пользователя, карты и мерчанта.
        rows = pd.DataFrame({
            "transaction_id": ["previous", "tie_a", "tie_b", "at_status", "later"],
            "event_ts": ["00:00", "00:10", "00:10", "00:20", "01:00"],
            "status_available_at": ["00:20", "00:20", "01:00", "00:30", "01:01"],
            "user_id": "user",
            "card_id": "card",
            "merchant_id": "merchant",
            "merchant_category": "retail",
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0],
            "country": "US",
            "country_raw": ["US", "US", "CA", "US", "US"],
            "channel": "web",
            "status": ["declined", "approved", "declined", "approved", "approved"],
            "is_fraud": pd.Series([1, 0, 1, 0, 0], dtype="Int8"),
            "operational_risk_flag": [True, False, False, False, False],
        })
        for field in ("event_ts", "status_available_at"):
            rows[field] = pd.to_datetime("2025-01-06 " + rows[field], utc=True)
        rows["label_available_at"] = rows["event_ts"] + pd.Timedelta(days=2)

        # При выходе из блока Parquet удаляются. Обе витрины остаются в памяти.
        with TemporaryDirectory() as folder:
            hourly, transactions = build_marts(
                rows, start=pd.Timestamp("2025-01-06T00:00:00Z"),
                end=pd.Timestamp("2025-01-06T03:00:00Z"), out=Path(folder),
            )
        tx = transactions.set_index("transaction_id")
        for name in ("tie_a", "tie_b"):
            # Соседний платёж с тем же временем ещё отсутствует в истории.
            self.assertEqual(tx.loc[name, "f_user_count_1h"], 1)
            self.assertEqual(tx.loc[name, "f_user_sum_1h"], 10)
            self.assertEqual(tx.loc[name, "f_user_seconds_since_previous"], 600)
            self.assertEqual(tx.loc[name, "f_card_distinct_countries_7d"], 1)
            self.assertEqual(tx.loc[name, "f_card_distinct_merchants_7d"], 1)
            self.assertEqual(tx.loc[name, "f_merchant_known_status_count_1h"], 0)
            self.assertTrue(pd.isna(tx.loc[name, "f_merchant_decline_rate_1h"]))

        # Статус, опубликованный ровно в 00:20, уже доступен в транзакционной витрине.
        self.assertEqual(tx.loc["at_status", "f_card_distinct_countries_7d"], 2)
        self.assertEqual(tx.loc["at_status", "f_merchant_known_status_count_1h"], 2)
        self.assertEqual(tx.loc["at_status", "f_merchant_decline_rate_1h"], 0.5)
        self.assertEqual(tx.loc["later", "f_user_count_1h"], 4)
        self.assertEqual(tx.loc["later", "f_merchant_decline_rate_1h"], 0.5)

        # Для первого часового отчёта учитываются статусы, известные строго до 01:00.
        self.assertEqual(hourly["n_transactions"].tolist(), [4, 1, 0])
        self.assertEqual(hourly["n_declined"].tolist(), [1, 0, 0])


class PolicyTests(unittest.TestCase):
    """Проверяет назначение действий и оптимальность порогов на малой выборке."""

    def test_thresholds_respect_equalities(self):
        """При порогах 0,2 и 0,4 равенства дают одобрение и проверку соответственно.

        Пять оценок покрывают оба порога и все три промежутка правил.
        """
        np.testing.assert_array_equal(
            _actions(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), 0.2, 0.4),
            ["approve", "approve", "review", "review", "reject"],
        )

    def test_optimizer_matches_enumeration_with_ties_and_pending_labels(self):
        """Сравнивает оптимизатор с перебором всех пар порогов.

        В наборе восемь операций, повторяющиеся оценки и одна незрелая метка.
        Допустимы не более двух ручных проверок и до 20% потерь от мошенничества
        при одобрении всех операций.
        Ожидаемая стоимость равна минимальной среди пар, прошедших оба лимита.
        """
        scores = np.array([0.1, 0.2, 0.2, 0.4, 0.5, 0.5, 0.9, 0.95])
        cohort = pd.DataFrame({
            "is_fraud": pd.Series([0, 0, pd.NA, 1, 0, 0, 1, 1], dtype="Int8"),
            "amount": [5.0, 10.0, 4.0, 100.0, 6.0, 2.0, 250.0, 500.0],
        })
        mature = cohort["is_fraud"].notna()
        costs = PolicyCosts(review_capacity=0.25, max_fraud_loss_ratio=0.2)

        # Ноль ставит порог ниже всех оценок. Порог по уникальным оценкам
        # оставляет одинаковые score вместе, а срез сохраняет порядок порогов.
        boundaries = [0.0, *np.unique(scores)]
        feasible_costs = []
        for i, review in enumerate(boundaries):
            for reject in boundaries[i:]:
                result = _policy_outcomes(cohort, mature, _actions(scores, review, reject), costs)
                if result["review_limit_met"] and result["fraud_loss_limit_met"]:
                    feasible_costs.append(result["total_cost"])

        # Проверяем выполнимость и стоимость, поскольку оптимальных пар может быть несколько.
        review, reject = _optimize_thresholds(cohort, mature, scores, costs)
        optimal = _policy_outcomes(cohort, mature, _actions(scores, review, reject), costs)
        self.assertTrue(optimal["review_limit_met"])
        self.assertTrue(optimal["fraud_loss_limit_met"])
        self.assertAlmostEqual(optimal["total_cost"], min(feasible_costs))
