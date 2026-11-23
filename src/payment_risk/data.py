"""Generate a reproducible example and validate the shared payment schema."""

from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "transaction_id", "event_ts", "user_id", "card_id", "merchant_id",
    "merchant_category", "amount", "country", "channel", "status",
    "status_available_at", "is_fraud", "label_available_at", "operational_risk_flag",
)
IDENTIFIERS = ("transaction_id", "user_id", "card_id", "merchant_id")
TIMESTAMPS = ("event_ts", "status_available_at", "label_available_at")
CATEGORIES = ("merchant_category", "country", "channel")
MISSING = "__missing__"
OTHER = "__other__"

STATUS_ALIASES = {
    "accepted": "approved", "approve": "approved", "approved": "approved",
    "authorized": "approved", "authorised": "approved", "completed": "approved",
    "ok": "approved", "paid": "approved", "settled": "approved", "success": "approved",
    "decline": "declined", "declined": "declined", "denied": "declined",
    "refused": "declined", "reject": "declined", "rejected": "declined",
    "pending": "pending", "processing": "pending", "in_progress": "pending",
    "canceled": "cancelled", "cancelled": "cancelled", "voided": "cancelled",
    "error": "error", "failed": "error", "technical_failure": "error",
    "unknown": "unknown",
}
BOOLEAN_VALUES = {
    "0": 0, "0.0": 0, "false": 0, "no": 0,
    "1": 1, "1.0": 1, "true": 1, "yes": 1,
}
NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "unknown", "nat"}


def _reject_rows(frame: pd.DataFrame, mask: pd.Series | np.ndarray, reason: str) -> None:
    if mask.any():
        examples = frame.loc[mask, "transaction_id"].head(5).fillna("<missing>").tolist()
        raise ValueError(f"{reason}: {int(mask.sum())} rows; transaction_id examples: {examples}")


def _parse_boolean(frame: pd.DataFrame, name: str, *, nullable: bool) -> pd.Series:
    tokens = frame[name].astype("string").str.strip().str.casefold()
    missing = tokens.isna() | tokens.isin(NULL_TOKENS)
    parsed = tokens.map(BOOLEAN_VALUES).astype("Int8")
    _reject_rows(frame, parsed.isna() & (~missing | (not nullable)), f"Invalid {name} (expected 0/1)")
    return parsed


def _balance(labels: pd.Series) -> dict[str, int | float | None]:
    known = int(labels.notna().sum())
    fraud = int(labels.sum())
    return {
        "known": known,
        "fraud": fraud,
        "non_fraud": known - fraud,
        "unknown": int(labels.isna().sum()),
        "fraud_rate": fraud / known if known else None,
    }


def clean_transactions(
    raw: pd.DataFrame, *, train_end: pd.Timestamp, rare_min_count: int = 25
) -> tuple[pd.DataFrame, dict]:
    """Normalize attempts and learn category vocabulary on pre-cutoff events."""
    if not raw.columns.is_unique:
        raise ValueError("Duplicate input column names")
    absent = sorted(set(REQUIRED_COLUMNS) - set(raw.columns))
    if absent:
        raise ValueError(f"Missing required columns: {', '.join(absent)}")
    if raw.empty or rare_min_count < 1:
        raise ValueError("Input must contain rows and rare_min_count must be positive")
    cutoff = pd.to_datetime(train_end, utc=True)
    if pd.isna(cutoff):
        raise ValueError("train_end must be a valid timestamp")

    frame = raw.loc[:, list(REQUIRED_COLUMNS)].drop_duplicates().reset_index(drop=True)
    exact_duplicates = len(raw) - len(frame)
    for name in IDENTIFIERS:
        frame[name] = frame[name].astype("string").str.strip().replace("", pd.NA)
        _reject_rows(frame, frame[name].isna(), f"Missing {name}")

    rounded_timestamps = 0
    for name in TIMESTAMPS:
        tokens = frame[name].astype("string").str.strip()
        missing = tokens.isna() | tokens.str.casefold().isin(NULL_TOKENS)
        parsed = pd.to_datetime(tokens.mask(missing), format="mixed", errors="coerce", utc=True)
        _reject_rows(frame, parsed.isna() & (~missing | (name == "event_ts")), f"Invalid {name}")
        rounded = parsed.dt.floor("us")
        rounded_timestamps += int((parsed.notna() & (parsed != rounded)).sum())
        frame[name] = rounded

    amounts = pd.to_numeric(frame["amount"], errors="coerce").astype("float64")
    _reject_rows(frame, ~np.isfinite(amounts) | (amounts <= 0), "Invalid amount (expected finite positive value)")
    frame["amount"] = amounts
    frame["is_fraud"] = _parse_boolean(frame, "is_fraud", nullable=True)
    frame["operational_risk_flag"] = _parse_boolean(frame, "operational_risk_flag", nullable=False).astype("int8")

    status_tokens = (
        frame["status"].astype("string").str.strip().str.casefold().str.replace(r"[\s-]+", "_", regex=True)
    )
    status_unrecognized = status_tokens.notna() & ~status_tokens.isin(set(STATUS_ALIASES) | NULL_TOKENS)
    unrecognized_status_rows = int(status_unrecognized.sum())
    frame["status"] = status_tokens.map(STATUS_ALIASES).fillna("unknown").astype("string")
    _reject_rows(
        frame, (frame["status"] != "unknown") & frame["status_available_at"].isna(),
        "A known status requires status_available_at",
    )
    _reject_rows(
        frame, frame["is_fraud"].isna() != frame["label_available_at"].isna(),
        "is_fraud and label_available_at must either both be present or both be empty",
    )
    for name in ("status_available_at", "label_available_at"):
        _reject_rows(frame, frame[name] < frame["event_ts"], f"{name} is before event_ts")

    for name in CATEGORIES:
        value = frame[name].astype("string").str.strip()
        if name == "country":
            # NA is ISO 3166-1 for Namibia, so preserve it as a valid country.
            value = value.mask(value.str.casefold().isin(NULL_TOKENS - {"na"})).str.upper()
        else:
            value = value.mask(value.str.casefold().isin(NULL_TOKENS)).str.casefold()
            value = value.str.replace(r"\s+", "_", regex=True)
        frame[name] = value

    before_canonical_dedup = len(frame)
    frame = frame.drop_duplicates().copy()
    canonical_duplicates = before_canonical_dedup - len(frame)
    _reject_rows(
        frame, frame["transaction_id"].duplicated(keep=False),
        "Conflicting records share the same transaction_id",
    )
    pretrain = frame["event_ts"] < cutoff
    if not pretrain.any():
        raise ValueError("No transactions before train_end (required to fit category vocabulary)")

    frame["country_raw"] = frame["country"]
    rare_categories = {}
    for name in CATEGORIES:
        values = frame[name]
        counts = values.loc[pretrain].value_counts()
        vocabulary = sorted(counts.loc[counts >= rare_min_count].index.tolist())
        other = values.notna() & ~values.isin(vocabulary)
        frame[name] = values.mask(other, OTHER).fillna(MISSING)
        rare_categories[name] = {
            "kept": vocabulary,
            "rare_training_categories": int((counts < rare_min_count).sum()),
            "unseen_later_categories": int(values.loc[~pretrain & ~values.isin(counts.index)].nunique()),
            "rows_mapped_to_other": int(other.sum()),
            "missing_rows": int(values.isna().sum()),
        }

    frame = frame.sort_values(["event_ts", "transaction_id"], ignore_index=True)
    train_rows = frame["event_ts"] < cutoff
    training_labels = frame.loc[train_rows, "is_fraud"].where(frame.loc[train_rows, "label_available_at"] <= cutoff)
    audit = {
        "raw_rows": len(raw),
        "clean_rows": len(frame),
        "exact_duplicate_rows": exact_duplicates,
        "canonical_duplicate_rows": canonical_duplicates,
        "transaction_id_unique": bool(frame["transaction_id"].is_unique),
        "start_utc": frame["event_ts"].min().isoformat(),
        "latest_event_utc": frame["event_ts"].max().isoformat(),
        "timezone": "UTC",
        "submicrosecond_timestamp_values_rounded_down": rounded_timestamps,
        "normalized_status_counts": {str(key): int(value) for key, value in frame["status"].value_counts().items()},
        "unrecognized_status_rows_before_canonical_dedup": unrecognized_status_rows,
        "rare_min_count": rare_min_count,
        "rare_categories": rare_categories,
        "fraud_class_balance": {
            "all_supplied_labels": _balance(frame["is_fraud"]),
            "mature_at_train_end": _balance(training_labels),
        },
    }
    return frame, audit


def _sigmoid(log_odds: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds))


def generate_demo(*, days: int = 98, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Simulate recurring users, seasonal payment volume, and delayed fraud labels."""
    if days < 1:
        raise ValueError("days must be positive")
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2025-01-06", periods=24 * days, freq="h", tz="UTC")
    hour_of_day = hours.hour.to_numpy()
    service_rate = (
        10.0
        + 19.0 * np.exp(-0.5 * ((hour_of_day - 13) / 4.7) ** 2)
        + 33.0 * np.exp(-0.5 * ((hour_of_day - 20) / 3.1) ** 2)
    )
    weekday_factor = np.array([0.94, 0.99, 1.02, 1.04, 1.12, 1.04, 0.86])[hours.dayofweek]
    counts = rng.poisson(service_rate * weekday_factor * (1 + 0.0015 * np.arange(len(hours)) / 24))

    category_names = np.array([
        "groceries", "restaurants", "transport", "subscriptions", "gaming",
        "digital_goods", "travel", "electronics", "specialty",
    ])
    category_prob = np.array([0.23, 0.19, 0.16, 0.10, 0.08, 0.09, 0.05, 0.098, 0.002])
    amount_center = np.array([41.0, 30.0, 17.0, 14.0, 41.0, 43.0, 123.0, 81.0, 29.0])
    online_prob = np.array([0.10, 0.19, 0.23, 0.96, 0.90, 0.97, 0.89, 0.60, 0.33])
    merchant_counts = np.array([33, 38, 21, 18, 23, 26, 18, 29, 5])
    injections = (
        (0.43, 17, "promotion", 7, 76),
        (0.73, 2, "card_testing", 4, 88),
        (0.84, 19, "promotion", 7, 174),
        (0.90, 13, "high_ticket_batch", 6, 67),
        (0.95, 3, "card_testing", 4, 129),
        (0.97, 16, "service_outage", None, 0),
    )
    injected_anomalies = []
    indexed_injections = []
    for fraction, clock_hour, kind, category, extra in injections:
        index = min(int(fraction * days), days - 1) * 24 + clock_hour
        change = extra
        if kind == "service_outage":
            change = -int(counts[index])
            counts[index] = 0
        indexed_injections.append((index, kind, category, extra))
        injected_anomalies.append({
            "hour": hours[index].isoformat(),
            "kind": kind,
            "merchant_category": category_names[category].item() if category is not None else None,
            "extra_transactions": change,
        })

    regular_hours = np.repeat(np.arange(len(hours)), counts)
    injected_hours = [np.full(extra, index) for index, _, _, extra in indexed_injections if extra]
    hour_index = np.concatenate([regular_hours, *injected_hours])
    size = len(hour_index)

    user_count = 2900
    countries = np.array(["US", "GB", "DE", "FR", "CA", "NL", "BR", "JP", "AE", "MX", "NZ", "SG"])
    country_prob = np.array([0.39, 0.15, 0.12, 0.10, 0.07, 0.05, 0.044, 0.032, 0.016, 0.020, 0.006, 0.002])
    user_home = rng.choice(countries, user_count, p=country_prob)
    user_spend = rng.lognormal(0, 0.45, user_count)
    user_weight = rng.lognormal(0, 1.20, user_count)
    user_weight /= user_weight.sum()
    user_preferred_category = rng.choice(len(category_names), user_count, p=category_prob)
    user_preferred_merchant = rng.integers(merchant_counts[user_preferred_category])
    has_second_card = rng.random(user_count) < 0.22

    user = rng.choice(user_count, size, p=user_weight)
    category = rng.choice(len(category_names), size, p=category_prob)
    choose_favorite = rng.random(size) < 0.27
    category[choose_favorite] = user_preferred_category[user[choose_favorite]]
    abusive_pattern = rng.random(size) < 0.027
    abusive_category = abusive_pattern & (rng.random(size) < 0.58)
    category[abusive_category] = rng.choice([4, 5, 6, 7], abusive_category.sum(), p=[0.29, 0.31, 0.10, 0.30])

    promotion = np.zeros(size, dtype=bool)
    high_ticket = np.zeros(size, dtype=bool)
    card_testing = np.zeros(size, dtype=bool)
    cursor = len(regular_hours)
    for _, kind, injected_category, extra in indexed_injections:
        rows = slice(cursor, cursor + extra)
        cursor += extra
        if kind == "service_outage":
            continue
        category[rows] = injected_category
        if kind == "promotion":
            promotion[rows] = True
        elif kind == "high_ticket_batch":
            high_ticket[rows] = True
        elif kind == "card_testing":
            card_testing[rows] = True
            targeted_users = rng.choice(user_count, 12, replace=False)
            user[rows] = rng.choice(targeted_users, extra)
            abusive_pattern[rows] = rng.random(extra) < 0.84

    merchant = rng.integers(merchant_counts[category])
    repeat_merchant = (
        (category == user_preferred_category[user]) & (rng.random(size) < 0.39)
    )
    merchant[repeat_merchant] = user_preferred_merchant[user[repeat_merchant]]
    # Injected events are concentrated at a few vendors within a category.
    merchant[promotion] = rng.choice([1, 2], promotion.sum(), p=[0.75, 0.25])
    merchant[high_ticket] = 3
    merchant[card_testing] = 0
    elevated_merchant = (merchant <= 1) & np.isin(category, [4, 5, 7])

    second_card = has_second_card[user] & (rng.random(size) < 0.23) & ~card_testing
    card = 2 * user + second_card
    country = user_home[user].copy()
    cross_border_observation = rng.random(size) < np.where(abusive_pattern, 0.76, 0.032)
    country[cross_border_observation] = rng.choice(
        countries, cross_border_observation.sum(), p=country_prob[::-1],
    )
    foreign = country != user_home[user]
    online = rng.random(size) < np.where(abusive_pattern | card_testing, 0.93, online_prob[category])
    channel = np.where(online, np.where(rng.random(size) < 0.63, "web", "app"), "pos").astype("U8")
    channel[(rng.random(size) < 0.00020) & ~abusive_pattern] = "partner"

    ratio_to_expected = rng.lognormal(0.0, 0.60, size)
    ratio_to_expected[abusive_pattern] *= rng.lognormal(1.33, 0.39, abusive_pattern.sum())
    ratio_to_expected[high_ticket] *= 4.3
    ratio_to_expected[card_testing] *= 0.66
    amount = np.round(np.clip(amount_center[category] * user_spend[user] * ratio_to_expected, 0.50, 4500), 2)
    unusually_large = ratio_to_expected >= 2.5
    local_hour = hour_of_day[hour_index]
    night = (local_hour < 6) | (local_hour == 23)
    # Distinct random draws for the observed risk alert and the latent outcome.
    fraud_rate = _sigmoid(
        -6.37 + 1.39 * online + 1.72 * foreign + 1.35 * unusually_large
        + 0.85 * night + 0.48 * np.isin(category, [4, 5, 7])
        + 0.83 * elevated_merchant + 1.48 * abusive_pattern + 0.83 * card_testing
    )
    fraud = (rng.random(size) < fraud_rate).astype("int8")
    alert_rate = _sigmoid(
        -4.20 + 1.14 * foreign + 0.62 * (channel == "web") + 0.65 * unusually_large
        + 0.66 * elevated_merchant + 1.28 * abusive_pattern + 2.35 * card_testing
    )
    operational_flag = rng.random(size) < alert_rate
    decline_rate = _sigmoid(
        -3.66 + 0.60 * online + 0.58 * foreign + 0.35 * unusually_large
        + 0.70 * abusive_pattern + 1.20 * operational_flag + 1.37 * card_testing
    )
    declined = rng.random(size) < decline_rate
    technical_error = (~declined) & (rng.random(size) < 0.003)
    still_pending = (~declined & ~technical_error) & (rng.random(size) < 0.002)
    status = np.where(rng.random(size) < 0.55, "Approved", " success ")
    status[declined] = np.where(rng.random(declined.sum()) < 0.55, "declined", "REJECTED")
    status[technical_error] = "failed"
    status[still_pending] = "pending"

    offsets_us = rng.integers(0, 3_600_000_000, size=size)
    event_ts = hours.take(hour_index) + pd.to_timedelta(offsets_us, unit="us")
    status_lag_seconds = rng.integers(1, 240, size=size)
    status_lag_seconds[still_pending] += 900
    status_at = event_ts + pd.to_timedelta(status_lag_seconds, unit="s")
    labels_at = event_ts + pd.to_timedelta(rng.integers(48 * 60, 72 * 60 + 1, size=size), unit="m")
    merchant_ids = np.array([
        f"merchant_{category_id:02d}_{merchant_id:02d}"
        for category_id, merchant_id in zip(category, merchant, strict=True)
    ])
    raw = pd.DataFrame({
        "transaction_id": [f"txn_{index:08d}" for index in range(size)],
        "event_ts": event_ts,
        "user_id": [f"user_{value:05d}" for value in user],
        "card_id": [f"card_{value:05d}" for value in card],
        "merchant_id": merchant_ids,
        "merchant_category": category_names[category],
        "amount": amount,
        "country": country,
        "channel": channel,
        "status": status,
        "status_available_at": status_at,
        "is_fraud": fraud,
        "label_available_at": labels_at,
        "operational_risk_flag": operational_flag.astype("int8"),
    })
    alternate_timestamps = rng.choice(size, 2 * (size // 20), replace=False)
    non_utc, naive_utc = np.split(alternate_timestamps, 2)
    for name in TIMESTAMPS:
        original = raw[name]
        mixed = original.astype("string")
        mixed.iloc[non_utc] = original.iloc[non_utc].dt.tz_convert("America/New_York").astype("string")
        mixed.iloc[naive_utc] = original.iloc[naive_utc].dt.tz_localize(None).astype("string")
        raw[name] = mixed

    duplicate_count = max(1, size // 1000)
    duplicates = raw.iloc[rng.choice(size, duplicate_count, replace=False)]
    raw = pd.concat([raw, duplicates], ignore_index=True)
    raw = raw.iloc[rng.permutation(len(raw))].reset_index(drop=True)
    return raw, {
        "source": "synthetic",
        "seed": seed,
        "start_utc": hours[0].isoformat(),
        "end_utc": (hours[-1] + pd.Timedelta(hours=1)).isoformat(),
        "transactions_before_duplicate_injection": size,
        "injected_duplicate_rows": duplicate_count,
        "timestamp_format_rows_per_variant_before_duplicate_injection": len(non_utc),
        "injected_anomalies": injected_anomalies,
        "label_delay_hours": [48, 72],
    }
