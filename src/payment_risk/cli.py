"""Run the same chronological pipeline on the demo or a transaction file."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import psycopg

from .data import clean_transactions, generate_demo
from .forecast import run_forecast
from .marts import build_marts
from .risk import run_risk


LOG = logging.getLogger(__name__)
DEMO_START = pd.Timestamp("2025-01-06T00:00:00Z")
HOUR = pd.Timedelta(hours=1)
DAY = pd.Timedelta(days=1)


def parse_instant(value: str) -> pd.Timestamp:
    try:
        stamp = pd.to_datetime(value, utc=True)
        if pd.isna(stamp):
            raise ValueError("missing date")
        return stamp
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(f"Invalid date/time: {value!r}") from error


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="CSV, CSV.GZ or Parquet transactions. Omit for the demo.")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--pg-dsn", help="PostgreSQL connection string. Defaults to PAYMENT_RISK_PG_DSN or libpq settings.")
    parser.add_argument("--days", type=int, default=98, help="Calendar days for the synthetic demo.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-end", type=parse_instant, help="Exclusive UTC cutoff, used as model training as-of time.")
    parser.add_argument("--validation-end", type=parse_instant, help="Exclusive UTC cutoff for validation and policy selection.")
    parser.add_argument("--observation-end", type=parse_instant, help="Data availability cutoff. Required with --input.")
    parser.add_argument("--rare-min-count", type=int, default=25)
    parser.add_argument("--review-capacity", type=float, default=0.06, help="Maximum validation fraction sent to review.")
    parser.add_argument("--max-fraud-loss-ratio", type=float, default=0.35, help="Maximum remaining fraud loss divided by approve-all loss.")
    parser.add_argument("--review-cost", type=float, default=2.0)
    parser.add_argument("--false-decline-cost", type=float, default=12.0)
    parser.add_argument("--review-fraud-catch-rate", type=float, default=0.80)
    parser.add_argument("--review-false-decline-rate", type=float, default=0.02)
    parser.add_argument("--fraud-lgd", type=float, default=1.0, help="Fraction of the transaction amount lost when fraud passes.")
    return parser


def read_transactions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.name.lower().endswith((".csv", ".csv.gz")):
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    raise ValueError("Input must be .csv, .csv.gz or .parquet.")


def timeline(args: argparse.Namespace, raw: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
    if args.input is None:
        start = DEMO_START
        end = start + args.days * DAY
        observation_end = args.observation_end or end
        end = min(end, observation_end.floor("h"))
    else:
        observation_end = args.observation_end
        if observation_end is None:
            raise ValueError("--observation-end is required with --input.")
        if "event_ts" not in raw:
            raise ValueError("Input has no event_ts column.")
        event = pd.to_datetime(raw["event_ts"], utc=True, format="mixed", errors="coerce")
        closed_hours = event[event < observation_end.floor("h")]
        if closed_hours.empty:
            raise ValueError("No transactions in complete UTC hours before --observation-end.")
        start = closed_hours.min().floor("h")
        end = closed_hours.max().floor("h") + HOUR

    calendar_start = start.ceil("D")
    calendar_days = (end.floor("D") - calendar_start).days
    default_train_days = max(21, (calendar_days * 64 + 99) // 100)
    train_end = args.train_end or calendar_start + default_train_days * DAY
    default_validation = calendar_start + (calendar_days * 79 // 100) * DAY
    validation_end = args.validation_end or max(default_validation, train_end + 6 * DAY)
    if not start < train_end < validation_end < end <= observation_end:
        raise ValueError("Expected start < train-end < validation-end < end <= observation-end.")
    if train_end != train_end.floor("D") or validation_end != validation_end.floor("D"):
        raise ValueError("Training and validation cutoffs must fall at 00:00 UTC for daily forecasts.")
    if train_end - start < 21 * DAY or validation_end - train_end < 6 * DAY or end - validation_end < 3 * DAY:
        raise ValueError("Provide at least 21 train, 6 validation and 3 test days.")
    return start, train_end, validation_end, end, observation_end


def run(args: argparse.Namespace) -> Path:
    if args.input is None and args.days < 34:
        raise ValueError("The demo requires at least 34 days. The default is 98.")
    if args.input is None:
        raw, synthetic = generate_demo(days=args.days, seed=args.seed)
    else:
        raw, synthetic = read_transactions(args.input), None
    start, train_end, validation_end, end, observation_end = timeline(args, raw)

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    LOG.info("Data window %s to %s, observation cutoff %s", start, end, observation_end)
    source_events = pd.to_datetime(raw["event_ts"], utc=True, format="mixed", errors="coerce")
    in_window = source_events.ge(start) & source_events.lt(end)
    excluded_events = int((source_events.notna() & ~in_window).sum())
    raw = raw.loc[in_window | source_events.isna()]
    clean, audit = clean_transactions(raw, train_end=train_end, rare_min_count=args.rare_min_count)
    audit["rows_outside_complete_hours"] = excluded_events

    labels_unavailable = clean["label_available_at"].gt(observation_end)
    audit["labels_hidden_at_observation_cutoff"] = int(labels_unavailable.sum())
    clean = clean.copy()
    clean.loc[labels_unavailable, "is_fraud"] = pd.NA
    clean.loc[labels_unavailable, "label_available_at"] = pd.NaT
    clean.to_parquet(out / "clean_transactions.parquet", index=False)
    (out / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    LOG.info("Building marts for %d cleaned transactions", len(clean))
    hourly, transactions = build_marts(clean, start=start, end=end, out=out, dsn=args.pg_dsn)
    LOG.info("Training risk models and searching validation thresholds")
    risk = run_risk(
        transactions, out, train_end=train_end, validation_end=validation_end,
        observation_end=observation_end, review_capacity=args.review_capacity,
        max_fraud_loss_ratio=args.max_fraud_loss_ratio, review_cost=args.review_cost,
        false_decline_cost=args.false_decline_cost, review_fraud_catch_rate=args.review_fraud_catch_rate,
        review_false_decline_rate=args.review_false_decline_rate, fraud_lgd=args.fraud_lgd, seed=args.seed,
    )
    LOG.info("Backtesting hourly forecasts and looking for anomalous peaks")
    scored = pd.read_csv(out / "test_scored.csv")
    forecast = run_forecast(
        hourly, scored, out, train_end=train_end, validation_end=validation_end,
        reference_review_rate=risk["validation_reference_review_rate"],
        risk_events_definition="Count of contemporaneous operational risk flags, available by hour close",
    )
    info = {
        "source": "synthetic" if args.input is None else str(args.input),
        "database": "PostgreSQL",
        "seed": args.seed,
        "start": start.isoformat(),
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "end": end.isoformat(),
        "observation_end": observation_end.isoformat(),
        "synthetic": synthetic,
        "n_transactions": len(transactions),
        "n_hourly_rows": len(hourly),
        "feature_exclusions": [
            "is_fraud", "label_available_at", "current status", "status_available_at",
            "operational_risk_flag", "raw customer and payment identifiers",
        ],
    }
    (out / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    from .report import write_report

    return write_report(out, run_info=info, audit=audit, risk=risk, forecast=forecast)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = make_parser()
    args = parser.parse_args()
    try:
        result = run(args)
    except (FileNotFoundError, ValueError, psycopg.OperationalError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
