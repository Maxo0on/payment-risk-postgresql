-- Inputs are the cleaned payments table and the UTC interval [starts_at, ends_at).
WITH hour_axis AS (
    SELECT series.hour
    FROM mart_bounds AS bounds,
    LATERAL generate_series(
        bounds.starts_at,
        bounds.ends_at - INTERVAL '1 hour',
        INTERVAL '1 hour'
    ) AS series(hour)
),
categories AS (
    SELECT DISTINCT payments.merchant_category
    FROM payments
    CROSS JOIN mart_bounds AS bounds
    WHERE payments.event_ts >= bounds.starts_at AND payments.event_ts < bounds.ends_at
),
payments_by_hour AS (
    SELECT
        DATE_TRUNC('hour', payments.event_ts) AS hour,
        payments.merchant_category,
        payments.amount,
        payments.user_id,
        payments.status,
        payments.operational_risk_flag,
        payments.status_available_at < DATE_TRUNC('hour', payments.event_ts) + INTERVAL '1 hour'
            AS status_known_at_close
    FROM payments
    CROSS JOIN mart_bounds AS bounds
    WHERE payments.event_ts >= bounds.starts_at AND payments.event_ts < bounds.ends_at
),
aggregate_hour AS (
    SELECT
        hour,
        merchant_category,
        COUNT(*) AS n_transactions,
        SUM(amount) AS total_amount,
        COUNT(DISTINCT user_id) AS n_users,
        COUNT(*) FILTER (WHERE status IN ('approved', 'declined') AND status_known_at_close)
            AS n_status_known,
        COUNT(*) FILTER (WHERE status = 'declined' AND status_known_at_close) AS n_declined,
        COUNT(*) FILTER (WHERE operational_risk_flag) AS risk_events
    FROM payments_by_hour
    GROUP BY hour, merchant_category
),
complete_grid AS (
    SELECT
        hour_axis.hour,
        categories.merchant_category,
        COALESCE(aggregate_hour.n_transactions, 0) AS n_transactions,
        COALESCE(aggregate_hour.total_amount, 0.0) AS total_amount,
        COALESCE(aggregate_hour.n_users, 0) AS n_users,
        COALESCE(aggregate_hour.n_status_known, 0) AS n_status_known,
        COALESCE(aggregate_hour.n_declined, 0) AS n_declined,
        COALESCE(aggregate_hour.risk_events, 0) AS risk_events
    FROM hour_axis
    CROSS JOIN categories
    LEFT JOIN aggregate_hour USING (hour, merchant_category)
)
SELECT
    *,
    total_amount / NULLIF(n_transactions, 0) AS avg_amount,
    n_declined::double precision / NULLIF(n_transactions, 0) AS declined_rate,
    n_declined::double precision / NULLIF(n_status_known, 0) AS declined_rate_known,
    risk_events::double precision / NULLIF(n_transactions, 0) AS risk_event_rate,
    LAG(n_transactions, 1) OVER category_hour AS n_transactions_lag_1h,
    LAG(n_transactions, 24) OVER category_hour AS n_transactions_lag_24h,
    (SUM(n_transactions) OVER past_day)::bigint AS n_transactions_previous_24h,
    AVG(total_amount) OVER past_day AS avg_hourly_amount_previous_24h
FROM complete_grid
WINDOW
    category_hour AS (PARTITION BY merchant_category ORDER BY hour),
    past_day AS (
        PARTITION BY merchant_category ORDER BY hour
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    )
ORDER BY hour, merchant_category;
