-- All history windows include the lower bound and exclude the current event time.
-- The input can include history before the exported interval to warm the windows.
-- Timestamps have microsecond precision. Ending RANGE at T - 1 µs excludes all ties at T.
WITH events AS (
    SELECT payments.*
    FROM payments
    CROSS JOIN mart_bounds AS bounds
    WHERE event_ts < bounds.ends_at
),
numbered_user_instants AS (
    SELECT
        user_id,
        event_ts,
        ROW_NUMBER() OVER (PARTITION BY user_id, event_ts ORDER BY transaction_id) AS instant_row
    FROM events
),
previous_user_instants AS (
    SELECT
        user_id,
        event_ts,
        LAG(event_ts) OVER (PARTITION BY user_id ORDER BY event_ts) AS previous_event_ts
    FROM numbered_user_instants
    WHERE instant_row = 1
),
rolling_features AS (
    SELECT
        *,
        COUNT(*) OVER user_hour AS f_user_count_1h,
        COUNT(*) OVER user_day AS f_user_count_24h,
        COUNT(*) OVER user_week AS f_user_count_7d,
        COALESCE(SUM(amount) OVER user_hour, 0.0) AS f_user_sum_1h,
        COALESCE(SUM(amount) OVER user_day, 0.0) AS f_user_sum_24h,
        COALESCE(SUM(amount) OVER user_week, 0.0) AS f_user_sum_7d,
        AVG(amount) OVER user_hour AS f_user_avg_amount_1h,
        AVG(amount) OVER user_day AS f_user_avg_amount_24h,
        AVG(amount) OVER user_week AS f_user_avg_amount_7d,
        COUNT(*) OVER card_week AS f_card_count_7d,
        AVG(amount) OVER card_week AS f_card_avg_amount_7d,
        STDDEV_SAMP(amount) OVER card_week AS card_amount_std_7d,
        COUNT(*) OVER merchant_hour AS f_merchant_count_1h,
        COALESCE(SUM(amount) OVER merchant_hour, 0.0) AS f_merchant_sum_1h,
        AVG(amount) OVER merchant_hour AS f_merchant_avg_amount_1h,
        COUNT(*) OVER merchant_week AS f_merchant_count_7d
    FROM events
    WINDOW
        user_hour AS (
            PARTITION BY user_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        user_day AS (
            PARTITION BY user_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        user_week AS (
            PARTITION BY user_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        card_week AS (
            PARTITION BY card_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        merchant_hour AS (
            PARTITION BY merchant_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ),
        merchant_week AS (
            PARTITION BY merchant_id ORDER BY event_ts
            RANGE BETWEEN INTERVAL '7 days' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        )
),
status_windows AS (
    SELECT *
    FROM (VALUES ('1h', INTERVAL '1 hour'), ('7d', INTERVAL '7 days')) AS widths(window_name, width)
),
known_status_intervals AS (
    SELECT
        events.merchant_id,
        status_windows.window_name,
        GREATEST(events.event_ts, events.status_available_at) AS starts_at,
        CASE WHEN events.status_available_at > events.event_ts THEN 0 ELSE 2 END AS start_phase,
        events.event_ts + status_windows.width AS ends_at,
        (events.status = 'declined')::INTEGER AS is_declined
    FROM events
    CROSS JOIN status_windows
    WHERE events.status IN ('approved', 'declined')
        AND events.status_available_at IS NOT NULL
        AND events.status_available_at <= events.event_ts + status_windows.width
),
merchant_instants AS (
    SELECT DISTINCT merchant_id, event_ts
    FROM events
),
status_change_events AS (
    -- At a timestamp: 0 = observed status arrives, 1 = score, 2 = open/close a historical window.
    -- Phase 2 opens exclude the current event itself and closes include the exact lower bound.
    SELECT merchant_id, window_name, starts_at AS instant, start_phase AS phase,
        1 AS delta_known, is_declined AS delta_declined
    FROM known_status_intervals
    UNION ALL
    SELECT merchant_id, window_name, ends_at, 2, -1, -is_declined
    FROM known_status_intervals
    UNION ALL
    SELECT merchant_instants.merchant_id, status_windows.window_name, merchant_instants.event_ts, 1, 0, 0
    FROM merchant_instants
    CROSS JOIN status_windows
),
coalesced_status_changes AS (
    SELECT merchant_id, window_name, instant, phase,
        SUM(delta_known) AS delta_known, SUM(delta_declined) AS delta_declined
    FROM status_change_events
    GROUP BY merchant_id, window_name, instant, phase
),
status_timeline AS (
    SELECT
        merchant_id, window_name, instant, phase,
        SUM(delta_known) OVER status_history AS known_count,
        SUM(delta_declined) OVER status_history AS declined_count
    FROM coalesced_status_changes
    WINDOW status_history AS (
        PARTITION BY merchant_id, window_name ORDER BY instant, phase
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
),
merchant_status_features AS (
    SELECT
        merchant_id,
        instant AS event_ts,
        MAX(known_count) FILTER (WHERE window_name = '1h')::BIGINT AS known_count_1h,
        MAX(known_count) FILTER (WHERE window_name = '7d')::BIGINT AS known_count_7d,
        MAX(declined_count) FILTER (WHERE window_name = '1h')::BIGINT AS declined_count_1h,
        MAX(declined_count) FILTER (WHERE window_name = '7d')::BIGINT AS declined_count_7d
    FROM status_timeline
    WHERE phase = 1
    GROUP BY merchant_id, instant
)
SELECT
    features.transaction_id,
    features.event_ts,
    features.user_id,
    features.card_id,
    features.merchant_id,
    features.merchant_category,
    features.amount,
    features.status,
    features.status_available_at,
    features.is_fraud,
    features.label_available_at,
    features.operational_risk_flag,
    features.amount AS f_amount,
    features.merchant_category AS f_merchant_category,
    features.country AS f_country,
    features.channel AS f_channel,
    EXTRACT(hour FROM features.event_ts)::integer AS f_hour_utc,
    EXTRACT(isodow FROM features.event_ts)::integer AS f_day_of_week,
    (EXTRACT(isodow FROM features.event_ts) >= 6)::INTEGER AS f_is_weekend,
    f_user_count_1h,
    f_user_count_24h,
    f_user_count_7d,
    f_user_sum_1h,
    f_user_sum_24h,
    f_user_sum_7d,
    f_user_avg_amount_1h,
    f_user_avg_amount_24h,
    f_user_avg_amount_7d,
    EXTRACT(epoch FROM features.event_ts - previous_user_instants.previous_event_ts)::double precision
        AS f_user_seconds_since_previous,
    f_card_count_7d,
    f_card_distinct_countries_7d,
    f_card_distinct_merchants_7d,
    f_card_avg_amount_7d,
    features.amount - f_card_avg_amount_7d AS f_card_amount_difference_7d,
    (features.amount - f_card_avg_amount_7d) / NULLIF(card_amount_std_7d, 0)
        AS f_card_amount_zscore_7d,
    f_merchant_count_1h,
    f_merchant_sum_1h,
    f_merchant_avg_amount_1h,
    f_merchant_count_7d,
    known_count_1h AS f_merchant_known_status_count_1h,
    declined_count_1h::double precision / NULLIF(known_count_1h, 0) AS f_merchant_decline_rate_1h,
    known_count_7d AS f_merchant_known_status_count_7d,
    declined_count_7d::double precision / NULLIF(known_count_7d, 0) AS f_merchant_decline_rate_7d,
    CASE
        WHEN known_count_7d < 20 THEN 'insufficient_history'
        WHEN declined_count_7d::double precision / known_count_7d >= 0.15 THEN 'high'
        WHEN declined_count_7d::double precision / known_count_7d >= 0.05 THEN 'medium'
        ELSE 'low'
    END AS f_merchant_risk_segment
FROM rolling_features AS features
INNER JOIN previous_user_instants USING (user_id, event_ts)
INNER JOIN merchant_status_features USING (merchant_id, event_ts)
-- PostgreSQL cannot use DISTINCT inside a window aggregate. The indexed
-- card_id/event_ts lookup keeps the same [T - 7d, T) semantics.
CROSS JOIN LATERAL (
    SELECT
        COUNT(DISTINCT history.country_raw)
            FILTER (WHERE history.country_raw NOT IN ('__UNKNOWN__', '__MISSING__'))
            AS f_card_distinct_countries_7d,
        COUNT(DISTINCT history.merchant_id) AS f_card_distinct_merchants_7d
    FROM payments AS history
    WHERE history.card_id = features.card_id
        AND history.event_ts >= features.event_ts - INTERVAL '7 days'
        AND history.event_ts < features.event_ts
) AS card_history
CROSS JOIN mart_bounds AS bounds
WHERE features.event_ts >= bounds.starts_at
ORDER BY features.event_ts, features.transaction_id;
