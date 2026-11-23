"""Построение часовой и транзакционной витрин в PostgreSQL.

Модуль загружает очищенные операции во временную таблицу, выполняет два
запроса из каталога `sql` и выгружает результаты в Parquet. Часовая витрина
агрегирует поток по часу UTC и категории мерчанта. Транзакционная витрина
содержит одну строку на операцию и признаки, доступные в момент её поступления.
Обе витрины строятся в одной сессии PostgreSQL, которая удаляет временные
таблицы по завершении транзакции.
"""

from importlib.resources import files
from os import environ
from pathlib import Path

import pandas as pd
import psycopg
from psycopg import sql


# Явный порядок столбцов связывает записи DataFrame со списком полей в COPY.
PAYMENT_COLUMNS = (
    "transaction_id", "event_ts", "user_id", "card_id", "merchant_id",
    "merchant_category", "amount", "country", "country_raw", "channel",
    "status", "status_available_at", "is_fraud", "label_available_at",
    "operational_risk_flag",
)
TIMESTAMP_COLUMNS = ("event_ts", "status_available_at", "label_available_at")
# PostgreSQL передаёт идентификаторы типов OID в описании результата запроса.
INTEGER_OIDS = {20, 21, 23}
FLOAT_OIDS = {700, 701}
# Временная таблица хранит три отдельные метки времени. Первая определяет
# момент попытки, остальные определяют доступность статуса и метки фрода.
CREATE_PAYMENTS = """
    CREATE TEMP TABLE payments (
        transaction_id text PRIMARY KEY,
        event_ts timestamptz NOT NULL,
        user_id text NOT NULL,
        card_id text NOT NULL,
        merchant_id text NOT NULL,
        merchant_category text NOT NULL,
        amount double precision NOT NULL,
        country text NOT NULL,
        country_raw text NOT NULL,
        channel text NOT NULL,
        status text NOT NULL,
        status_available_at timestamptz,
        is_fraud smallint,
        label_available_at timestamptz,
        operational_risk_flag boolean NOT NULL
    ) ON COMMIT DROP
"""


def _utc_hour(value: pd.Timestamp, name: str) -> pd.Timestamp:
    """Проверяет, что граница выгрузки совпадает с началом часа UTC.

    Принимает отметку времени `value` с указанным часовым поясом и переводит
    её в UTC. Граница с минутами, секундами или микросекундами вызывает ошибку.
    Параметр `name` указывает в сообщении, какая граница требует исправления.
    """
    instant = pd.Timestamp(value)
    if instant.tz is None:
        raise ValueError(f"{name} must have an explicit timezone")
    instant = instant.tz_convert("UTC")
    if instant != instant.floor("h"):
        raise ValueError(f"{name} must be aligned to the start of a UTC hour")
    return instant


def _copy_payments(cursor: psycopg.Cursor, clean: pd.DataFrame) -> None:
    """Загружает очищенный DataFrame в сессионную таблицу `payments`.

    Курсор `cursor` принадлежит сессии, в которой уже создана временная таблица.
    Из `clean` выбираются поля схемы в порядке, указанном в COPY. Драйвер
    передаёт строки потоком, а пропуски pandas становятся SQL NULL.
    """
    statement = sql.SQL("COPY pg_temp.payments ({}) FROM STDIN").format(
        sql.SQL(", ").join(sql.Identifier(column) for column in PAYMENT_COLUMNS)
    )
    rows = clean.loc[:, list(PAYMENT_COLUMNS)].to_numpy(dtype=object, na_value=None)
    with cursor.copy(statement) as copy:
        for row in rows:
            copy.write_row(row)


def _fetch_frame(cursor: psycopg.Cursor, query: str) -> pd.DataFrame:
    """Выполняет запрос витрины и возвращает результат с числовыми типами.

    Имена и типы столбцов берутся из метаданных PostgreSQL. Для столбцов
    вещественного типа сохраняется числовой dtype, даже если все ответы
    равны NULL. Целочисленные столбцы с NULL переводятся в тип Int64 pandas,
    который хранит пропуски вместе с целыми числами.
    """
    cursor.execute(query)
    columns = cursor.description
    frame = pd.DataFrame.from_records(cursor.fetchall(), columns=[column.name for column in columns])
    for column in columns:
        name = column.name
        if column.type_code in FLOAT_OIDS:
            frame[name] = frame[name].astype(float)
        if column.type_code in INTEGER_OIDS and frame[name].isna().any():
            frame[name] = frame[name].astype("Int64")
    return frame


def build_marts(
    clean: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    out: Path,
    dsn: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит и сохраняет часовую и транзакционную витрины.

    `clean` содержит операции после очистки и нормализации. `start` и `end`
    задают полуоткрытый интервал выгрузки с точностью до целого часа UTC.
    Более ранние операции из `clean` доступны транзакционному запросу как
    история для окон. Операции начиная с `end` отсутствуют в обеих витринах.

    Каталог `out` получает файлы `hourly_mart.parquet` и
    `transaction_mart.parquet`. Функция также возвращает эти витрины в виде
    двух DataFrame в том же порядке. Строку подключения можно передать в
    `dsn` или в переменной `PAYMENT_RISK_PG_DSN`. При пропуске обоих
    параметров соединение использует стандартные настройки libpq.
    """
    start = _utc_hour(start, "start")
    end = _utc_hour(end, "end")
    if start >= end:
        raise ValueError("start must precede end")
    if clean["transaction_id"].isna().any() or not clean["transaction_id"].is_unique:
        raise ValueError("The clean input must contain unique non-null transaction_id values")

    # Число попыток в интервале служит контрольной суммой для обеих витрин.
    expected_transactions = int(clean["event_ts"].between(start, end, inclusive="left").sum())
    if expected_transactions == 0:
        raise ValueError("The requested range contains no transactions")

    sql_directory = files("payment_risk").joinpath("sql")
    conninfo = dsn if dsn is not None else environ.get("PAYMENT_RISK_PG_DSN", "")
    with psycopg.connect(conninfo) as connection, connection.cursor() as cursor:
        # UTC задаёт единые часовые границы для DATE_TRUNC, календарных
        # признаков и интервалов в днях при любом часовом поясе сервера.
        cursor.execute("SET LOCAL TIME ZONE 'UTC'")
        # Запросы обращаются к временным таблицам текущей сессии. ON COMMIT
        # DROP удаляет их после успешного завершения блока подключения.
        cursor.execute("SET LOCAL search_path TO pg_temp, pg_catalog")
        cursor.execute(CREATE_PAYMENTS)
        cursor.execute(
            "CREATE TEMP TABLE mart_bounds (starts_at timestamptz, ends_at timestamptz) ON COMMIT DROP"
        )
        cursor.execute(
            "INSERT INTO pg_temp.mart_bounds (starts_at, ends_at) VALUES (%s, %s)",
            (start.to_pydatetime(), end.to_pydatetime()),
        )
        _copy_payments(cursor, clean)
        # Индекс ускоряет поиск семидневной истории карты в LATERAL.
        # Статистика ANALYZE помогает планировщику оценить объёмы таблицы.
        cursor.execute("CREATE INDEX ON pg_temp.payments (card_id, event_ts)")
        cursor.execute("ANALYZE pg_temp.payments")
        hourly = _fetch_frame(cursor, sql_directory.joinpath("hourly.sql").read_text(encoding="utf-8"))
        transactions = _fetch_frame(
            cursor, sql_directory.joinpath("transactions.sql").read_text(encoding="utf-8")
        )

    # Приводим метки времени к UTC в выходных DataFrame и восстанавливаем
    # компактный тип бинарной метки с поддержкой неизвестных значений.
    hourly["hour"] = pd.to_datetime(hourly["hour"], utc=True)
    for column in TIMESTAMP_COLUMNS:
        transactions[column] = pd.to_datetime(transactions[column], utc=True)
    transactions["is_fraud"] = transactions["is_fraud"].astype("Int8")

    # Каждая попытка должна попасть в часовую сумму и ровно в одну строку
    # транзакционной витрины. Проверяем это до записи файлов.
    if (
        len(transactions) != expected_transactions
        or not transactions["transaction_id"].is_unique
        or hourly["n_transactions"].sum() != expected_transactions
    ):
        raise RuntimeError("Mart outputs failed transaction count or uniqueness checks")

    out.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(out / "hourly_mart.parquet", index=False)
    transactions.to_parquet(out / "transaction_mart.parquet", index=False)
    return hourly, transactions
