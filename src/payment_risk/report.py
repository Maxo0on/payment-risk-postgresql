"""Собирает русскоязычный HTML-отчёт по завершённому запуску конвейера.

Основные таблицы берутся из сводок аудита, прогноза и скоринга, а почасовые
графики и список аномалий — из CSV-файлов прогона. Модуль форматирует готовые
метрики и явно показывает периоды и когорты, на которых они рассчитаны.
Стили и PNG-графики встраиваются в документ для просмотра одним файлом.
"""

from __future__ import annotations

import base64
from html import escape
from io import BytesIO
from math import floor
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import pandas as pd


INK = "#162e40"
MUTED = "#5e6e79"
BLUE = "#245681"
TEAL = "#037d77"
AMBER = "#b36416"
GRID = "#e6edf2"

# Широкие таблицы прокручиваются внутри страницы. Правила печати уменьшают
# отступы и стараются сохранить строки таблиц и графики на одной странице.
CSS = """
:root {
  color-scheme: light;
  --ink: #162e40; --muted: #526574; --accent: #076e68;
  --border: #dce5eb; --soft: #f2f6f8; --max: 1100px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
* { box-sizing: border-box; }
body { margin: 0; color: var(--ink); background: #fff; line-height: 1.64; }
.wrap { width: min(var(--max), calc(100% - 48px)); margin-inline: auto; }
header { background: #112e42; color: #fff; padding: clamp(36px, 7vw, 78px) 0 38px; }
header h1 { max-width: 850px; margin: 0 0 14px; font-size: clamp(2rem, 5.3vw, 4.1rem); line-height: 1.08; letter-spacing: -.045em; }
header p { color: #d1e2ed; max-width: 780px; font-size: 1.08rem; margin: 0 0 24px; }
nav { display: flex; flex-wrap: wrap; gap: 9px 20px; }
nav a { color: #c9f0e9; text-decoration: underline; text-underline-offset: 4px; font-size: .91rem; }
nav a:hover, nav a:focus-visible { color: #fff; }
main { padding: 28px 0 60px; }
section { margin-top: 64px; scroll-margin-top: 24px; }
section:first-child { margin-top: 0; }
h2 { font-size: clamp(1.55rem, 4vw, 2.4rem); line-height: 1.15; letter-spacing: -.032em; margin: 0 0 20px; }
h3 { font-size: 1.25rem; line-height: 1.3; margin: 27px 0 9px; }
p { margin: 0 0 14px; }
ul { margin: 8px 0 17px; padding-left: 1.4rem; }
li + li { margin-top: 7px; }
code { background: #eaf1f4; color: #173a4d; padding: .09em .28em; border-radius: 3px; overflow-wrap: anywhere; font-size: .88em; }
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0 0 18px; }
.kpi { border: 1px solid var(--border); border-top: 3px solid var(--accent); border-radius: 8px; padding: 15px; background: #fff; }
.kpi strong { display: block; font-size: clamp(1.35rem, 2.6vw, 1.85rem); letter-spacing: -.025em; line-height: 1.18; margin: 2px 0 8px; overflow-wrap: anywhere; }
.kpi span { display: block; color: var(--muted); font-size: .91rem; }
.note { background: var(--soft); border-left: 3px solid #649aa4; padding: 13px 16px; margin: 15px 0; }
.warning { background: #fff6e7; border-left-color: #b36416; }
.note p:last-child { margin-bottom: 0; }
.source-line { color: var(--muted); font-size: .9rem; }
.table-scroll { overflow-x: auto; margin: 10px 0 17px; }
table { border-collapse: collapse; width: 100%; text-align: left; font-size: .92rem; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 12px; vertical-align: top; border-bottom: 1px solid var(--border); }
thead th { background: var(--soft); color: #19384b; }
tbody tr:nth-child(even) { background: #f8fafb; }
figure { margin: 20px 0; border: 1px solid var(--border); padding: 12px; border-radius: 8px; }
figure img { width: 100%; height: auto; display: block; }
figcaption { font-size: .9rem; color: var(--muted); padding: 7px 9px 2px; }
footer { padding: 20px 0 30px; background: var(--soft); color: var(--muted); font-size: .9rem; }
@media (max-width: 760px) { .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 475px) { .wrap { width: calc(100% - 28px); } .kpi { padding: 10px; } th, td { padding: 7px; } }
@media print {
  .wrap { width: 100%; } header { background: #fff; color: #132d40; padding: 0 0 14px; }
  header p { color: #394e5d; } nav { display: none; } section { margin-top: 28px; }
  h2, h3 { break-after: avoid; } figure, tr, .kpi { break-inside: avoid; }
  table { font-size: 9pt; } th, td { padding: 5px 6px; }
}
"""


def _text(value: object) -> str:
    """Преобразует значение в строку и экранирует HTML, включая кавычки в атрибутах."""
    return escape(str(value), quote=True)


def _number(value: object, digits: int = 0) -> str:
    """Форматирует число с `digits` знаками после запятой и группировкой разрядов.

    Пробелы между разрядами неразрывные. Для отсутствующего значения выводится тире.
    """
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):,.{digits}f}".replace(",", "\u00a0").replace(".", ",")


def _share(value: object, digits: int = 1) -> str:
    """Переводит долю в проценты с `digits` знаками. Для пропуска возвращает тире."""
    if value is None or pd.isna(value):
        return "—"
    return f"{_number(float(value) * 100, digits)}\u00a0%"


def _utc(value: object) -> str:
    """Форматирует метку времени UTC с точностью до минуты для вывода в отчёте.

    При явном смещении время переводится в UTC, а метка без смещения трактуется
    как UTC. Для пропущенной метки выводится тире.
    """
    if value is None or pd.isna(value):
        return "—"
    return pd.to_datetime(value, utc=True).strftime("%d.%m.%Y %H:%M")


def _table(headers: tuple[str, ...], rows) -> str:
    """Возвращает HTML-таблицу по заголовкам и итерируемой последовательности строк.

    Ячейки и заголовки экранируются как текст. При отсутствии строк выводится
    короткое сообщение, а контейнер таблицы позволяет прокручивать широкие столбцы.
    """
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    if not body:
        return '<p class="source-line">Нет строк для отображения.</p>'
    head = "".join(f'<th scope="col">{_text(header)}</th>' for header in headers)
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _csv(out: Path, name: str) -> pd.DataFrame:
    """Читает CSV-артефакт `name` из `out` для таблицы или графика.

    Для отсутствующего либо физически пустого файла возвращается пустой DataFrame.
    Прочие ошибки ввода и разбора передаются вызывающему коду.
    """
    try:
        return pd.read_csv(out / name)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _figure(figure: Figure, alt: str, caption: str) -> str:
    """Формирует HTML-блок с PNG, альтернативным текстом и видимой подписью.

    Agg рендерит фигуру Matplotlib в памяти. Данные PNG кодируются в Base64,
    чтобы браузер мог открыть изображение из HTML без соседних файлов.
    """
    buffer = BytesIO()
    FigureCanvasAgg(figure).print_png(buffer)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<figure><img src="data:image/png;base64,{encoded}" alt="{_text(alt)}">'
        f"<figcaption>{_text(caption)}</figcaption></figure>"
    )


def _style_axis(axis) -> None:
    """Задаёт оси Matplotlib общую сетку и умеренную плотность подписей делений."""
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def _card(value: str, label: str) -> str:
    """Возвращает HTML-карточку с экранированными значением и подписью показателя."""
    return f'<div class="kpi"><strong>{_text(value)}</strong><span>{_text(label)}</span></div>'


def _data_section(run_info: dict, audit: dict) -> str:
    """Собирает раздел об источнике, очистке и временных границах выборок.

    `run_info` задаёт источник данных, интервалы и время наблюдения меток.
    `audit` содержит счётчики дублей, нормализованных статусов, редких категорий
    и операций, исключённых из неполных часов. Возвращается фрагмент HTML.
    """
    source_is_synthetic = bool(run_info.get("synthetic"))
    source = "Синтетический набор" if source_is_synthetic else "Пользовательский CSV"
    # Граница train_end одновременно завершает обучение и открывает валидацию.
    # Аналогично validation_end служит первой меткой времени теста.
    periods = _table(
        ("Интервал", "Начало UTC", "Правая граница UTC"),
        (
            ("Обучение", _utc(run_info.get("start")), _utc(run_info.get("train_end"))),
            ("Валидация", _utc(run_info.get("train_end")), _utc(run_info.get("validation_end"))),
            ("Тест", _utc(run_info.get("validation_end")), _utc(run_info.get("end"))),
        ),
    )
    statuses = audit.get("normalized_status_counts", {})
    status_text = ", ".join(f"{name} — {_number(count)}" for name, count in statuses.items())
    status_note = f"<p>Нормализованные статусы после очистки. {_text(status_text)}.</p>" if status_text else ""
    rare_counts = {
        field: item["rows_mapped_to_other"]
        for field, item in audit.get("rare_categories", {}).items() if item.get("rows_mapped_to_other")
    }
    rare_note = ""
    if rare_counts:
        rare_changes = ", ".join(f"{field} — {_number(count)}" for field, count in rare_counts.items())
        rare_note = f"<p>Число замен на общую категорию по каждому полю. {_text(rare_changes)}.</p>"
    excluded = audit.get("rows_outside_complete_hours", 0)
    excluded_note = f"<p>За пределами полных часов исключено {_number(excluded)} операций.</p>" if excluded else ""
    return f"""
<section id="data"><h2>1. Данные и временное разбиение</h2>
<p><strong>{source}.</strong> Обработано {_number(audit.get("raw_rows"))} строк, после очистки осталось
{_number(audit.get("clean_rows"))} уникальных операций. Удалено {_number(audit.get("exact_duplicate_rows"))}
полностью одинаковых строк и {_number(audit.get("canonical_duplicate_rows"))} строк, совпавших после
нормализации. Различающиеся записи с одним идентификатором считаются конфликтом входных данных.</p>
<p>Время приведено к UTC, а дневные и недельные сезонности рассчитываются по часам UTC. Метки времени без зоны
трактуются как UTC. Словарь частых стран, каналов и категорий мерчантов строится на обучающем периоде.
Редкие и новые значения преобразуются в общую категорию. Исходный код страны сохраняется для подсчёта
различных стран карты.</p>
{status_note}{rare_note}{excluded_note}{periods}
<p>Правая граница каждого интервала исключается. Метка участвует в расчёте только после
<code>label_available_at</code>. Время среза наблюдения для итоговой оценки — {_utc(run_info.get("observation_end"))} UTC.</p>
</section>
"""


def _marts_section() -> str:
    """Возвращает HTML с описанием двух витрин PostgreSQL и состава признаков.

    Раздел объясняет окна строго до момента платежа, доступность исторического
    статуса и правила риск-сегмента мерчанта.
    """
    return """
<section id="sql"><h2>2–3. SQL-витрины и признаки</h2>
<p>PostgreSQL строит часовую витрину на уровне «час UTC × категория мерчанта». Часы без операций заполнены нулями.
В строке есть число операций, сумма, число уникальных пользователей, отклонения и оперативные риск-события.
Отклонение попадает в час операции, если его статус известен до начала следующего часа. В витрине доступны
доля отклонений среди всех операций и доля среди операций с известным финальным статусом.</p>
<p>Транзакционная витрина содержит одну строку на операцию. CTE разделяют расчёт окон, историю статусов и итоговые
признаки. <code>COUNT() OVER</code> и <code>SUM() OVER</code> агрегируют прошлое, <code>LAG()</code> находит
предыдущее время, <code>ROW_NUMBER()</code> упорядочивает записи. Для операции в момент <em>t</em> окно длины
<em>W</em> имеет границы [<em>t</em> − <em>W</em>, <em>t</em>). Операции с одинаковой меткой времени не попадают
в историю друг друга.</p>
<ul>
<li><strong>Пользователь.</strong> Число операций, суммы и средний чек за 1 час, 24 часа и 7 дней.
Также рассчитывается интервал от предыдущей операции.</li>
<li><strong>Карта.</strong> Число стран и мерчантов за 7 дней, исторический средний чек, отклонение текущей суммы
в денежных единицах и стандартизованное отклонение. Уникальные значения считает запрос <code>LATERAL</code>
по индексу на карте и времени операции.</li>
<li><strong>Мерчант.</strong> Объём и средний чек за предыдущий час, доля отклонений за час и 7 дней.
Доля использует лишь предыдущие операции с известным к моменту оценки результатом <code>approved</code>
или <code>declined</code>. При менее чем 20 таких операциях за 7 дней истории недостаточно. Иначе доля
от 15% обозначает высокий сегмент, доля от 5% — средний, оставшиеся значения — низкий.</li>
<li><strong>Платёж и календарь.</strong> Текущая сумма, страна, категория, канал, час суток, день недели и выходной.</li>
</ul>
<div class="note"><p>Для скоринга используются столбцы с префиксом <code>f_</code>. Идентификаторы, статус текущей
операции, метка мошенничества и момент её подтверждения остаются за пределами набора признаков.</p></div>
<p class="source-line">Результаты сборки доступны в <code>hourly_mart.parquet</code> и <code>transaction_mart.parquet</code>.</p>
</section>
"""


# Машинные имена используются при чтении артефактов. Подписи для отчёта общие
# для графиков, таблиц и пояснений.
TARGET_LABELS = {
    "n_transactions": "Число операций",
    "total_amount": "Оборот",
    "risk_events": "Риск-события",
}
METHOD_LABELS = {"sarimax": "SARIMAX", "lag_boosting": "Бустинг на лагах"}
SPLIT_LABELS = {"train": "Обучение", "validation": "Валидация", "test": "Тест"}


def _forecast_figure(test: pd.DataFrame, review: pd.DataFrame) -> str:
    """Строит четыре почасовых графика с общей шкалой времени UTC.

    `test` содержит `hour` и пары фактических и прогнозных значений трёх рядов.
    `review` содержит `hour` и значения нагрузки по решениям политики и прогнозу.
    Возвращает HTML со встроенным PNG или пустую строку при отсутствии данных.
    """
    if test.empty or review.empty:
        return ""
    figure = Figure(figsize=(10.6, 7.6), dpi=125, layout="constrained")
    # Общая ось времени даёт возможность сопоставить пики в одни часы.
    # Для каждого показателя оставлена своя ось Y из-за разных единиц и масштаба.
    axes = figure.subplots(4, 1, sharex=True)
    hour = pd.to_datetime(test["hour"], utc=True)
    for axis, (target, label) in zip(axes, TARGET_LABELS.items()):
        axis.plot(hour, test[f"actual_{target}"], color=BLUE, linewidth=1.05, label="Факт")
        axis.plot(hour, test[f"pred_{target}"], color=AMBER, linewidth=1.15, label="Прогноз")
        axis.set_title(label, loc="left", color=INK, fontweight="semibold", fontsize=10)
        _style_axis(axis)
    axes[0].legend(loc="upper right", frameon=False, ncols=2, fontsize=9)

    # Нагрузка по скорингу получена из решений ретроспективной симуляции.
    axis = axes[-1]
    review_hour = pd.to_datetime(review["hour"], utc=True)
    axis.plot(review_hour, review["observed_review_count"], color=BLUE, linewidth=1.05,
              label="По скорингу")
    axis.plot(review_hour, review["expected_review_count"], color=TEAL, linewidth=1.15,
              label="Ожидалось")
    axis.set_title("Операции на ручную проверку", loc="left", color=INK,
                   fontweight="semibold", fontsize=10)
    axis.legend(loc="upper right", frameon=False, ncols=2, fontsize=9)
    _style_axis(axis)
    locator = AutoDateLocator(minticks=4, maxticks=8)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    axis.set_xlabel("Час UTC", color=MUTED, fontsize=9)
    return _figure(
        figure,
        "Часовые факты и прогнозы для количества, оборота, риск-событий и ручных проверок на тесте.",
        "Прогноз рассчитан блоками до 24 часов. По скорингу показана нагрузка при симуляции политики.",
    )


def _forecast_section(out: Path, forecast: dict) -> str:
    """Собирает сравнение методов прогноза и разбор нагрузки ручной проверки.

    `forecast` содержит выбранные методы, размеры частей валидации и сводку пиков.
    CSV-файлы в `out` дают метрики по разрезам и почасовые ряды для графиков.
    Возвращается фрагмент HTML с показателями, прогнозами и их интерпретацией.
    """
    metrics = _csv(out, "forecast_metrics.csv")
    review = _csv(out, "review_load.csv")
    selected = forecast.get("selected_models", {})
    forecast_rows = []
    for target, label in TARGET_LABELS.items():
        for method, method_label in METHOD_LABELS.items():
            if metrics.empty:
                continue
            matches = metrics.loc[metrics["target"].eq(target) & metrics["method"].eq(method)]
            splits = matches.set_index("split").to_dict(orient="index")
            # Выбор метода основан на первой части валидации. Тестовые метрики
            # характеризуют качество уже выбранного метода на другом периоде.
            validation = splits.get("validation_selection", {})
            test = splits.get("test", {})
            forecast_rows.append((
                label,
                f"{method_label} (выбран)" if selected.get(target) == method else method_label,
                _number(validation.get("mae"), 2),
                _number(test.get("mae"), 2),
                _number(test.get("rmse"), 2),
                _share(test.get("wape")),
            ))
    metrics_table = _table(
        ("Показатель", "Метод", "MAE для выбора", "MAE на тесте", "RMSE на тесте", "WAPE на тесте"),
        forecast_rows,
    )
    peak_table = _table(
        ("Оценка пика", "Час UTC", "Проверок за час"),
        (
            ("Прогноз на тестовом периоде", _utc(forecast.get("peak_test_expected_review_hour")),
             _number(forecast.get("peak_test_expected_reviews"), 1)),
            ("Решения скоринга на тестовом периоде", _utc(forecast.get("peak_test_actual_review_hour")),
             _number(forecast.get("peak_test_actual_reviews"))),
            ("Прогноз на следующие 24 часа", _utc(forecast.get("peak_next_expected_review_hour")),
             _number(forecast.get("peak_next_expected_reviews"), 1)),
        ),
    )
    peak_note = ""
    if not review.empty:
        # Сравнение в один час показывает ошибку оценки пика. Максимум прогноза
        # может относиться к другой дате, поэтому он вынесен в таблицу отдельно.
        observed_peak = review.loc[review["observed_review_count"].idxmax()]
        peak_note = (
            f'<p>В час максимальной нагрузки скоринг отправил на проверку '
            f'{_number(observed_peak["observed_review_count"])} операций, при прогнозе '
            f'{_number(observed_peak["expected_review_count"], 1)}. Для планирования смен оценка описывает '
            f'обычный поток. Всплески рискованных операций требуют отдельных сигналов и резерва ёмкости.</p>'
        )
    classic_status = forecast.get("classic_fit_status", {})
    classic_notes = [
        f"{TARGET_LABELS[target]} — {status_text}"
        for target, status in classic_status.items()
        if (status_text := {
            "optimizer_did_not_converge": "оптимизатор SARIMAX не достиг критерия сходимости",
            "constant_training_series": "на обучении ряд постоянен, используется его уровень",
        }.get(status))
    ]
    classic_note = (
        f'<p class="note warning">{_text(". ".join(classic_notes))}.</p>' if classic_notes else ""
    )
    return f"""
<section id="forecast"><h2>4. Прогноз часовой нагрузки</h2>
<p>Прогнозируются число операций, оборот и число операционных риск-событий. Риск-событие — сигнал
<code>operational_risk_flag</code>, доступный при поступлении платежа. Подтверждённые chargeback и мошенничество
поступают позже и используются в классификаторе как целевые метки.</p>
<p><strong>SARIMAX</strong> оценивает авторегрессию первого порядка, дневные и недельные гармоники Фурье
и эффект выходного. После завершения каждого блока фактические наблюдения обновляют состояние фильтра.
<strong>Градиентный бустинг</strong> использует календарь, лаги до 168 часов и оконные средние и стандартные
отклонения. Внутри блока модель подставляет собственные предсказания в последующие лаги.</p>
<p>Обе модели обучаются до границы обучения. Каждый блок до 24 часов строится по наблюдениям,
доступным к его началу. Метод для каждого ряда выбирается по минимальной MAE за первые
{_number(forecast.get("selection_hours"))} часов валидации. Последующие
{_number(forecast.get("alert_calibration_hours"))} часов валидации служат для настройки детектора аномалий.</p>
{classic_note}{metrics_table}
<p class="source-line">MAE — средняя абсолютная ошибка, RMSE сильнее штрафует большие ошибки. WAPE делит
сумму абсолютных ошибок на суммарное фактическое значение ряда. Полное сравнение доступно в
<code>forecast_metrics.csv</code>.</p>
{_forecast_figure(_csv(out, "forecast_test.csv"), review)}
<h3>Нагрузка ручной проверки</h3>
<p>Прогноз числа платежей умножается на долю проверок на валидации
({_share(forecast.get("reference_review_rate"), 2)}). Средняя абсолютная ошибка на тесте —
{_number(forecast.get("review_count_mae"), 2)} проверки в час. Почасовое сравнение учитывает все транзакции
в тестовом часовом интервале, включая операции без созревшей метки.</p>
{peak_table}{peak_note}
<p>Максимум прогноза риск-событий на следующие сутки приходится на
{_utc(forecast.get("peak_next_expected_risk_hour"))} UTC. Ожидаемое число сигналов в этот час —
{_number(forecast.get("peak_next_expected_risk_events"), 1)}.</p>
<p class="source-line">План для ближайших 24 часов доступен в <code>forecast_next_24h.csv</code>.
<code>review_load.csv</code> содержит прогноз и расчёт по решениям скоринга для каждого тестового часа.</p>
</section>
"""


def _anomaly_section(out: Path, run_info: dict, forecast: dict) -> str:
    """Показывает сильнейшие алерты теста и проверяет их на сценариях генератора.

    Данные для таблицы читаются из `anomaly_hours.csv`. Если `run_info` содержит
    сценарии генератора, их часы сопоставляются с алертами после перевода в UTC.
    Сводка `forecast` задаёт общее число подозрительных часов. Результат — HTML.
    """
    anomalies = _csv(out, "anomaly_hours.csv")
    synthetic = run_info.get("synthetic") or {}
    scenarios = synthetic.get("injected_anomalies", [])
    alert_hours = set(pd.to_datetime(anomalies["hour"], utc=True)) if not anomalies.empty else set()
    start_test = pd.to_datetime(run_info.get("validation_end"), utc=True)
    end_test = pd.to_datetime(run_info.get("end"), utc=True)
    test_scenarios = []
    for scenario in scenarios:
        hour = pd.to_datetime(scenario["hour"], utc=True)
        if start_test <= hour < end_test:
            test_scenarios.append((hour, scenario))
    # Сопоставление ведётся по уникальным часам. Несколько сценариев или
    # аномальных показателей в одном часе дают один час в сводном счётчике.
    synthetic_count = sum(hour in alert_hours for hour in {hour for hour, _ in test_scenarios})
    test_hours = len({hour for hour, _ in test_scenarios})
    scenario_hours = {hour for hour, _ in test_scenarios}
    anomaly_rows = []
    for row in anomalies.head(10).to_dict(orient="records"):
        hour = pd.to_datetime(row["hour"], utc=True)
        anomaly_rows.append((
            _utc(hour),
            TARGET_LABELS.get(row["primary_metric"], row["primary_metric"]),
            _number(row["max_robust_z"], 2),
            _number(row["observed"], 1),
            _number(row["forecast"], 1),
            row["merchant_category"],
            row["possible_explanation"],
            "Да" if hour in scenario_hours else "—",
        ))
    headers = ("Час UTC", "Основной сигнал", "z", "Факт", "Прогноз", "Категория", "Гипотеза")
    if synthetic:
        alert_table = _table((*headers, "Сценарий генератора"), anomaly_rows)
    else:
        alert_table = _table(headers, (row[:-1] for row in anomaly_rows))
    synthetic_note = ""
    if test_scenarios:
        scenario_labels = {
            "promotion": "Промоакция",
            "high_ticket_batch": "Серия дорогих покупок",
            "card_testing": "Массовое тестирование карт",
            "service_outage": "Сбой сервиса",
        }
        scenario_table = _table(
            ("Час UTC", "Сценарий", "Категория", "Изменение числа операций", "Детектор"),
            (
                (_utc(hour), scenario_labels.get(scenario.get("kind"), scenario.get("kind")),
                 scenario.get("merchant_category") or "—",
                 _number(scenario.get("extra_transactions")), "Обнаружен" if hour in alert_hours else "Пропуск")
                for hour, scenario in test_scenarios
            ),
        )
        outage_note = ""
        if any(scenario.get("kind") == "service_outage" and hour not in alert_hours
               for hour, scenario in test_scenarios):
            outage_note = " Спад при сбое относится к нижнему хвосту, поэтому детектор всплесков его пропустил."
        synthetic_note = f"""
<h3>Проверка на сценариях из генератора</h3>
<p>Детектор отметил {synthetic_count} из {test_hours} тестовых часов, в которые генератор добавил
известный сценарий.{outage_note} Среди остальных алертов возможны естественные колебания ряда.</p>{scenario_table}
"""
    return f"""
<section id="anomaly"><h2>5. Аномальные часы</h2>
<p>Алерт возникает, когда положительный остаток «факт − прогноз» становится необычно большим.
Остаток центрируется медианой и масштабируется через медианное абсолютное отклонение
(<code>1,4826 × MAD</code>). Порог для каждого ряда задан максимумом из 3,5 и верхнего квантиля
валидационных остатков <code>1 − 0,05 / 3</code>. Порог фиксируется до тестового периода.</p>
<p>На тесте обнаружено {_number(forecast.get("anomaly_hours"))} подозрительных часов. Ниже до десяти часов
с наибольшим превышением. Для ведущего сигнала выделяется категория с наибольшим приростом относительно
того же часа в предыдущих четырёх неделях. Это ориентир для расследования. Установление причины требует
проверки логов, отдельных мерчантов и контекста платежей.</p>
{alert_table}{synthetic_note}
<p class="source-line">Все алерты и доли риск-событий и отклонений в ведущих категориях доступны в
<code>anomaly_hours.csv</code>.</p>
</section>
"""


def _risk_section(out: Path, risk: dict) -> str:
    """Собирает HTML о зрелости меток, качестве моделей и выборе политики.

    `risk` предоставляет численности и метрики по периодам, выбранную модель
    и оценки её конкурентов на валидации. Из `feature_importance.csv` в `out`
    берётся первая пятёрка отсортированных по важности признаков.
    """
    label_rows = []
    for split, label in SPLIT_LABELS.items():
        counts = risk.get("label_counts", {}).get(split, {})
        mature = counts.get("mature", 0)
        fraud = counts.get("fraud", 0)
        # Доля fraud относится к известным исходам. Для когорты без созревших
        # меток доля неизвестна и отображается тире.
        label_rows.append((
            label, _number(counts.get("transactions")), _number(mature), _number(fraud),
            _share(fraud / mature if mature else None, 2),
        ))
    label_table = _table(("Период", "Операций", "Созревших меток", "Fraud", "Доля fraud"), label_rows)
    metrics = risk.get("metrics", [])
    metric_table = _table(
        ("Период", "Модель", "ROC-AUC", "PR-AUC (AP)", "K", "Recall@K", "Precision@K"),
        (
            (SPLIT_LABELS[row["split"]], row["model"], _number(row["roc_auc"], 3),
             _number(row["pr_auc_ap"], 3), _number(row["top_k"]),
             _share(row["recall_at_top_k"], 2), _share(row["precision_at_top_k"], 2))
            for row in sorted(metrics, key=lambda row: (row["split"] == "test", row["model"]))
        ),
    )
    selected = risk.get("selected_model", "—")
    importance = _csv(out, "feature_importance.csv")
    top_features = ", ".join(importance["feature"].head(5)) if not importance.empty else ""
    importance_note = (
        f'<p>Пять ведущих признаков по внутренней важности {_text(selected)} — '
        f'<code>{_text(top_features)}</code>. Рейтинг зависит от метода обучения '
        f'и описывает вклад в модель.</p>' if top_features else ""
    )
    candidates = _table(
        ("Модель", "Доля проверок", "Остаток fraud loss", "Совокупные потери, ден. ед."),
        (
            (name, _share(row["review_rate"], 2), _share(row["fraud_loss_ratio"], 2),
             _number(row["total_cost"], 2))
            for name, row in risk.get("validation_policy_candidates", {}).items()
        ),
    )
    return f"""
<section id="risk"><h2>6. Скоринг мошенничества</h2>
<p>CatBoost и LightGBM обучены на одних и тех же признаках и хронологических выборках. Для оценки метрик
используются созревшие метки. Таблица показывает, сколько исходов известно к дате среза и насколько редок
положительный класс.</p>
{label_table}
<p>ROC-AUC оценивает качество ранжирования. PR-AUC в таблице рассчитан как Average Precision. Он учитывает
точность на разных уровнях полноты и помогает сравнить модели при редком fraud. Верхние K — операции с
наибольшим скором, где K соответствует округлённой вверх доле лимита ручной проверки от размеченной выборки.
Recall@K показывает, какая доля всего fraud попала в этот набор. Precision@K показывает долю fraud
внутри набора.</p>
{metric_table}
<h3>Выбор модели для политики</h3>
<p>Для каждой модели на валидации подобраны пороги при одинаковых бизнес-ограничениях. Выбрана
<strong>{_text(selected)}</strong> с наименьшими ожидаемыми потерями. При равных потерях предпочтение получает
модель с большей AP на валидации. Решение зафиксировано перед тестом.</p>
{candidates}{importance_note}
<p class="source-line">Полные метрики и важности доступны в <code>risk_metrics.csv</code> и
<code>feature_importance.csv</code>.</p>
</section>
"""


def _policy_figure(approve_all: dict, optimized: dict) -> str:
    """Рисует разложение тестовых потерь на три денежных компонента.

    Сводки двух политик рассчитаны на одной когорте с созревшими метками.
    Сегменты столбца показывают ущерб от fraud, проверки и ложные отказы.
    Возвращается HTML со встроенным PNG либо пустая строка без обеих сводок.
    """
    if not approve_all or not optimized:
        return ""
    figure = Figure(figsize=(10.6, 2.55), dpi=125, layout="constrained")
    axis = figure.subplots()
    _style_axis(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_yticks([0, 1], ["Пороговая политика", "Одобрять все"])
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
    axis.set_xlabel("Ожидаемые потери на тесте, ден. ед.", color=MUTED, fontsize=9)
    components = (
        ("fraud_loss", "Fraud loss", BLUE),
        ("review_cost", "Проверки", TEAL),
        ("false_decline_cost", "Ложные отклонения", AMBER),
    )
    # Каждый компонент начинается там, где закончился предыдущий, поэтому
    # полная длина столбца соответствует общей стоимости политики.
    offsets = [0.0, 0.0]
    for key, label, color in components:
        widths = [optimized[key], approve_all[key]]
        axis.barh([0, 1], widths, height=0.48, left=offsets, color=color, label=label)
        offsets = [offset + width for offset, width in zip(offsets, widths)]
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncols=3, frameon=False, fontsize=9)
    return _figure(
        figure,
        "Состав ожидаемых тестовых потерь для пороговой политики и одобрения всех операций.",
        "Денежные оценки построены на одной тестовой когорте с созревшими метками.",
    )


def _policy_section(risk: dict) -> str:
    """Объясняет пороги, бизнес-ограничения и стоимость политики в HTML.

    `risk` содержит параметры симуляции, пороги и результаты обеих политик
    на валидации и тесте. Таблицы показывают объём решений на всех операциях
    и ожидаемые потери по когорте с созревшими метками.
    """
    costs = risk.get("costs", {})
    thresholds = risk.get("thresholds", {})
    outcomes = risk.get("policy_metrics", [])
    baseline, optimized = _test_policies(risk)
    policy_names = {"approve_all": "Одобрять все", "optimized": "Пороговая политика"}
    policy_table = _table(
        ("Период и политика", "Одобрено", "На проверку", "Отклонено", "Доля проверок",
         "Fraud loss, ден. ед.", "Остаток fraud loss", "Ложные отклонения, ожид.", "Итого, ден. ед."),
        (
            (f"{SPLIT_LABELS[row['split']]} · {policy_names[row['policy']]}",
             _number(row["approve_count"]), _number(row["review_count"]), _number(row["reject_count"]),
             _share(row["review_rate"], 2), _number(row["fraud_loss"], 2),
             _share(row["fraud_loss_ratio"], 2), _number(row["false_decline_count_expected"], 2),
             _number(row["total_cost"], 2))
            for row in outcomes
        ),
    )
    # Лимиты оцениваются для оптимизированной политики. Базовая политика
    # даёт точку сравнения для объёмов решений и финансовых результатов.
    limit_check = _table(
        ("Период", "Лимит ручной проверки", "Лимит fraud loss"),
        (
            (SPLIT_LABELS[row["split"]], "Соблюдён" if row["review_limit_met"] else "Превышен",
             "Соблюдён" if row["fraud_loss_limit_met"] else "Превышен")
            for row in outcomes if row["policy"] == "optimized"
        ),
    )
    return f"""
<section id="policy"><h2>7. Оптимизация двух порогов</h2>
<p>Пороги настраиваются на валидации. Оптимизатор сортирует операции по скору и группирует совпадающие
значения. Накопленные суммы по группам позволяют вычислить для каждого порога отклонения допустимый
диапазон порога проверки. Поиск минимума на этом диапазоне находит лучший вариант среди всех
различных разбиений с соблюдением ограничений. Одинаковые скоры получают одинаковое решение.</p>
<ul>
<li>При <code>score ≤ review_threshold</code> операция одобряется. Порог проверки равен
<strong>{_number(thresholds.get("review_threshold"), 6)}</strong>.</li>
<li>При <code>review_threshold &lt; score ≤ reject_threshold</code> операция направляется на ручную проверку.</li>
<li>При <code>score &gt; reject_threshold</code> операция отклоняется. Порог отклонения равен
<strong>{_number(thresholds.get("reject_threshold"), 6)}</strong>.</li>
</ul>
<p>Доля ручных проверок ограничена {_share(costs.get("review_capacity"), 2)} от всех операций
валидационного периода. Допустимый остаток fraud loss равен {_share(costs.get("max_fraud_loss_ratio"), 2)}
от сценария одобрения всех операций. Лимит проверок применяется к периоду в целом.</p>
<h3>Функция потерь</h3>
<ul>
<li><strong>Fraud loss.</strong> Сумма мошеннических операций в одобрении и ожидаемая пропущенная сумма
в проверке, умноженная на долю ущерба при мошенничестве (LGD {_share(costs.get("fraud_lgd"))}). Допущение
о проверке — она выявляет {_share(costs.get("review_fraud_catch_rate"))} мошеннической суммы в очереди.</li>
<li><strong>Проверки.</strong> Число проверок в размеченной когорте × {_number(costs.get("review_cost"), 2)} ден. ед.</li>
<li><strong>Ложные отклонения.</strong> Легитимные автоматические отклонения плюс ожидаемая доля ошибочно
отклонённых легитимных операций после проверки ({_share(costs.get("review_false_decline_rate"))}).
Одно ложное отклонение стоит {_number(costs.get("false_decline_cost"), 2)} ден. ед.</li>
</ul>
<p>Минимизируется сумма трёх компонентов. Денежные оценки и ложные отклонения рассчитаны по операциям
с созревшими метками. Операционные объёмы и доля проверок включают все операции соответствующего периода.</p>
{policy_table}{limit_check}{_policy_figure(baseline, optimized)}
<p class="source-line">Пороги полной точности доступны в <code>risk_summary.json</code>, сравнение политик —
в <code>policy_metrics.csv</code>, решения по операциям теста — в <code>test_scored.csv</code>.</p>
</section>
"""


def _test_policies(risk: dict) -> tuple[dict, dict]:
    """Возвращает тестовые сводки `approve_all` и `optimized` в таком порядке.

    Для отсутствующей политики возвращается пустой словарь.
    """
    outcomes = risk.get("policy_metrics", [])
    test = {row["policy"]: row for row in outcomes if row["split"] == "test"}
    return test.get("approve_all", {}), test.get("optimized", {})


def _reduction(before: object, after: object) -> float | None:
    """Выражает изменение как долю снижения относительно исходного значения.

    Возвращает `(before − after) / before`. Для нулевой или отсутствующей базы
    и для отсутствующего результата возвращает `None`.
    """
    if before and after is not None:
        return (float(before) - float(after)) / float(before)
    return None


def _overview_section(run_info: dict, risk: dict) -> str:
    """Формирует вводные карточки по тесту и отмечает происхождение данных.

    Снижение ущерба и общих потерь рассчитано относительно одобрения всех
    операций. Остальные карточки показывают объём ручных проверок и ожидаемое
    число ложных отказов. Возвращается фрагмент HTML.
    """
    baseline, optimized = _test_policies(risk)
    improvement_fraud = _reduction(baseline.get("fraud_loss"), optimized.get("fraud_loss"))
    improvement_total = _reduction(baseline.get("total_cost"), optimized.get("total_cost"))
    cards = "".join((
        _card(_share(improvement_fraud, 1), "Снижение fraud loss"),
        _card(_number(optimized.get("review_count")), f"Ручных проверок · {_share(optimized.get('review_rate'), 2)} операций"),
        _card(_number(optimized.get("false_decline_count_expected"), 2), "Ожидаемых ложных отклонений"),
        _card(_share(improvement_total, 1), "Снижение совокупных потерь"),
    ))
    synthetic = run_info.get("synthetic") or {}
    if synthetic:
        population = "на синтетических данных"
    else:
        population = "на предоставленных данных"
    return f"""
<section id="summary" aria-labelledby="summary-title"><h2 id="summary-title">Результат теста</h2>
<p>Пороговая политика с моделью <strong>{_text(risk.get("selected_model", "—"))}</strong> проверена
{population} в период с {_utc(run_info.get("validation_end"))} до {_utc(run_info.get("end"))} UTC.
Изменения измеряются относительно сценария «одобрять все».</p>
<div class="kpis">{cards}</div>
</section>
"""


def _business_section(run_info: dict, risk: dict) -> str:
    """Формирует итог об эффективности, ёмкости и ограничениях симуляции.

    `risk` даёт фактическую долю проверок в симуляции, лимиты и ожидаемые
    финансовые результаты. `run_info` определяет применимые оговорки для
    генератора или пользовательских данных. Возвращается фрагмент HTML.
    """
    _, optimized = _test_policies(risk)
    costs = risk.get("costs", {})
    capacity = costs.get("review_capacity")
    review_met = optimized.get("review_limit_met")
    if review_met is False:
        # Лимит — доля от всех операций. Округление вниз даёт максимальное
        # допустимое целое число проверок, а добавка сглаживает погрешность float.
        max_reviews = floor(capacity * optimized["transactions"] + 1e-9)
        extra_reviews = optimized["review_count"] - max_reviews
        capacity_note = (
            f'<div class="note warning"><p>На тесте фиксированные пороги отправили на проверку '
            f'{_share(optimized["review_rate"], 2)} операций при лимите {_share(capacity, 2)}. '
            f'Это на {_number(extra_reviews)} операций выше ёмкости за период. Для внедрения требуется '
            f'проверка лимита на свежем потоке и отдельное правило при переполнении очереди.</p></div>'
        )
    elif review_met:
        capacity_note = (
            f'<div class="note"><p>На тесте доля ручных проверок {_share(optimized["review_rate"], 2)} '
            f'уложилась в лимит {_share(capacity, 2)} за период. Часовой пик оценивается отдельно '
            f'по прогнозу нагрузки.</p></div>'
        )
    else:
        capacity_note = ""
    fraud_met = optimized.get("fraud_loss_limit_met")
    fraud_note = ""
    if optimized.get("fraud_loss_ratio") is not None:
        fraud_condition = "в пределах" if fraud_met else "выше"
        fraud_note = (
            f'<p>Остаточный fraud loss составил {_share(optimized["fraud_loss_ratio"], 2)} '
            f'от базового уровня и оказался {fraud_condition} лимита '
            f'{_share(costs.get("max_fraud_loss_ratio"), 2)}. Среди легитимных операций с созревшей меткой '
            f'ожидаемая доля ложных отклонений — {_share(optimized.get("false_decline_rate_expected"), 2)}. '
            f'Она включает {_number(optimized.get("auto_false_decline_count"))} автоматических отказов '
            f'и {_number(optimized.get("review_false_decline_expected"), 2)} ожидаемых отказов после проверки.</p>'
        )
    has_demo = bool(run_info.get("synthetic"))
    data_limit = (
        "Генератор задаёт связь между признаками, fraud и аномалиями. Численные эффекты демонстрационного "
        "прогона требуют проверки на истории платёжного сервиса." if has_demo else
        "Исторические метки могут зависеть от действующей политики. Измерения требуют отдельного внимания "
        "к отклонённым платежам и к операциям без подтверждённого исхода."
    )
    return f"""
<section id="business"><h2>8. Бизнес-вывод и ограничения</h2>
<p>Автоматическое отклонение защищает от крупного ущерба в верхнем риск-сегменте. Ручная проверка
создаёт промежуточное решение с собственной стоимостью и нагрузкой на команду. Снижение финансовых
потерь нужно оценивать вместе с ложными отказами и пропускной способностью очереди.</p>
{fraud_note}{capacity_note}
<h3>Область применимости</h3>
<ul>
<li>{data_limit} Истинная результативность ручной проверки зависит от состава операций и
пропускной способности команды.</li>
<li>Денежные результаты — ретроспективная симуляция со статичными LGD, стоимостью отказа и качеством проверки.
Пороговый score используется для ранжирования. Для интерпретации score как вероятности нужна отдельная калибровка.</li>
<li>Прогноз ручной проверки предполагает постоянную долю проверок. Она может меняться при дрейфе риска,
даже когда прогноз количества платежей точен. Детектор ориентирован на всплески. Мониторинг спадов
нагрузки требует отдельного порога нижнего хвоста.</li>
</ul>
<h3>Следующая итерация</h3>
<ol>
<li>Проверить метки и экономические параметры на реальной истории и повторить оценку на нескольких
последовательных временных срезах.</li>
<li>Запустить теневой расчёт решений и измерить пиковую нагрузку очереди, время обработки и качество
ручной проверки. Для планирования смен ввести прогноз доли проверок и почасовую ёмкость.</li>
<li>Сверять тревоги с инцидентами, акциями и сбоями. Контролировать зрелость меток, дрейф скора,
false decline и полный ущерб после обновления политики.</li>
</ol>
</section>
"""


def write_report(out: Path, *, run_info: dict, audit: dict, risk: dict, forecast: dict) -> Path:
    """Записывает автономный `report.html` в `out` и возвращает путь к файлу.

    `out` — существующий каталог результатов с CSV-файлами прогона.
    `run_info` задаёт источник и временные границы, `audit` описывает очистку,
    `risk` содержит метрики моделей и политик, `forecast` — итог прогноза
    и детекции. Разделы записываются в UTF-8. CSS и графики входят в HTML.
    """
    out = Path(out)
    path = out / "report.html"
    # Краткий тестовый результат открывает документ. Дальше идут этапы
    # конвейера, а бизнес-интерпретация опирается на уже показанные таблицы.
    sections = (
        _overview_section(run_info, risk),
        _data_section(run_info, audit),
        _marts_section(),
        _forecast_section(out, forecast),
        _anomaly_section(out, run_info, forecast),
        _risk_section(out, risk),
        _policy_section(risk),
        _business_section(run_info, risk),
    )
    path.write_text(
        f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Риск платежей и нагрузка по часам</title><style>{CSS}</style></head>
<body><header><div class="wrap">
<h1>Риск платежей и нагрузка по часам</h1>
<p>SQL-витрины, прогноз нагрузки, аномалии, скоринг операций и политика с двумя порогами.</p>
<nav aria-label="Разделы отчёта">
<a href="#data">Данные</a><a href="#sql">Признаки</a><a href="#forecast">Прогноз</a>
<a href="#anomaly">Аномалии</a><a href="#risk">Скоринг</a><a href="#policy">Пороги</a>
<a href="#business">Бизнес-вывод</a>
</nav></div></header>
<main class="wrap">{"".join(sections)}</main>
<footer><div class="wrap">Отчёт сформирован из результатов запуска Python и PostgreSQL. Все часы приведены к UTC.</div></footer>
</body></html>\n""",
        encoding="utf-8",
    )
    return path
