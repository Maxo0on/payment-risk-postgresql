"""Точка входа для команды python -m payment_risk."""

from .cli import main

# SystemExit передаёт оболочке код завершения, по которому CI проверяет успех запуска.
raise SystemExit(main())
