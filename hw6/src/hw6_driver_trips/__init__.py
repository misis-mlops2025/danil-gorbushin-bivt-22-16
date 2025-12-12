"""HW6 driver trips package.

Содержит:
- генерацию синтетических данных (offline source для Feast)
- train pipeline (Feast -> train -> MLflow)
- inference pipeline (Feast -> predict -> parquet + MLflow)

Здесь нет зависимостей от hw5/hw4 и т.д. — всё автономно.
"""
