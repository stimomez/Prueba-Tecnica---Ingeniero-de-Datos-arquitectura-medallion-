"""Configuración central del pipeline ETL medallion.

Centraliza rutas, entidades y constantes de negocio para evitar
strings mágicos repartidos por el código.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas base
# ---------------------------------------------------------------------------
# Raíz del proyecto: dos niveles arriba de este archivo (src/config.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
QUALITY_DIR = DATA_DIR / "quality"

# Base DuckDB para consulta analítica (gold + quality)
DUCKDB_PATH = DATA_DIR / "warehouse.duckdb"

# ---------------------------------------------------------------------------
# Entidades del dominio
# ---------------------------------------------------------------------------
# Nombre lógico -> archivo fuente en data/raw/
ENTITIES = {
    "customers": "customers.csv",
    "products": "products.csv",
    "orders": "orders.csv",
    "order_items": "order_items.csv",
    "payments": "payments.csv",
    "reviews": "reviews.csv",
}

# ---------------------------------------------------------------------------
# Reglas de negocio / rangos válidos (capa Silver)
# ---------------------------------------------------------------------------
REVIEW_SCORE_MIN = 1
REVIEW_SCORE_MAX = 5

PRICE_MIN = 0.0          # precios estrictamente positivos
WEIGHT_MIN = 0.0         # peso de producto no puede ser negativo
QUANTITY_MIN = 1         # un item de pedido tiene al menos 1 unidad

# Integridad referencial: tabla hija -> (columna FK, tabla padre, columna PK)
REFERENTIAL_RULES = [
    ("orders", "customer_id", "customers", "customer_id"),
    ("order_items", "order_id", "orders", "order_id"),
    ("order_items", "product_id", "products", "product_id"),
    ("payments", "order_id", "orders", "order_id"),
    ("reviews", "order_id", "orders", "order_id"),
]


def ensure_dirs() -> None:
    """Crea los directorios de las capas si no existen."""
    for d in (BRONZE_DIR, SILVER_DIR, GOLD_DIR, QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)