"""Capa Silver: limpieza, normalización y validación por entidad.

Cada entidad tiene una función clean_<entity> que compone las
transformaciones puras de cleaning/validation según sus reglas de
negocio y devuelve (df_limpio, [piezas_de_cuarentena]).

El orden general dentro de cada entidad es:
    normalizar texto -> castear/parsear -> deduplicar -> imputar
    centinelas -> cuarentena/anulación de inválidos

run() orquesta las 6 entidades, escribe los Parquet de Silver y de
cuarentena, y registra los quality checks de la etapa 'silver' en la
misma tabla acumulada (bronze + silver).
"""
import polars as pl

from src import config, quality
from src.transforms import cleaning, validation


# ---------------------------------------------------------------------------
# Limpieza por entidad
# ---------------------------------------------------------------------------
def clean_customers(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.normalize_text(df, "state", case="upper")
    df = cleaning.normalize_text(df, "email", case="lower")
    df = cleaning.parse_datetime(df, "created_at")
    df = cleaning.dedup_by_key(df, "customer_id")
    # state alimenta gold_sales_by_state: centinela en vez de descartar.
    df = cleaning.fill_categorical(df, "state", "UNKNOWN")
    # name/email/city: nulos no críticos, se conservan tal cual.
    return df, []


def clean_products(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.normalize_text(df, "category", case="lower")
    df = cleaning.cast_numeric(df, "price", pl.Float64)
    df = cleaning.cast_numeric(df, "weight_kg", pl.Float64)
    df = cleaning.dedup_by_key(df, "product_id")
    df = cleaning.fill_categorical(df, "category", "unknown")
    # peso negativo: dato imposible pero no crítico -> se anula, fila se queda.
    df = validation.null_out_where(df, "weight_kg", pl.col("weight_kg") < config.WEIGHT_MIN)
    # precio: crítico para revenue -> cuarentena.
    clean, q_null = validation.quarantine_where(df, pl.col("price").is_null(), "null_price")
    clean, q_neg = validation.quarantine_where(
        clean, pl.col("price") <= config.PRICE_MIN, "price_no_positivo"
    )
    return clean, [q_null, q_neg]


def clean_orders(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.normalize_text(df, "status", case="lower")
    df = cleaning.parse_datetime(df, "order_date")
    df = cleaning.parse_datetime(df, "approved_at")
    df = cleaning.parse_datetime(df, "delivered_at")
    df = cleaning.dedup_by_key(df, "order_id")
    df = cleaning.fill_categorical(df, "status", "unknown")
    # order_date es el evento base de los KPIs mensuales: sin fecha -> cuarentena.
    # approved_at/delivered_at: nulos legítimos (no aprobado / no entregado).
    clean, q_date = validation.quarantine_where(
        df, pl.col("order_date").is_null(), "null_order_date"
    )
    return clean, [q_date]


def clean_order_items(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.cast_numeric(df, "quantity", pl.Int64)
    df = cleaning.cast_numeric(df, "unit_price", pl.Float64)
    df = cleaning.cast_numeric(df, "freight_value", pl.Float64)
    df = cleaning.dedup_by_key(df, "item_id")
    df = validation.null_out_where(df, "freight_value", pl.col("freight_value") < 0)
    # unit_price y quantity: críticos para revenue (quantity * unit_price).
    clean, q_up_null = validation.quarantine_where(
        df, pl.col("unit_price").is_null(), "null_unit_price"
    )
    clean, q_up_neg = validation.quarantine_where(
        clean, pl.col("unit_price") <= config.PRICE_MIN, "unit_price_no_positivo"
    )
    clean, q_qty_null = validation.quarantine_where(
        clean, pl.col("quantity").is_null(), "null_quantity"
    )
    clean, q_qty_inv = validation.quarantine_where(
        clean, pl.col("quantity") < config.QUANTITY_MIN, "quantity_invalida"
    )
    return clean, [q_up_null, q_up_neg, q_qty_null, q_qty_inv]


def clean_payments(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.normalize_text(df, "payment_type", case="lower")
    df = cleaning.cast_numeric(df, "installments", pl.Int64)
    df = cleaning.cast_numeric(df, "amount", pl.Float64)
    df = cleaning.dedup_by_key(df, "payment_id")
    df = cleaning.fill_categorical(df, "payment_type", "unknown")
    # un pago debe ser positivo; montos <= 0 se consideran inválidos.
    clean, q_null = validation.quarantine_where(df, pl.col("amount").is_null(), "null_amount")
    clean, q_neg = validation.quarantine_where(
        clean, pl.col("amount") <= 0, "amount_no_positivo"
    )
    return clean, [q_null, q_neg]


def clean_reviews(df: pl.DataFrame) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    df = cleaning.cast_numeric(df, "score", pl.Int64)
    df = cleaning.dedup_by_key(df, "review_id")
    # score fuera de 1-5: se anula (no confiable) pero la reseña se conserva,
    # porque el order_id y la existencia de la reseña siguen siendo útiles.
    # score null tampoco se descarta: solo no aportará al promedio en Gold.
    df = validation.null_out_where(
        df,
        "score",
        (pl.col("score") < config.REVIEW_SCORE_MIN)
        | (pl.col("score") > config.REVIEW_SCORE_MAX),
    )
    return df, []


CLEANERS = {
    "customers": clean_customers,
    "products": clean_products,
    "orders": clean_orders,
    "order_items": clean_order_items,
    "payments": clean_payments,
    "reviews": clean_reviews,
}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run() -> pl.DataFrame:
    """Ejecuta la limpieza Silver de todas las entidades.

    Escribe data/silver/<entity>.parquet, las cuarentenas en
    data/quality/quarantine_<entity>.parquet, y acumula los quality
    checks de la etapa 'silver' junto a los de 'bronze'.
    """
    config.ensure_dirs()
    collector = quality.QualityCollector()

    for entity, cleaner in CLEANERS.items():
        bronze = pl.read_parquet(config.BRONZE_DIR / f"{entity}.parquet")
        input_height = bronze.height

        clean, q_pieces = cleaner(bronze)
        clean.write_parquet(config.SILVER_DIR / f"{entity}.parquet")

        # Cuarentena: une las piezas no vacías y registra conteos por motivo.
        pieces = [p for p in q_pieces if p.height > 0]
        quarantine_total = sum(p.height for p in pieces)
        if pieces:
            q_df = pl.concat(pieces, how="diagonal_relaxed")
            q_df.write_parquet(config.QUALITY_DIR / f"quarantine_{entity}.parquet")
            by_reason = q_df.group_by(validation.QUARANTINE_COL).len()
            for row in by_reason.iter_rows(named=True):
                collector.add(
                    quality.quarantine_record(
                        reason=row[validation.QUARANTINE_COL],
                        table=entity,
                        records_checked=input_height,
                        records_failed=row["len"],
                        stage="silver",
                    )
                )

        # Perfil post-limpieza (nulos restantes, duplicados=0, row_count).
        collector.add(
            quality.profile_entity(
                clean, table=entity, stage="silver", primary_key=config.PRIMARY_KEYS[entity]
            )
        )

        # dedup y cuarentena son descartes distintos: se reportan por separado.
        dedup_removed = input_height - clean.height - quarantine_total
        print(
            f"[silver] {entity:<12} {input_height:>6} -> {clean.height:>6} "
            f"| dedup {dedup_removed:>3} | cuarentena {quarantine_total:>4}"
        )

    return _merge_quality(collector.to_frame())


def _merge_quality(silver_checks: pl.DataFrame) -> pl.DataFrame:
    """Acumula los checks de silver junto a los existentes (idempotente).

    Relee la tabla de calidad, elimina filas previas de la etapa 'silver'
    (para que reejecutar no duplique) y reescribe bronze + silver.
    """
    path = config.QUALITY_DIR / "quality_checks.parquet"
    if path.exists():
        existing = pl.read_parquet(path).filter(pl.col("stage") != "silver")
        combined = pl.concat([existing, silver_checks], how="diagonal_relaxed")
    else:
        combined = silver_checks

    combined.write_csv(config.QUALITY_DIR / "quality_checks.csv")
    combined.write_parquet(path)
    print(f"[silver] checks acumulados (bronze + silver): {combined.height}")
    return combined


if __name__ == "__main__":
    run()