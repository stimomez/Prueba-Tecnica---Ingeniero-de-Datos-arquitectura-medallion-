"""Capa Gold: tablas analíticas sobre los datos limpios de Silver.

Cuatro vistas de negocio:
  - gold_sales_by_state      : revenue, pedidos y ticket promedio por estado del cliente.
  - gold_product_performance : ventas, score promedio y tasa de devolución por producto.
  - gold_customer_segments   : segmentación RFM simplificada (Recency/Frequency/Monetary).
  - gold_monthly_kpis        : KPIs mensuales (revenue, órdenes, cancelaciones, score).

Todos los outputs se escriben como Parquet en data/gold/ y se registran
en la tabla de calidad bajo la etapa 'gold'.
"""
import polars as pl

from src import config, quality


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------
def _order_revenue(items: pl.DataFrame) -> pl.DataFrame:
    """Revenue total por order_id (sum unit_price * quantity)."""
    return (
        items.with_columns(
            (pl.col("unit_price") * pl.col("quantity")).alias("line_revenue")
        )
        .group_by("order_id")
        .agg(pl.col("line_revenue").sum().alias("order_revenue"))
    )


def _score_col(df: pl.DataFrame, col: str, ascending: bool) -> pl.Expr:
    """Expresión de scoring 1–4 por cuartiles sobre una columna numérica.

    ascending=True  → valores más altos = score más alto (frequency, monetary).
    ascending=False → valores más bajos = score más alto (recency_days).
    """
    s = df.get_column(col).drop_nulls()
    q1, q2, q3 = s.quantile(0.25), s.quantile(0.50), s.quantile(0.75)
    if ascending:
        expr = (
            pl.when(pl.col(col) <= q1).then(1)
            .when(pl.col(col) <= q2).then(2)
            .when(pl.col(col) <= q3).then(3)
            .otherwise(4)
        )
    else:
        expr = (
            pl.when(pl.col(col) <= q1).then(4)
            .when(pl.col(col) <= q2).then(3)
            .when(pl.col(col) <= q3).then(2)
            .otherwise(1)
        )
    return expr.alias(f"{col}_score")


# ---------------------------------------------------------------------------
# Tablas Gold
# ---------------------------------------------------------------------------
def build_sales_by_state(silver: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Revenue, pedidos y ticket promedio por estado del cliente.

    Órdenes cuyo customer_id no existe en customers (huérfanos de dimensión,
    no descartados en Silver) quedan bajo state = 'UNKNOWN'.
    """
    orders = silver["orders"]
    items = silver["order_items"]
    customers = silver["customers"]

    return (
        orders
        .join(_order_revenue(items), on="order_id", how="left")
        .join(customers.select(["customer_id", "state"]), on="customer_id", how="left")
        .with_columns(pl.col("state").fill_null("UNKNOWN"))
        .group_by("state")
        .agg(
            pl.col("order_id").n_unique().alias("order_count"),
            pl.col("order_revenue").sum().alias("total_revenue"),
        )
        .with_columns(
            (pl.col("total_revenue") / pl.col("order_count")).round(2).alias("avg_ticket")
        )
        .sort("total_revenue", descending=True)
    )


def build_product_performance(silver: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Ventas, score promedio y tasa de devolución por producto.

    avg_score: promedio del score de las reseñas de los pedidos que
    contienen ese producto. Dado que la reseña es por pedido (no por
    item), la asignación es una aproximación válida para el análisis.

    return_rate: pedidos con status='returned' / total pedidos del producto.
    """
    items = silver["order_items"]
    products = silver["products"]
    orders = silver["orders"]
    reviews = silver["reviews"]

    product_sales = (
        items.with_columns(
            (pl.col("unit_price") * pl.col("quantity")).alias("line_revenue")
        )
        .group_by("product_id")
        .agg(
            pl.col("line_revenue").sum().alias("total_revenue"),
            pl.col("quantity").sum().alias("units_sold"),
            pl.col("order_id").n_unique().alias("order_count"),
        )
    )

    product_scores = (
        items.select(["product_id", "order_id"])
        .join(reviews.select(["order_id", "score"]), on="order_id", how="left")
        .group_by("product_id")
        .agg(pl.col("score").mean().round(2).alias("avg_score"))
    )

    product_returns = (
        items.select(["product_id", "order_id"])
        .join(orders.select(["order_id", "status"]), on="order_id", how="left")
        .group_by("product_id")
        .agg(
            pl.col("order_id").n_unique().alias("total_orders"),
            (pl.col("status") == "returned").sum().alias("returned_orders"),
        )
        .with_columns(
            (pl.col("returned_orders") / pl.col("total_orders")).round(4).alias("return_rate")
        )
        .select(["product_id", "returned_orders", "return_rate"])
    )

    return (
        product_sales
        .join(product_scores, on="product_id", how="left")
        .join(product_returns, on="product_id", how="left")
        .join(
            products.select(["product_id", "product_name", "category"]),
            on="product_id", how="left",
        )
        .with_columns([
            pl.col("product_name").fill_null("UNKNOWN"),
            pl.col("category").fill_null("unknown"),
        ])
        .select([
            "product_id", "product_name", "category",
            "units_sold", "order_count", "total_revenue",
            "avg_score", "returned_orders", "return_rate",
        ])
        .sort("total_revenue", descending=True)
    )


def build_customer_segments(silver: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Segmentación RFM simplificada por cliente.

    Métricas base:
      recency_days : días desde el último pedido hasta la fecha máxima del dataset.
      frequency    : número de pedidos distintos.
      monetary     : gasto total (sum unit_price * quantity de sus pedidos).

    Score 1–4 por cuartiles en cada dimensión; suma → rfm_score (3–12).
    Segmentos: Champions (>=10), Loyal (>=7), Needs Attention (>=4), At Risk (<4).

    Se usa la fecha máxima del dataset como referencia para que el análisis
    sea reproducible independientemente de cuándo se ejecute el pipeline.
    """
    orders = silver["orders"]
    items = silver["order_items"]
    customers = silver["customers"]

    reference_date = orders.get_column("order_date").drop_nulls().max()

    rfm_base = (
        orders
        .join(_order_revenue(items), on="order_id", how="left")
        .group_by("customer_id")
        .agg(
            pl.col("order_date").max().alias("last_order_date"),
            pl.col("order_id").n_unique().alias("frequency"),
            pl.col("order_revenue").sum().alias("monetary"),
        )
        .with_columns(
            (pl.lit(reference_date) - pl.col("last_order_date"))
            .dt.total_days()
            .alias("recency_days")
        )
    )

    return (
        rfm_base.with_columns([
            _score_col(rfm_base, "recency_days", ascending=False),
            _score_col(rfm_base, "frequency", ascending=True),
            _score_col(rfm_base, "monetary", ascending=True),
        ])
        .with_columns(
            (
                pl.col("recency_days_score")
                + pl.col("frequency_score")
                + pl.col("monetary_score")
            ).alias("rfm_score")
        )
        .with_columns(
            pl.when(pl.col("rfm_score") >= 10).then(pl.lit("Champions"))
            .when(pl.col("rfm_score") >= 7).then(pl.lit("Loyal"))
            .when(pl.col("rfm_score") >= 4).then(pl.lit("Needs Attention"))
            .otherwise(pl.lit("At Risk"))
            .alias("segment")
        )
        .join(
            customers.select(["customer_id", "customer_name", "state"]),
            on="customer_id", how="left",
        )
        .select([
            "customer_id", "customer_name", "state",
            "recency_days", "frequency", "monetary",
            "recency_days_score", "frequency_score", "monetary_score",
            "rfm_score", "segment",
        ])
        .sort("rfm_score", descending=True)
    )


def build_monthly_kpis(silver: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """KPIs mensuales: revenue, órdenes, cancelaciones, devoluciones y score.

    La fecha de corte es order_date. Los pedidos sin order_date fueron
    a cuarentena en Silver y no aparecen aquí.
    """
    orders = silver["orders"]
    items = silver["order_items"]
    reviews = silver["reviews"]

    order_scores = (
        reviews.group_by("order_id")
        .agg(pl.col("score").mean().alias("order_score"))
    )

    return (
        orders
        .join(_order_revenue(items), on="order_id", how="left")
        .join(order_scores, on="order_id", how="left")
        .with_columns(pl.col("order_date").dt.truncate("1mo").alias("year_month"))
        .filter(pl.col("year_month").is_not_null())
        .group_by("year_month")
        .agg(
            pl.col("order_id").n_unique().alias("total_orders"),
            pl.col("order_revenue").sum().alias("total_revenue"),
            (pl.col("status") == "canceled").sum().alias("canceled_orders"),
            (pl.col("status") == "returned").sum().alias("returned_orders"),
            pl.col("order_score").mean().round(2).alias("avg_score"),
        )
        .with_columns([
            (pl.col("canceled_orders") / pl.col("total_orders")).round(4).alias("cancellation_rate"),
            (pl.col("returned_orders") / pl.col("total_orders")).round(4).alias("return_rate"),
            (pl.col("total_revenue") / pl.col("total_orders")).round(2).alias("avg_ticket"),
        ])
        .sort("year_month")
    )


BUILDERS = {
    "gold_sales_by_state": build_sales_by_state,
    "gold_product_performance": build_product_performance,
    "gold_customer_segments": build_customer_segments,
    "gold_monthly_kpis": build_monthly_kpis,
}


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def run() -> pl.DataFrame:
    """Construye las 4 tablas Gold y registra quality checks."""
    config.ensure_dirs()
    collector = quality.QualityCollector()

    silver = {
        e: pl.read_parquet(config.SILVER_DIR / f"{e}.parquet")
        for e in config.ENTITIES
    }

    for name, builder in BUILDERS.items():
        df = builder(silver)
        df.write_parquet(config.GOLD_DIR / f"{name}.parquet")
        collector.add(quality.row_count(df, table=name, stage="gold"))
        print(f"[gold] {name:<32} {df.height:>6} filas")

    return _merge_quality(collector.to_frame())


def _merge_quality(gold_checks: pl.DataFrame) -> pl.DataFrame:
    """Acumula los checks de gold (idempotente sobre la etapa)."""
    path = config.QUALITY_DIR / "quality_checks.parquet"
    if path.exists():
        existing = pl.read_parquet(path).filter(pl.col("stage") != "gold")
        combined = pl.concat([existing, gold_checks], how="diagonal_relaxed")
    else:
        combined = gold_checks
    combined.write_csv(config.QUALITY_DIR / "quality_checks.csv")
    combined.write_parquet(path)
    print(f"[gold] checks acumulados: {combined.height}")
    return combined


if __name__ == "__main__":
    run()
