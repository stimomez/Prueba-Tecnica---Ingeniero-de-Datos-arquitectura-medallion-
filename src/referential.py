"""Capa Silver - integridad referencial (validación cruzada entre entidades).

Estrategia pragmática:
  - Hijos del 'spine' (pedidos): order_items, payments y reviews sin un
    order_id válido NO tienen sentido -> cuarentena (orphan_order_id).
  - Referencias de dimensión (orders->customers, order_items->products):
    la fila sigue siendo válida (el pedido ocurrió; el item tiene su
    propio precio y cantidad). NO se descarta: se cuenta y la dimensión
    faltante se resuelve como 'UNKNOWN' en la capa Gold.

Se ejecuta DESPUÉS de silver.run(): lee las tablas Silver ya limpias,
reescribe los hijos del spine sin sus huérfanos y registra los checks
referenciales en la tabla de calidad bajo la etapa 'referential'.

IMPORTANTE: este paso reescribe algunas tablas Silver, así que debe
correr siempre después de silver.run() (main.py garantiza ese orden).
"""
import polars as pl

from src import config, quality
from src.transforms import validation

# Padre cuya ausencia invalida al hijo (la columna vertebral del modelo).
SPINE_PARENT = "orders"


def run() -> pl.DataFrame:
    """Valida integridad referencial sobre las tablas Silver."""
    config.ensure_dirs()
    collector = quality.QualityCollector()

    silver = {
        e: pl.read_parquet(config.SILVER_DIR / f"{e}.parquet") for e in config.ENTITIES
    }

    for child in config.ENTITIES:
        rules = [r for r in config.REFERENTIAL_RULES if r[0] == child]
        if not rules:
            continue

        cdf = silver[child]
        input_height = cdf.height
        q_pieces: list[pl.DataFrame] = []

        for _child, fk, parent, pk in rules:
            parent_keys = silver[parent].get_column(pk).unique().to_list()

            # Conteo de huérfanos (siempre, informativo) sobre el estado actual.
            collector.add(
                quality.referential_check(
                    cdf, fk, silver[parent], pk, table=child, stage="referential"
                )
            )

            # Cuarentena solo si el padre es el spine (pedidos).
            if parent == SPINE_PARENT:
                orphan_mask = pl.col(fk).is_in(parent_keys).not_() & pl.col(fk).is_not_null()
                cdf, q = validation.quarantine_where(cdf, orphan_mask, f"orphan_{fk}")
                if q.height:
                    q_pieces.append(q)

        if q_pieces:
            # El hijo perdió huérfanos del spine: reescribir Silver y cuarentena.
            silver[child] = cdf
            cdf.write_parquet(config.SILVER_DIR / f"{child}.parquet")

            q_df = pl.concat(q_pieces, how="diagonal_relaxed")
            q_df.write_parquet(
                config.QUALITY_DIR / f"quarantine_referential_{child}.parquet"
            )
            for row in q_df.group_by(validation.QUARANTINE_COL).len().iter_rows(named=True):
                collector.add(
                    quality.quarantine_record(
                        reason=row[validation.QUARANTINE_COL],
                        table=child,
                        records_checked=input_height,
                        records_failed=row["len"],
                        stage="referential",
                    )
                )
            print(
                f"[referential] {child:<12} {input_height:>6} -> {cdf.height:>6} "
                f"(huerfanos {input_height - cdf.height})"
            )
        else:
            print(f"[referential] {child:<12} {input_height:>6} (solo conteo, sin cuarentena)")

    return _merge_quality(collector.to_frame())


def _merge_quality(ref_checks: pl.DataFrame) -> pl.DataFrame:
    """Acumula los checks referenciales (idempotente sobre la etapa)."""
    path = config.QUALITY_DIR / "quality_checks.parquet"
    if path.exists():
        existing = pl.read_parquet(path).filter(pl.col("stage") != "referential")
        combined = pl.concat([existing, ref_checks], how="diagonal_relaxed")
    else:
        combined = ref_checks
    combined.write_csv(config.QUALITY_DIR / "quality_checks.csv")
    combined.write_parquet(path)
    print(f"[referential] checks acumulados: {combined.height}")
    return combined


if __name__ == "__main__":
    run()