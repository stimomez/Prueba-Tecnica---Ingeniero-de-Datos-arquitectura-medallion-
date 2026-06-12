"""Capa Bronze: ingesta cruda de los CSV a Parquet.

Principio de diseño: Bronze preserva, no interpreta. Los CSV se leen
con TODAS las columnas como string (infer_schema_length=0) para que
ningún valor corrupto (un precio "abc", una fecha malformada) se pierda
o se altere durante la ingesta. El casteo y la limpieza son
responsabilidad de la capa Silver.

Se agrega metadata técnica de ingesta:
  - _source_file: archivo CSV de origen
  - _ingested_at: timestamp UTC de la ingesta
"""
from datetime import datetime, timezone

import polars as pl

from src import config


def ingest_entity(entity: str, source_file: str) -> pl.DataFrame:
    """Lee un CSV crudo y le agrega metadata de ingesta.

    Args:
        entity: nombre lógico de la entidad (ej. "customers").
        source_file: nombre del archivo CSV en data/raw/.

    Returns:
        DataFrame con todas las columnas como string más la metadata.
    """
    source_path = config.RAW_DIR / source_file

    # infer_schema_length=0 -> todas las columnas se leen como Utf8.
    # Máxima fidelidad al origen: nada se castea ni se pierde en Bronze.
    df = pl.read_csv(source_path, infer_schema_length=0)

    ingested_at = datetime.now(timezone.utc)
    df = df.with_columns(
        pl.lit(source_file).alias("_source_file"),
        pl.lit(ingested_at).alias("_ingested_at"),
    )
    return df


def write_bronze(df: pl.DataFrame, entity: str) -> None:
    """Escribe el DataFrame de una entidad en data/bronze/ como Parquet."""
    output_path = config.BRONZE_DIR / f"{entity}.parquet"
    df.write_parquet(output_path)


def run() -> dict[str, int]:
    """Ejecuta la ingesta Bronze para todas las entidades.

    Returns:
        Diccionario {entidad: número de filas ingestadas}.
    """
    config.ensure_dirs()
    summary: dict[str, int] = {}

    for entity, source_file in config.ENTITIES.items():
        df = ingest_entity(entity, source_file)
        write_bronze(df, entity)
        summary[entity] = df.height
        print(f"[bronze] {entity:<12} {df.height:>6} filas -> {entity}.parquet")

    return summary


if __name__ == "__main__":
    run()