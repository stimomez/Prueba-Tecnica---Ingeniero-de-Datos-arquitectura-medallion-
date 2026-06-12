"""Motor de calidad de datos (observabilidad transversal del pipeline).

Cada función de check devuelve un dict con un esquema ESTÁNDAR, de modo
que checks de cualquier tipo y de cualquier capa (Bronze/Silver/Gold)
se acumulan en una sola tabla consultable:

    check_name | table | column | records_checked | records_failed
    | pct_failed | stage | executed_at

La clase QualityCollector junta esos dicts y los materializa en CSV y
Parquet, para que el evaluador pueda abrirlos con Excel o consultarlos
con SQL (DuckDB).
"""
from datetime import datetime, timezone

import polars as pl

from src import config


# ---------------------------------------------------------------------------
# Constructor del registro estándar
# ---------------------------------------------------------------------------
def _record(
    check_name: str,
    table: str,
    column: str | None,
    records_checked: int,
    records_failed: int,
    stage: str,
) -> dict:
    """Arma una fila de calidad con el esquema estándar."""
    pct = round(records_failed / records_checked * 100, 2) if records_checked else 0.0
    return {
        "check_name": check_name,
        "table": table,
        "column": column,
        "records_checked": records_checked,
        "records_failed": records_failed,
        "pct_failed": pct,
        "stage": stage,
        "executed_at": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# Checks individuales (cada uno devuelve un dict)
# ---------------------------------------------------------------------------
def row_count(df: pl.DataFrame, table: str, stage: str) -> dict:
    """Registra cuántas filas tiene la tabla en una etapa (entraron vs salieron)."""
    return _record("row_count", table, None, df.height, 0, stage)


def null_check(df: pl.DataFrame, column: str, table: str, stage: str) -> dict:
    """Cuenta valores nulos en una columna."""
    failed = df.select(pl.col(column).is_null().sum()).item()
    return _record("null_check", table, column, df.height, failed, stage)


def duplicate_check(df: pl.DataFrame, key: str, table: str, stage: str) -> dict:
    """Cuenta registros duplicados por una clave (filas que se eliminarían)."""
    failed = df.height - df.select(pl.col(key).n_unique()).item()
    return _record("duplicate_check", table, key, df.height, failed, stage)


def range_check(
    df: pl.DataFrame,
    column: str,
    table: str,
    stage: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> dict:
    """Cuenta valores fuera del rango [min_value, max_value].

    Asume que la columna ya está casteada a numérico (capa Silver).
    Los nulos se ignoran aquí: se contabilizan en null_check.
    """
    col = pl.col(column)
    conditions = []
    if min_value is not None:
        conditions.append(col < min_value)
    if max_value is not None:
        conditions.append(col > max_value)

    if not conditions:
        failed = 0
    else:
        out_of_range = conditions[0]
        for cond in conditions[1:]:
            out_of_range = out_of_range | cond
        # & col.is_not_null() para no contar nulos como fuera de rango
        failed = df.select((out_of_range & col.is_not_null()).sum()).item()

    return _record("range_check", table, column, df.height, failed, stage)


def referential_check(
    child_df: pl.DataFrame,
    child_col: str,
    parent_df: pl.DataFrame,
    parent_col: str,
    table: str,
    stage: str,
) -> dict:
    """Cuenta filas hijas cuya FK (no nula) no existe en la tabla padre."""
    parent_keys = parent_df.select(pl.col(parent_col).unique())
    fk = pl.col(child_col)
    orphans = (
        child_df.filter(fk.is_not_null())
        .join(parent_keys, left_on=child_col, right_on=parent_col, how="anti")
        .height
    )
    return _record("referential_check", table, child_col, child_df.height, orphans, stage)


# ---------------------------------------------------------------------------
# Colector
# ---------------------------------------------------------------------------
class QualityCollector:
    """Acumula registros de calidad y los materializa en CSV + Parquet."""

    SCHEMA_ORDER = [
        "check_name", "table", "column", "records_checked",
        "records_failed", "pct_failed", "stage", "executed_at",
    ]

    def __init__(self) -> None:
        self._records: list[dict] = []

    def add(self, record: dict | list[dict]) -> None:
        """Agrega un registro o una lista de registros."""
        if isinstance(record, list):
            self._records.extend(record)
        else:
            self._records.append(record)

    def to_frame(self) -> pl.DataFrame:
        """Devuelve los registros acumulados como DataFrame ordenado."""
        return pl.DataFrame(self._records).select(self.SCHEMA_ORDER)

    def write(self, filename: str = "quality_checks") -> pl.DataFrame:
        """Escribe la tabla de calidad en data/quality/ (CSV + Parquet)."""
        df = self.to_frame()
        df.write_csv(config.QUALITY_DIR / f"{filename}.csv")
        df.write_parquet(config.QUALITY_DIR / f"{filename}.parquet")
        return df


# ---------------------------------------------------------------------------
# Perfilado de una entidad (orquestación reutilizable)
# ---------------------------------------------------------------------------
def profile_entity(
    df: pl.DataFrame, table: str, stage: str, primary_key: str
) -> list[dict]:
    """Perfilado estándar de una entidad: row_count + nulos + duplicados.

    Excluye las columnas de metadata (las que empiezan con '_') del
    chequeo de nulos, ya que son agregadas por el pipeline, no del origen.
    """
    records = [row_count(df, table, stage)]

    for column in df.columns:
        if column.startswith("_"):
            continue
        records.append(null_check(df, column, table, stage))

    records.append(duplicate_check(df, primary_key, table, stage))
    return records


# ---------------------------------------------------------------------------
# CLI: perfila las tablas Bronze ya generadas
# ---------------------------------------------------------------------------
def run() -> pl.DataFrame:
    """Perfila todas las entidades de la capa Bronze y escribe la tabla."""
    config.ensure_dirs()
    collector = QualityCollector()

    for entity in config.ENTITIES:
        df = pl.read_parquet(config.BRONZE_DIR / f"{entity}.parquet")
        pk = config.PRIMARY_KEYS[entity]
        collector.add(profile_entity(df, table=entity, stage="bronze", primary_key=pk))
        print(f"[quality] perfilado bronze: {entity}")

    result = collector.write()
    print(f"[quality] {result.height} checks escritos en data/quality/")
    return result


if __name__ == "__main__":
    run()