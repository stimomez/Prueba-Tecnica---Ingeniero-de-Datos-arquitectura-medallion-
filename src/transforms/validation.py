"""Validaciones puras para la capa Silver (cuarentena y saneamiento).

Dos estrategias frente a un valor inválido:

  1. Cuarentena (quarantine_where): la fila entera se aparta a una tabla
     de descartes con un motivo. Se usa cuando el registro no sirve para
     el análisis (ej. un item de pedido sin precio: no aporta revenue).

  2. Anulación (null_out_where): la fila se conserva pero el valor
     inválido se vuelve null. Se usa cuando el resto del registro sigue
     siendo útil (ej. una reseña con score fuera de rango: el order_id y
     la existencia de la reseña valen, solo no confiamos en el score).

Todas las funciones son puras (DataFrame -> DataFrame). La columna
_quarantine_reason documenta por qué se descartó cada fila, respondiendo
directamente a "¿cuántos registros fueron descartados y por qué?".
"""
import polars as pl

QUARANTINE_COL = "_quarantine_reason"


def quarantine_where(
    df: pl.DataFrame, invalid_mask: pl.Expr, reason: str
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separa las filas que cumplen invalid_mask hacia una cuarentena.

    Los nulos en la máscara (p. ej. una comparación sobre un valor null)
    se tratan como NO inválidos: esas filas se conservan. La criticidad
    de los nulos se maneja con una máscara explícita is_null() aparte.

    Returns:
        (validos, cuarentena). La cuarentena lleva _quarantine_reason.
    """
    mask = invalid_mask.fill_null(False)
    invalid = df.filter(mask).with_columns(pl.lit(reason).alias(QUARANTINE_COL))
    valid = df.filter(~mask)
    return valid, invalid


def null_out_where(
    df: pl.DataFrame, column: str, invalid_mask: pl.Expr
) -> pl.DataFrame:
    """Pone en null los valores de `column` donde invalid_mask es verdadero.

    Conserva la fila completa; solo neutraliza el valor no confiable.
    """
    mask = invalid_mask.fill_null(False)
    dtype = df.schema[column]
    return df.with_columns(
        pl.when(mask)
        .then(pl.lit(None, dtype=dtype))
        .otherwise(pl.col(column))
        .alias(column)
    )