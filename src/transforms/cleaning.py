"""Transformaciones puras de limpieza (capa Silver).

Todas las funciones son puras: reciben un DataFrame y devuelven un
DataFrame nuevo, sin efectos de lado ni I/O. Eso las hace triviales de
testear (se les pasa un DataFrame pequeño construido a mano y se
verifica la salida) y reutilizables en cualquier entidad.
"""
import polars as pl


def normalize_text(df: pl.DataFrame, column: str, *, case: str = "lower") -> pl.DataFrame:
    """Quita espacios sobrantes y unifica mayúsculas/minúsculas.

    Resuelve formatos inconsistentes como 'DELIVERED' vs 'delivered' o
    estados con espacios al inicio/fin. No mapea typos: si hubiera
    variantes mal escritas, se añadiría un diccionario de reemplazo.

    Args:
        case: "lower" o "upper".
    """
    expr = pl.col(column).str.strip_chars()
    if case == "lower":
        expr = expr.str.to_lowercase()
    elif case == "upper":
        expr = expr.str.to_uppercase()
    else:
        raise ValueError(f"case debe ser 'lower' o 'upper', no {case!r}")
    return df.with_columns(expr.alias(column))


def fill_categorical(
    df: pl.DataFrame, column: str, sentinel: str = "unknown"
) -> pl.DataFrame:
    """Imputa un centinela en una columna categórica usada para agrupar.

    Se usa en columnas como state/category/status, donde un nulo haría
    que el registro desaparezca de los GROUP BY de la capa Gold. En vez
    de descartarlo, se conserva bajo un valor explícito ('unknown').
    """
    return df.with_columns(pl.col(column).fill_null(sentinel).alias(column))


def cast_numeric(
    df: pl.DataFrame, column: str, dtype: pl.DataType = pl.Float64
) -> pl.DataFrame:
    """Castea una columna string a numérico.

    strict=False: los valores no parseables (ej. 'abc') se vuelven null
    en vez de lanzar excepción. Esos nulos quedan luego visibles en los
    quality checks de la etapa Silver.
    """
    return df.with_columns(pl.col(column).cast(dtype, strict=False).alias(column))


def parse_datetime(
    df: pl.DataFrame, column: str, fmt: str = "%Y-%m-%d %H:%M:%S"
) -> pl.DataFrame:
    """Parsea una columna string a Datetime.

    strict=False: las fechas con formato inválido se vuelven null. Para
    columnas críticas (ej. order_date) esos nulos se mandan a cuarentena
    en la etapa de validación; para fechas opcionales (approved_at,
    delivered_at) el nulo es un estado de negocio válido.
    """
    return df.with_columns(
        pl.col(column).str.to_datetime(fmt, strict=False).alias(column)
    )


def dedup_by_key(df: pl.DataFrame, key: str) -> pl.DataFrame:
    """Elimina registros duplicados por una clave, conservando el primero.

    keep='first' + maintain_order=True => sobrevive la primera aparición
    en el orden del archivo. Decisión determinista e independiente del
    parseo de fechas. Si el negocio exigiera el registro más reciente,
    se ordenaría por la columna de fecha descendente antes de deduplicar.
    """
    return df.unique(subset=[key], keep="first", maintain_order=True)