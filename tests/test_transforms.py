"""Tests unitarios para las transformaciones puras de la capa Silver.

Cada test construye un DataFrame pequeño a mano, aplica la función
bajo prueba y verifica la salida. No hay I/O ni dependencias externas.
"""
import polars as pl
import pytest

from src.transforms.cleaning import (
    cast_numeric,
    dedup_by_key,
    fill_categorical,
    normalize_text,
    parse_datetime,
)
from src.transforms.validation import QUARANTINE_COL, null_out_where, quarantine_where


# ---------------------------------------------------------------------------
# normalize_text
# ---------------------------------------------------------------------------
class TestNormalizeText:
    def test_lowercase_and_strip(self):
        df = pl.DataFrame({"status": ["  DELIVERED", "Shipped ", "CANCELED"]})
        result = normalize_text(df, "status", case="lower")
        assert result["status"].to_list() == ["delivered", "shipped", "canceled"]

    def test_uppercase_and_strip(self):
        df = pl.DataFrame({"state": [" sp", "RJ ", "mg"]})
        result = normalize_text(df, "state", case="upper")
        assert result["status" if False else "state"].to_list() == ["SP", "RJ", "MG"]

    def test_invalid_case_raises(self):
        df = pl.DataFrame({"col": ["a"]})
        with pytest.raises(ValueError):
            normalize_text(df, "col", case="title")


# ---------------------------------------------------------------------------
# fill_categorical
# ---------------------------------------------------------------------------
class TestFillCategorical:
    def test_nulls_replaced_with_sentinel(self):
        df = pl.DataFrame({"category": ["electronics", None, "clothing", None]})
        result = fill_categorical(df, "category", "unknown")
        assert result["category"].null_count() == 0
        assert result["category"].to_list() == ["electronics", "unknown", "clothing", "unknown"]

    def test_non_null_values_unchanged(self):
        df = pl.DataFrame({"state": ["SP", "RJ"]})
        result = fill_categorical(df, "state", "UNKNOWN")
        assert result["state"].to_list() == ["SP", "RJ"]


# ---------------------------------------------------------------------------
# cast_numeric
# ---------------------------------------------------------------------------
class TestCastNumeric:
    def test_valid_strings_cast_to_float(self):
        df = pl.DataFrame({"price": ["10.5", "200.0", "0.99"]})
        result = cast_numeric(df, "price", pl.Float64)
        assert result["price"].dtype == pl.Float64
        assert result["price"].to_list() == [10.5, 200.0, 0.99]

    def test_unparseable_becomes_null(self):
        df = pl.DataFrame({"price": ["10.5", "abc", None]})
        result = cast_numeric(df, "price", pl.Float64)
        assert result["price"][0] == 10.5
        assert result["price"][1] is None
        assert result["price"][2] is None

    def test_valid_strings_cast_to_int(self):
        df = pl.DataFrame({"quantity": ["1", "3", "10"]})
        result = cast_numeric(df, "quantity", pl.Int64)
        assert result["quantity"].dtype == pl.Int64
        assert result["quantity"].to_list() == [1, 3, 10]


# ---------------------------------------------------------------------------
# parse_datetime
# ---------------------------------------------------------------------------
class TestParseDatetime:
    def test_valid_datetime_parsed(self):
        df = pl.DataFrame({"order_date": ["2023-05-15 10:30:00"]})
        result = parse_datetime(df, "order_date")
        assert result["order_date"].dtype == pl.Datetime
        assert result["order_date"][0].year == 2023
        assert result["order_date"][0].month == 5

    def test_invalid_format_becomes_null(self):
        df = pl.DataFrame({"order_date": ["not-a-date", "2023-05-15 10:30:00"]})
        result = parse_datetime(df, "order_date")
        assert result["order_date"][0] is None
        assert result["order_date"][1] is not None


# ---------------------------------------------------------------------------
# dedup_by_key
# ---------------------------------------------------------------------------
class TestDedupByKey:
    def test_duplicates_removed_first_kept(self):
        df = pl.DataFrame({
            "order_id": ["A", "B", "A", "C", "B"],
            "value": [1, 2, 3, 4, 5],
        })
        result = dedup_by_key(df, "order_id")
        assert result.height == 3
        assert result["order_id"].to_list() == ["A", "B", "C"]
        assert result["value"].to_list() == [1, 2, 4]

    def test_no_duplicates_all_rows_kept(self):
        df = pl.DataFrame({"id": ["X", "Y", "Z"]})
        result = dedup_by_key(df, "id")
        assert result.height == 3


# ---------------------------------------------------------------------------
# quarantine_where
# ---------------------------------------------------------------------------
class TestQuarantineWhere:
    def test_invalid_rows_go_to_quarantine(self):
        df = pl.DataFrame({"price": [10.0, -5.0, 0.0, 20.0]})
        clean, quarantine = quarantine_where(df, pl.col("price") <= 0, "precio_invalido")
        assert clean.height == 2
        assert quarantine.height == 2
        assert all(r == "precio_invalido" for r in quarantine[QUARANTINE_COL].to_list())

    def test_clean_rows_have_no_quarantine_column(self):
        df = pl.DataFrame({"price": [10.0, 20.0]})
        clean, quarantine = quarantine_where(df, pl.col("price") <= 0, "precio_invalido")
        assert QUARANTINE_COL not in clean.columns
        assert quarantine.height == 0

    def test_null_mask_treated_as_valid(self):
        # Comparar null < 0 produce null en la máscara; debe tratarse como False (fila válida).
        df = pl.DataFrame({"price": [None, 5.0]}, schema={"price": pl.Float64})
        clean, quarantine = quarantine_where(df, pl.col("price") < 0, "precio_negativo")
        assert clean.height == 2
        assert quarantine.height == 0


# ---------------------------------------------------------------------------
# null_out_where
# ---------------------------------------------------------------------------
class TestNullOutWhere:
    def test_invalid_value_becomes_null(self):
        df = pl.DataFrame({"weight_kg": [-1.0, 0.5, -0.1, 2.0]})
        result = null_out_where(df, "weight_kg", pl.col("weight_kg") < 0)
        assert result["weight_kg"].to_list() == [None, 0.5, None, 2.0]

    def test_row_is_preserved(self):
        df = pl.DataFrame({"id": ["A", "B"], "score": [6, 3]})
        result = null_out_where(df, "score", pl.col("score") > 5)
        assert result.height == 2
        assert result["id"].to_list() == ["A", "B"]
        assert result["score"][0] is None
        assert result["score"][1] == 3
