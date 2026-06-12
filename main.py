"""Orquestador del pipeline ETL medallion.

Ejecuta las capas en el orden correcto:
  1. Bronze     : ingesta CSV -> Parquet (schema todo-string).
  2. Quality    : perfilado de Bronze (nulos, duplicados, row_count).
  3. Silver     : limpieza, normalización y cuarentena por entidad.
  4. Referential: integridad referencial post-Silver (cuarentena huérfanos).
  5. Gold       : tablas analíticas sobre Silver limpio.

Uso:
    python main.py
"""
from src import bronze, gold, quality, referential, silver


def main() -> None:
    print("=" * 60)
    print("PIPELINE ETL - ARQUITECTURA MEDALLION")
    print("=" * 60)

    print("\n[1/5] BRONZE — ingesta cruda")
    bronze.run()

    print("\n[2/5] QUALITY — perfilado Bronze")
    quality.run()

    print("\n[3/5] SILVER — limpieza y normalización")
    silver.run()

    print("\n[4/5] REFERENTIAL — integridad referencial")
    referential.run()

    print("\n[5/5] GOLD — tablas analíticas")
    gold.run()

    print("\n" + "=" * 60)
    print("Pipeline completado.")
    print("  data/bronze/   -> Parquet crudos")
    print("  data/silver/   -> Parquet limpios")
    print("  data/gold/     -> Tablas analíticas")
    print("  data/quality/  -> Checks de calidad + cuarentenas")
    print("=" * 60)


if __name__ == "__main__":
    main()
