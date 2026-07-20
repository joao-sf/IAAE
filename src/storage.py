from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def build_duckdb(gold_dir: Path, database_path: Path) -> None:
    import duckdb

    fact_path = (gold_dir / "fact_compras.parquet").as_posix()
    material_path = (gold_dir / "dim_material.parquet").as_posix()
    quality_path = (gold_dir / "relatorio_qualidade.psv").as_posix()

    connection = duckdb.connect(database_path.as_posix())
    try:
        connection.execute(
            "CREATE OR REPLACE TABLE fact_compras AS SELECT * FROM read_parquet(?)",
            [fact_path],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE dim_material AS SELECT * FROM read_parquet(?)",
            [material_path],
        )
        connection.execute(
            "CREATE OR REPLACE TABLE relatorio_qualidade AS "
            "SELECT * FROM read_csv(?, delim='|', header=true)",
            [quality_path],
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_compras_validas AS
            SELECT *
            FROM fact_compras
            WHERE COALESCE(dq_possui_erro, FALSE) = FALSE
              AND familia_material IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_benchmark_preco AS
            SELECT *
            FROM vw_compras_validas
            WHERE COALESCE(is_price_outlier, FALSE) = FALSE
              AND unidade_comparavel IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_outliers_preco AS
            SELECT *
            FROM fact_compras
            WHERE COALESCE(is_price_outlier, FALSE) = TRUE
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE VIEW vw_indicadores_mensais AS
            SELECT
                date_trunc('month', data_publicacao) AS mes,
                familia_material,
                codigo_catmat,
                unidade_comparavel,
                SUM(quantidade) AS quantidade,
                SUM(valor_total) AS valor_total,
                SUM(valor_total) / NULLIF(SUM(quantidade), 0)
                    AS preco_medio_ponderado,
                MEDIAN(valor_unitario) AS preco_mediano,
                COUNT(DISTINCT cnpj_fornecedor) AS fornecedores
            FROM vw_benchmark_preco
            GROUP BY ALL
            """
        )
    finally:
        connection.close()
