import os
import pandas as pd
from sqlalchemy import create_engine, text


def get_engine(connection_string: str = None):
    """Create SQLAlchemy engine from DATABASE_URL env var or provided string."""
    url = connection_string or os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL environment variable is not set")
    return create_engine(url)


def save_to_parquet(df: pd.DataFrame, path: str):
    """Save DataFrame to Parquet file."""
    df.to_parquet(path, index=False)
    print(f"Saved {len(df)} rows to {path}")


def save_to_postgres(df: pd.DataFrame, table_name: str, engine, if_exists: str = "replace"):
    """Write DataFrame to a PostgreSQL table (full replace — use for initial loads)."""
    df.to_sql(table_name, engine, if_exists=if_exists, index=False)
    print(f"Saved {len(df)} rows to postgres table '{table_name}'")


def _get_table_columns(table_name: str, engine) -> set:
    """Return the set of column names that exist in the target PostgreSQL table."""
    with engine.connect() as conn:
        result = conn.execute(text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = :t"
        ), {"t": table_name})
        return {row[0] for row in result}


def upsert_to_postgres(df: pd.DataFrame, table_name: str, engine, conflict_cols: list):
    """
    Insert DataFrame rows into a PostgreSQL table using ON CONFLICT DO NOTHING.
    Only columns that exist in the target table schema are inserted — extra
    columns coming from raw API responses are silently dropped.
    Re-running the ETL pipeline never produces duplicate rows.
    """
    if df.empty:
        print(f"Skipping upsert to '{table_name}': empty DataFrame")
        return

    # Drop columns not in the target schema to avoid UndefinedColumn errors
    table_cols  = _get_table_columns(table_name, engine)
    valid_cols  = [c for c in df.columns if c in table_cols]
    dropped     = len(df.columns) - len(valid_cols)
    if dropped:
        print(f"  Dropping {dropped} columns not in '{table_name}' schema")
    df = df[valid_cols]

    temp_table = f"_tmp_{table_name}"
    df.to_sql(temp_table, engine, if_exists="replace", index=False)

    cols     = ", ".join(f'"{c}"' for c in valid_cols)
    conflict = ", ".join(f'"{c}"' for c in conflict_cols)

    with engine.connect() as conn:
        conn.execute(text(f"""
            INSERT INTO "{table_name}" ({cols})
            SELECT {cols} FROM "{temp_table}"
            ON CONFLICT ({conflict}) DO NOTHING;
        """))
        conn.execute(text(f'DROP TABLE IF EXISTS "{temp_table}";'))
        conn.commit()

    print(f"Upserted into '{table_name}' (conflict on {conflict_cols})")
