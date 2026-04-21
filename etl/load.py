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


def upsert_to_postgres(df: pd.DataFrame, table_name: str, engine, conflict_cols: list):
    """
    Insert DataFrame rows into a PostgreSQL table using ON CONFLICT DO NOTHING.
    Re-running the ETL pipeline never produces duplicate rows.
    """
    if df.empty:
        print(f"Skipping upsert to '{table_name}': empty DataFrame")
        return

    temp_table = f"_tmp_{table_name}"
    df.to_sql(temp_table, engine, if_exists="replace", index=False)

    cols = ", ".join(f'"{c}"' for c in df.columns)
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
