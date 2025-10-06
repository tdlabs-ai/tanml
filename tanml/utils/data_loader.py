from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import pandas as pd


def load_dataframe(filepath: str | Path,
                   sheet_name: Optional[str | int] = None,
                   sep: Optional[str] = None,
                   encoding: Optional[str] = None,
                   **_: Any) -> pd.DataFrame:
    """
    Universal table loader with safe fallbacks.

    Supports:
      .csv, .txt, .tsv
      .xlsx, .xls          (needs openpyxl)
      .parquet             (pyarrow or fastparquet)
      .feather/.ft         (pyarrow)
      .json
      .pkl/.pickle
      .sas7bdat/.xpt       (pyreadstat preferred; falls back to pandas.read_sas)
      .sav                 (SPSS; pyreadstat)
      .dta                 (Stata)
    """
    p = Path(filepath)
    ext = p.suffix.lower()

    # --- Delimited text ---
    if ext in {".csv", ".txt"}:
        # If sep is unspecified:
        # - CSV: assume comma (keeps fast C engine, no warning)
        # - TXT: keep sep=None (we'll use Python engine to infer)
        eff_sep = sep if sep is not None else ("," if ext == ".csv" else None)
        engine = "c" if eff_sep is not None else "python"
        try:
            return pd.read_csv(p, sep=eff_sep, encoding=encoding, engine=engine)
        except UnicodeDecodeError:
            return pd.read_csv(p, sep=eff_sep, encoding=encoding or "latin-1", engine=engine)

    if ext == ".tsv":
        try:
            return pd.read_csv(p, sep="\t", encoding=encoding, engine="c")
        except UnicodeDecodeError:
            return pd.read_csv(p, sep="\t", encoding=encoding or "latin-1", engine="c")


    # --- Excel ---
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p, sheet_name=0 if sheet_name is None else sheet_name)
        except ImportError as e:
            raise ModuleNotFoundError(
                "openpyxl>=3.1 is required for Excel files. Install with: pip install openpyxl"
            ) from e

    # --- Columnar ---
    if ext == ".parquet":
        try:
            return pd.read_parquet(p, engine="pyarrow")
        except Exception:
            return pd.read_parquet(p, engine="fastparquet")

    if ext in {".feather", ".ft"}:
        try:
            import pyarrow  # noqa: F401
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "pyarrow is required for Feather files. Install with: pip install pyarrow"
            ) from e
        return pd.read_feather(p)

    # --- Other common formats ---
    if ext in {".pkl", ".pickle"}:
        return pd.read_pickle(p)

    if ext == ".json":
        return pd.read_json(p, convert_dates=True)

    # --- SAS / SPSS / Stata ---
    if ext in {".sas7bdat", ".xpt"}:
        try:
            import pyreadstat  # type: ignore
            if ext == ".sas7bdat":
                df, _ = pyreadstat.read_sas7bdat(str(p))
            else:
                df, _ = pyreadstat.read_xport(str(p))
            return df
        except ModuleNotFoundError:
            # best-effort fallback
            return pd.read_sas(p)

    if ext == ".sav":  # SPSS
        try:
            import pyreadstat  # type: ignore
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "pyreadstat is required for SPSS .sav files. Install with: pip install pyreadstat"
            ) from e
        df, _ = pyreadstat.read_sav(str(p))
        return df

    if ext == ".dta":  # Stata
        return pd.read_stata(p)

    raise ValueError(f"Unsupported file format: {ext} (path={p})")
