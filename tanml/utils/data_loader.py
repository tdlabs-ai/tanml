from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataframe(
    filepath: str | Path,
    sheet_name: str | int | None = None,
    sep: str | None = None,
    encoding: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Universal table loader with safe fallbacks and memory optimization.
    """
    df = _load_raw(filepath, sheet_name, sep, encoding, **kwargs)

    # --- Memory Optimization ---
    # Convert object columns to category if cardinality is low (<50%)
    # This significantly reduces RAM usage for large datasets with repeated strings
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")
        except Exception:
            pass  # Keep as object if conversion fails

    return df


def _load_raw(
    filepath: str | Path,
    sheet_name: str | int | None = None,
    sep: str | None = None,
    encoding: str | None = None,
    header: int | str | None = "infer",
    **_: Any,
) -> pd.DataFrame:
    """
    Internal raw loader (original logic).

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
    # .data and .test are common UCI ML Repository extensions
    if ext in {".csv", ".txt", ".data", ".test"}:
        # Common missing value indicators (UCI uses " ?" and "?")
        # Note: Removed single space " " as it incorrectly treats values with leading spaces as missing
        na_vals = ["?", " ?", "NA", "N/A", "n/a", "na", "NaN", "nan", "", "null", "NULL", "None"]

        # If sep is unspecified:
        # - CSV/.data: assume comma (keeps fast C engine, no warning)
        # - TXT/.test: keep sep=None (Python engine infers separator)
        eff_sep = sep if sep is not None else ("," if ext in {".csv", ".data"} else None)
        engine = "c" if eff_sep is not None else "python"
        # Handle header parameter: 'infer' means use default (0), None means no header
        hdr = None if header == "none" else (0 if header == "infer" else header)

        try:
            # skipinitialspace=True strips leading spaces from values (common in UCI data)
            return pd.read_csv(
                p,
                sep=eff_sep,
                encoding=encoding,
                engine=engine,
                na_values=na_vals,
                header=hdr,
                skipinitialspace=True,
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                p,
                sep=eff_sep,
                encoding=encoding or "latin-1",
                engine=engine,
                na_values=na_vals,
                header=hdr,
                skipinitialspace=True,
            )

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

    # --- Unknown extension: try to parse as delimited text (like UCI files) ---
    # This handles .names, .info, and other non-standard extensions
    try:
        # Try comma first (most common)
        return pd.read_csv(p, sep=",", encoding="utf-8", engine="c")
    except Exception:
        try:
            # Try with Python engine (auto-infers separator)
            return pd.read_csv(p, sep=None, encoding="utf-8", engine="python")
        except Exception:
            try:
                # Try latin-1 encoding as fallback
                return pd.read_csv(p, sep=None, encoding="latin-1", engine="python")
            except Exception as e:
                raise ValueError(
                    f"Could not parse file: {ext} (path={p}). "
                    f"Tried as delimited text but failed. Error: {e}"
                )
