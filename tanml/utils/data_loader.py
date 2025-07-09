import os
import pandas as pd

def load_dataframe(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return pd.read_csv(filepath)
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(filepath)
    elif ext == ".parquet":
        return pd.read_parquet(filepath)
    elif ext == ".sas7bdat":
        return pd.read_sas(filepath)
    elif ext in [".txt", ".tsv"]:
        return pd.read_csv(filepath, sep="\t")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
