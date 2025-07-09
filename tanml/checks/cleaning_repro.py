import pandas as pd

class CleaningReproCheck:
    def __init__(self, raw_data, cleaned_data):
        self.raw_data = raw_data
        self.cleaned_data = cleaned_data

    def run(self):
        result = {}

        try:
            raw_df = self.raw_data
            cleaned_df = self.cleaned_data

            if not isinstance(raw_df, pd.DataFrame) or not isinstance(cleaned_df, pd.DataFrame):
                raise ValueError("Missing or invalid raw_data / cleaned_data.")

            validator_df = (
                raw_df
                .drop(columns=["constant_col"], errors="ignore")
                .drop_duplicates()
                .reset_index(drop=True)
            )

            result["dev_shape"]       = cleaned_df.shape
            result["validator_shape"] = validator_df.shape
            result["same_shape"]      = cleaned_df.shape == validator_df.shape

            dev_cols = set(cleaned_df.columns)
            val_cols = set(validator_df.columns)
            result["extra_columns_in_dev"]   = sorted(list(dev_cols - val_cols))
            result["missing_columns_in_dev"] = sorted(list(val_cols - dev_cols))

            common = cleaned_df.columns.intersection(validator_df.columns)
            if cleaned_df[common].shape == validator_df[common].shape:
                mismatch_count = (
                    cleaned_df[common].reset_index(drop=True) !=
                    validator_df[common].reset_index(drop=True)
                ).to_numpy().sum()
                result["cell_mismatches"] = int(mismatch_count)
            else:
                result["cell_mismatches"] = "Column/shape mismatch — cannot compare"

        except Exception as e:
            result["error"] = str(e)

        return result
