# tanml/ui/pages/preprocessing.py
"""
Data Preprocessing page logic.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import LabelEncoder
from pathlib import Path


def render_preprocessing_hub(run_dir):
    st.markdown(
        """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    ">
        <h2 style="color: white; margin: 0;">🔧 Data Preprocessing</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.caption("Clean and prepare your dataset for modeling.")

    # 1. Load Data
    df = st.session_state.get("df_cleaned")
    if df is None:
        df = st.session_state.get("df_raw")
    if df is None:
        df = st.session_state.get("df_profiling")
    if df is None:
        df = st.session_state.get("df_train")

    if df is None:
        st.info("Please upload a dataset in 'Data Profiling' or Home first.")
        return

    # Helper: Reset
    c_reset, c_msg = st.columns([1, 3])
    with c_reset:
        if st.button("🔄 Reset", help="Undo all imputation and encoding changes"):
            # Find the original data source (raw > profiling > train)
            original_df = (
                st.session_state.get("df_raw")
                or st.session_state.get("df_profiling")
                or st.session_state.get("df_train")
            )
            if original_df is not None:
                st.session_state["df_cleaned"] = original_df.copy()
                # Clear history completely, then add reset message
                st.session_state["fe_history"] = [
                    "✅ Dataset reset to original state. All changes undone."
                ]
                st.rerun()
            else:
                st.warning("No original data found to reset to.")

    # Persistent Status History
    if st.session_state.get("fe_history"):
        with c_msg:
            for msg in st.session_state["fe_history"]:
                st.success(msg)

    # Work on a copy in session state to allow undo/reset (simplified here: just work on df)
    # Ideally we have a "df_processing"

    st.subheader("1. Imputation (Missing Values)")

    # Identify missing
    miss_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
    if not miss_cols:
        st.success("✅ No missing values found!")
    else:
        st.write(f"Found {len(miss_cols)} columns with missing values.")

        # Partition missing columns
        all_nums = df.select_dtypes(include="number").columns
        all_cats = df.select_dtypes(exclude="number").columns

        nums_miss_all = [c for c in all_nums if c in miss_cols]
        cats_miss_all = [c for c in all_cats if c in miss_cols]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Numeric Imputation**")
            # Multiselect for columns
            if nums_miss_all:
                target_nums = st.multiselect(
                    "Select Numeric Columns", nums_miss_all, default=nums_miss_all, key="ms_imp_num"
                )
                imp_num_strat = st.selectbox(
                    "Numeric Strategy",
                    ["Mean", "Median", "Zero", "KNN (Advanced)"],
                    key="imp_num_strat",
                )
            else:
                st.info("No numeric columns with missing values.")
                target_nums = []
                imp_num_strat = "Mean"

        with c2:
            st.markdown("**Categorical Imputation**")
            # Multiselect for columns
            if cats_miss_all:
                target_cats = st.multiselect(
                    "Select Categorical Columns",
                    cats_miss_all,
                    default=cats_miss_all,
                    key="ms_imp_cat",
                )
                imp_cat_strat = st.selectbox(
                    "Categorical Strategy",
                    ["Mode (Most Frequent)", "Missing_Label"],
                    key="imp_cat_strat",
                )
            else:
                st.info("No categorical columns with missing values.")
                target_cats = []
                imp_cat_strat = "Mode"

        drop_rows = st.checkbox("Or just Drop Rows with missing values?", key="imp_drop_rows")

        if st.button("Apply Imputation", type="primary"):
            df_new = df.copy()

            if drop_rows:
                # Naive drop of ANY missing, or just selected?
                # "Drop Rows" usually implies dropping rows if they have ANY missing value in the selected subset or globally.
                # Let's drop rows where ANY of the *selected* columns are missing.
                cols_to_check = target_nums + target_cats
                if cols_to_check:
                    df_new.dropna(subset=cols_to_check, inplace=True)
                    # Add to history
                    hist = st.session_state.get("fe_history", [])
                    hist.append(
                        f"Dropped rows based on selected columns. Shape: {df.shape} -> {df_new.shape}"
                    )
                    st.session_state["fe_history"] = hist
                else:
                    st.warning("No columns selected for checking NaNs.")
                    return  # Stop
            else:
                # Numeric
                if target_nums:
                    if imp_num_strat == "Mean":
                        si = SimpleImputer(strategy="mean")
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif imp_num_strat == "Median":
                        si = SimpleImputer(strategy="median")
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif imp_num_strat == "Zero":
                        si = SimpleImputer(strategy="constant", fill_value=0)
                        df_new[target_nums] = si.fit_transform(df_new[target_nums])
                    elif "KNN" in imp_num_strat:
                        knn = KNNImputer(n_neighbors=5)
                        df_new[target_nums] = knn.fit_transform(df_new[target_nums])

                # Categorical
                if target_cats:
                    if "Mode" in imp_cat_strat:
                        si = SimpleImputer(strategy="most_frequent")
                        res = si.fit_transform(df_new[target_cats])
                        df_new[target_cats] = res
                    else:
                        si = SimpleImputer(strategy="constant", fill_value="Missing")
                        res = si.fit_transform(df_new[target_cats])
                        df_new[target_cats] = res

            # Update State
            st.session_state["df_cleaned"] = df_new

            # Construct detailed message
            msg_parts = []
            if drop_rows and cols_to_check:
                msg_parts.append("Dropped rows")
            if target_nums:
                msg_parts.append(f"Imputed Numeric ({imp_num_strat}): {', '.join(target_nums)}")
            if target_cats:
                msg_parts.append(f"Imputed Categorical ({imp_cat_strat}): {', '.join(target_cats)}")

            full_msg = f"✅ Applied Changes: {'; '.join(msg_parts)}. Continue to impute other features or move to Encoding."

            # Append to history
            hist = st.session_state.get("fe_history", [])
            hist.append(full_msg)
            st.session_state["fe_history"] = hist

            st.rerun()

    st.divider()

    st.subheader("2. Encoding (Categorical)")
    df = st.session_state.get("df_cleaned", df)  # Refresh

    cat_cols_all = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols_all:
        st.success("✅ No categorical columns found (all numeric)!")
    else:
        st.write("Convert text to numbers:")
        target_enc = st.multiselect(
            "Select Categorical Columns to Encode",
            cat_cols_all,
            default=cat_cols_all,
            key="ms_enc_cols",
        )

        enc_method = st.radio("Encoding Method", ["One-Hot Encoding", "Label Encoding"])

        if st.button("Apply Encoding", type="primary"):
            if not target_enc:
                st.warning("Please select at least one column to encode.")
            else:
                df_enc = df.copy()

                if enc_method.startswith("One-Hot"):
                    df_enc = pd.get_dummies(df_enc, columns=target_enc, drop_first=True)
                else:
                    le = LabelEncoder()
                    for c in target_enc:
                        # Convert to string to be safe
                        df_enc[c] = le.fit_transform(df_enc[c].astype(str))

                st.session_state["df_cleaned"] = df_enc

                # Detailed message
                full_msg = f"✅ Applied Changes: Encoded ({enc_method}): {', '.join(target_enc)}. Encode other features or Save the dataset."

                # Append to history
                hist = st.session_state.get("fe_history", [])
                hist.append(full_msg)
                st.session_state["fe_history"] = hist

                st.rerun()

    st.divider()

    st.subheader("3. Data Splitting")
    df_for_split = st.session_state.get("df_cleaned", df)
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 50, 20, 5) / 100
        random_state = st.number_input("Random Seed", value=42)
    
    with col2:
        stratify_col = st.selectbox(
            "Stratify by (Optional)", 
            ["None"] + list(df_for_split.columns),
            help="Ensures the Train/Test sets have the same proportion of classes as the original data for the selected column (useful for imbalanced targets)."
        )
        shuffle = st.checkbox(
            "Shuffle data", 
            value=True,
            help="Randomizes row order before splitting. Recommended for most machine learning tasks."
        )

    stratify = None
    if stratify_col != "None":
        stratify = df_for_split[stratify_col]

    if st.button("🚀 Execute Split", type="primary", use_container_width=True):
        from sklearn.model_selection import train_test_split
        try:
            df_train, df_test = train_test_split(
                df_for_split,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle,
                stratify=stratify
            )
            
            st.session_state["df_split_train"] = df_train
            st.session_state["df_split_test"] = df_test
            st.session_state["df_train"] = df_train
            st.session_state["df_test"] = df_test
            
            # Append detailed message to history
            full_msg = f"✅ Applied Changes: Split data into Train ({df_train.shape[0]} rows) and Test ({df_test.shape[0]} rows)."
            hist = st.session_state.get("fe_history", [])
            hist.append(full_msg)
            st.session_state["fe_history"] = hist
            
            st.success(f"Split Complete! Train: {df_train.shape[0]} rows | Test: {df_test.shape[0]} rows")
        except Exception as e:
            st.error(f"Split Failed: {e}")

    if "df_split_train" in st.session_state:
        # Download buttons
        st.markdown("**Download Splits**")
        dcol1, dcol2 = st.columns(2)
        train_csv = st.session_state["df_split_train"].to_csv(index=False).encode('utf-8')
        dcol1.download_button(
            "📥 Download Training Set",
            data=train_csv,
            file_name="train_split.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        test_csv = st.session_state["df_split_test"].to_csv(index=False).encode('utf-8')
        dcol2.download_button(
            "📥 Download Testing Set",
            data=test_csv,
            file_name="test_split.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        with st.expander("Preview Splits"):
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.caption("Training Data")
                st.dataframe(st.session_state["df_split_train"].head(5))
            with pcol2:
                st.caption("Testing Data")
                st.dataframe(st.session_state["df_split_test"].head(5))

    st.divider()

    # Save / Export
    st.subheader("4. Finish & Save")
    
    # Determine original extension
    orig_ext = ".csv"
    if "path_raw" in st.session_state and st.session_state.get("path_raw"):
        orig_ext = Path(st.session_state["path_raw"]).suffix.lower()
    elif "path_profiling" in st.session_state and st.session_state.get("path_profiling"):
        orig_ext = Path(st.session_state["path_profiling"]).suffix.lower()

    # Friendly names for formats
    format_options = [
        f"Original Format ({orig_ext})", 
        "CSV (.csv)", 
        "Parquet (.parquet)", 
        "Excel (.xlsx)", 
        "JSON (.json)", 
        "Pickle (.pkl)", 
        "Feather (.feather)", 
        "Stata (.dta)", 
        "SPSS (.sav)", 
        "SAS XPORT (.xpt)"
    ]
    save_format = st.selectbox("Export Format", format_options, index=0)
    
    if st.button("Save Processed Dataset"):
        final_df = st.session_state.get("df_cleaned", df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to visible directory
        save_dir = run_dir / "exported_data"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine actual extension
        target_ext = orig_ext
        if not save_format.startswith("Original Format"):
            # Extract extension from parens, e.g. "SAS XPORT (.xpt)" -> ".xpt"
            target_ext = save_format.split("(")[1].strip(")")
        
        p = save_dir / f"processed_data_{ts}{target_ext}"

        def save_df(dataframe: pd.DataFrame, target_path: Path, ext: str):
            try:
                if ext in [".csv", ".txt", ".data", ".test"]:
                    dataframe.to_csv(target_path, index=False)
                elif ext == ".tsv":
                    dataframe.to_csv(target_path, sep="\t", index=False)
                elif ext == ".parquet":
                    dataframe.to_parquet(target_path, index=False)
                elif ext in [".xlsx", ".xls"]:
                    dataframe.to_excel(target_path, index=False)
                elif ext == ".json":
                    dataframe.to_json(target_path, orient="records")
                elif ext in [".pkl", ".pickle"]:
                    dataframe.to_pickle(target_path)
                elif ext in [".feather", ".ft"]:
                    dataframe.to_feather(target_path)
                elif ext == ".dta":
                    dataframe.to_stata(target_path, write_index=False)
                elif ext == ".sav":
                    import pyreadstat
                    pyreadstat.write_sav(dataframe, str(target_path))
                elif ext in [".xpt", ".sas7bdat"]:
                    # SAS7BDAT writing is usually not supported in python natively, fallback to XPORT for SAS
                    target_path = target_path.with_suffix(".xpt")
                    import pyreadstat
                    pyreadstat.write_xport(dataframe, str(target_path))
                else:
                    # Fallback to CSV for unknown types
                    target_path = target_path.with_suffix(".csv")
                    dataframe.to_csv(target_path, index=False)
                    st.warning(f"Native write for '{ext}' is unsupported. Saved as CSV instead.")
                return target_path
            except ModuleNotFoundError as e:
                st.error(f"Missing dependency for {ext}. Try installing pyreadstat/pyarrow. Falling back to CSV.")
                target_path = target_path.with_suffix(".csv")
                dataframe.to_csv(target_path, index=False)
                return target_path
            except Exception as e:
                st.error(f"Failed to save {ext}: {e}. Falling back to CSV.")
                target_path = target_path.with_suffix(".csv")
                dataframe.to_csv(target_path, index=False)
                return target_path

        # Save main processed data
        actual_p = save_df(final_df, p, target_ext)
        st.session_state["path_cleaned"] = str(actual_p)
        st.session_state["df_cleaned"] = final_df  # reinforce
        
        st.success(f"Full Dataset saved to: `{actual_p}`")
        
        # Save splits if they exist
        if "df_split_train" in st.session_state and "df_split_test" in st.session_state:
            p_train = save_dir / f"train_split_{ts}{target_ext}"
            p_test = save_dir / f"test_split_{ts}{target_ext}"
            
            actual_p_train = save_df(st.session_state["df_split_train"], p_train, target_ext)
            actual_p_test = save_df(st.session_state["df_split_test"], p_test, target_ext)
            
            st.success(f"Training Set saved to: `{actual_p_train}`")
            st.success(f"Testing Set saved to: `{actual_p_test}`")

        st.info("You can now go to **Feature Power Ranking** or **Model Development** and use this dataset.")
