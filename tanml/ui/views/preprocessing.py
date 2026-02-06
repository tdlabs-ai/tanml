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

    # Save / Export
    st.subheader("3. Finish & Save")
    if st.button("Save Processed Dataset"):
        final_df = st.session_state.get("df_cleaned", df)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save to visible directory
        save_dir = run_dir / "exported_data"
        save_dir.mkdir(parents=True, exist_ok=True)
        p = save_dir / f"processed_data_{ts}.csv"

        # Always overwrite current cleaned path so next steps use it
        final_df.to_csv(p, index=False)
        st.session_state["path_cleaned"] = str(p)
        st.session_state["df_cleaned"] = final_df  # reinforce

        st.success(f"Saved to: `{p}`")
        st.info("You can now go to **Model Validation** and use this dataset.")
