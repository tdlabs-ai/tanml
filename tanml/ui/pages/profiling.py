# tanml/ui/pages/profiling.py
"""
Data Profiling page logic.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tanml.ui.services.data import _save_upload
from tanml.utils.data_loader import load_dataframe

def _render_rich_profile(df):
    """Shared logic for detailed data profiling (KPIs, Risks, Tabs)."""
    st.divider()
    st.subheader("Dataset Readiness Snapshot")
    
    # 1. KPIs
    mem_usage = df.memory_usage(deep=True).sum() / 1024 # KB
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows", f"{len(df):,}")
    k2.metric("Columns", len(df.columns))
    k3.metric("Memory usage", f"{int(mem_usage)} KB")
    k4.empty() # Placeholder or maybe "Target" if known
    
    # 2. Risks Analysis (Heuristics)
    st.write("")
    c_risks, c_info = st.columns([1, 2])
    
    with c_risks:
        st.markdown("**Top Risks**")
        # Risk Logic
        high_miss = [c for c in df.columns if df[c].isnull().mean() > 0.05]
        high_card = [c for c in df.select_dtypes(include=['object', 'category']).columns if df[c].nunique() > 50]
        constant = [c for c in df.columns if df[c].nunique() <= 1]
        
        if high_miss:
            msg = ", ".join(high_miss[:3]) + ("..." if len(high_miss) > 3 else "")
            st.warning(f"⚠️ {len(high_miss)} Columns >5% Missing\n({msg})")
        if high_card:
            msg = ", ".join(high_card[:3]) + ("..." if len(high_card) > 3 else "")
            st.warning(f"⚠️ {len(high_card)} High-Cardinality Categoricals\n({msg})")
        if constant:
            msg = ", ".join(constant[:3]) + ("..." if len(constant) > 3 else "")
            st.error(f"⚠️ {len(constant)} Near-Constant Features\n({msg})")
        
        if not (high_miss or high_card or constant):
            st.success("✅ No critical data quality risks detected.")

    with c_info:
            st.info("""
            💡 **Definitions**:
            - **High Cardinality**: A categorical feature with many unique values (e.g., 'User_ID' or 'ZipCode'). This can cause issues like overfitting or massive data expansion during One-Hot Encoding.
            - **Constant**: A feature with only 1 unique value. It provides no information to the model.
            """)

    st.divider()
    
    # 3. Tabs
    t0, t1, t2, t3, t4, t5 = st.tabs(["Data Preview", "Missing Values", "Duplicates & Integrity", "Feature Distributions", "Outliers", "Feature Correlation"])
    
    with t0:
        st.markdown("#### Data Preview (Top 10 Rows)")
        st.caption("A quick look at the first 10 rows of your dataset.")
        st.dataframe(df.head(10), use_container_width=True)

    with t1:
        st.markdown("#### Missing Values")
        if df.isnull().sum().sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            m1, m2 = st.columns(2)
            with m1:
                st.caption("Missing By Column")
                st.bar_chart(df.isnull().sum())
            with m2:
                st.caption("Missing Count Table")
                nulls = df.isnull().sum()
                miss_df = pd.DataFrame({"Missing Count": nulls, "Missing %": (nulls/len(df)).map(lambda x:f"{x:.1%}")})
                st.dataframe(miss_df[miss_df["Missing Count"] > 0], use_container_width=True)

    with t2:
        st.markdown("#### Duplicates & Integrity")
        dups = df.duplicated().sum()
        st.metric("Duplicate Rows", dups, delta_color="inverse")
        if dups > 0:
            st.warning(f"Found {dups} duplicate rows. Consider dropping them if they are data errors.")
        else:
            st.success("No duplicate rows found.")
    
    with t3:
        st.markdown("#### Feature Distributions")
        nums = df.select_dtypes(include="number").columns
        if len(nums) > 0:
            num_sel = st.selectbox("Select Numeric Feature", nums, key="prof_dist_feat_sel")
            try:
                arr = df[num_sel].dropna()
                
                # Layout: Histogram | Box Plot
                c_hist, c_box = st.columns(2)
                
                with c_hist:
                    st.caption("Histogram")
                    fig_hist, ax_hist = plt.subplots(figsize=(5, 4))
                    ax_hist.hist(arr, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
                    ax_hist.set_xlabel(num_sel, fontsize=10)
                    ax_hist.set_ylabel("Frequency", fontsize=10)
                    ax_hist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                    ax_hist.tick_params(axis='x', rotation=45)
                    ax_hist.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_hist)
                    plt.close(fig_hist)
                    
                with c_box:
                    st.caption("Box Plot")
                    fig_box, ax_box = plt.subplots(figsize=(5, 4))
                    ax_box.boxplot(arr, vert=True, patch_artist=True, 
                                   boxprops=dict(facecolor='steelblue', alpha=0.7),
                                   medianprops=dict(color='red', linewidth=2))
                    ax_box.set_ylabel(num_sel, fontsize=10)
                    ax_box.set_xticklabels([num_sel])
                    ax_box.grid(axis='y', alpha=0.3)
                    st.pyplot(fig_box)
                    plt.close(fig_box)

                # Stats
                st.write("**Descriptive Statistics**")
                st.write(arr.describe().to_frame().T)
            except:
                st.info("Could not plot distribution")
        else:
            st.info("No numeric features to plot.")
    
    with t4:
        nums = df.select_dtypes(include="number").columns
        if len(nums) > 0:
                # Simple IQR check
                outlier_cols = []
                outlier_masks = {}  # Store masks for each column
                for c in nums:
                    q1 = df[c].quantile(0.25)
                    q3 = df[c].quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[c] < (q1 - 1.5*iqr)) | (df[c] > (q3 + 1.5*iqr))
                    n_out = mask.sum()
                    if n_out > 0:
                        outlier_cols.append((c, n_out))
                        outlier_masks[c] = mask
                
                if outlier_cols:
                    st.write("**Potential Outliers (IQR Method):**")
                    st.dataframe(pd.DataFrame(outlier_cols, columns=["Feature", "Outlier Count"]), use_container_width=True)
                    
                    # Feature selector to view outlier details
                    st.divider()
                    st.write("**View Outlier Details**")
                    outlier_features = [c for c, _ in outlier_cols]
                    selected_feature = st.selectbox("Select feature to view outliers:", outlier_features, key="outlier_feature_select")
                    
                    if selected_feature:
                        mask = outlier_masks[selected_feature]
                        df_outliers = df[mask].copy()
                        
                        # Show stats
                        q1 = df[selected_feature].quantile(0.25)
                        q3 = df[selected_feature].quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5*iqr
                        upper_bound = q3 + 1.5*iqr
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Outliers", len(df_outliers))
                        col2.metric("Lower Bound", f"{lower_bound:.2f}", help="Values below this are outliers")
                        col3.metric("Upper Bound", f"{upper_bound:.2f}", help="Values above this are outliers")
                        col4.metric("% of Data", f"{100*len(df_outliers)/len(df):.1f}%")
                        
                        # Explanation of IQR method
                        st.caption(f"""
                        📌 **How outliers are detected (IQR Method):**
                        - Q1 (25th percentile) = {q1:.2f}, Q3 (75th percentile) = {q3:.2f}
                        - IQR = Q3 - Q1 = {iqr:.2f}
                        - **Lower Bound** = Q1 - 1.5×IQR = {lower_bound:.2f} → Values **below** this are outliers
                        - **Upper Bound** = Q3 + 1.5×IQR = {upper_bound:.2f} → Values **above** this are outliers
                        """)
                        
                        # Show outlier rows
                        with st.expander(f"📊 Outlier Rows ({len(df_outliers)} rows)", expanded=True):
                            st.dataframe(df_outliers.head(100), use_container_width=True)
                            if len(df_outliers) > 100:
                                st.caption(f"Showing first 100 of {len(df_outliers)} outlier rows")
                        
                        # Download button
                        csv_outliers = df_outliers.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label=f"📥 Download {selected_feature} Outliers as CSV",
                            data=csv_outliers,
                            file_name=f"outliers_{selected_feature}.csv",
                            mime="text/csv"
                        )
                else:
                    st.success("No significant outliers detected via IQR method.")
    
    with t5:
        n_df = df.select_dtypes(include="number")
        if n_df.shape[1] > 1:
            corr = n_df.corr()
            st.dataframe(corr.style.background_gradient(cmap="coolwarm", axis=None), use_container_width=True)
        else:
            st.info("Not enough numeric features for correlation.")


def render_data_profiling_hub(run_dir):
    st.header("Data Profiling")
    st.caption("Upload any dataset to generate a comprehensive data quality profile.")

    # 1. State Management for Profiling Data
    if "df_profiling" not in st.session_state:
        st.session_state["df_profiling"] = None

    # 2. Upload Logic
    upl = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "xls", "parquet", "dta", "sav", "sas7bdat"], key="upl_prof_unified")
    if upl:
        path = _save_upload(upl, run_dir)
        if path:
            # Avoid reloading if same file
            current_path = st.session_state.get("path_profiling")
            if current_path != str(path):
                try:
                    df = load_dataframe(path)
                    st.session_state["df_profiling"] = df
                    st.session_state["path_profiling"] = str(path)
                    st.session_state["run_prof_gen"] = False # Reset report on new data
                    st.success(f"✅ Loaded ({df.shape[0]} rows, {df.shape[1]} columns)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading file: {e}")

    # 3. Display Current Status
    df = st.session_state.get("df_profiling")
    
    if df is not None:
        st.info(f"**Current Dataset**: {len(df)} rows, {len(df.columns)} columns")
        
        # 4. Action: Generate Profile
        if st.button("🚀 Generate Profile", type="primary", key="btn_prof_gen"):
            st.session_state["run_prof_gen"] = True
            
        if st.session_state.get("run_prof_gen"):
            _render_rich_profile(df)
    else:
        st.info("Please upload a dataset to begin.")
