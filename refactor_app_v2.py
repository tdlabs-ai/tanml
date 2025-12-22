
import os
import shutil

# Read original
with open("tanml/ui/app.py", "r") as f:
    original_lines = f.readlines()

# Cut off before "Main layout" (Line 790 is index 789)
# We found "# ==========================\n# Main layout — single flow" at line 790-792
cutoff_index = -1
for i, line in enumerate(original_lines):
    if "Main layout" in line and "single flow" in line:
        cutoff_index = i - 1 # Keep headers? No, cut before.
        break

if cutoff_index == -1:
    print("Could not find cutoff point!")
    exit(1)

header_lines = original_lines[:cutoff_index]

# New Content
new_code = r'''
# ==========================
# Validation Logic (Refactored)
# ==========================

def render_setup_page(run_dir):
    st.header("Home / Setup")
    st.write("Upload your datasets here for analysis.")
    
    st.session_state.setdefault("data_source", "Single cleaned file (you split)")
    
    ds = st.radio(
        "Data source strategy",
        ("Single cleaned file (you split)", "Already split: Train & Test"),
        key="setup_ds_radio"
    )
    st.session_state["data_source"] = ds

    c1, c2 = st.columns(2)
    
    if ds.startswith("Single"):
        with c1:
            st.subheader("Principal Dataset")
            cleaned_file = st.file_uploader("Cleaned Data (CSV/Parquet)", key="upl_cleaned_pg")
            if cleaned_file:
                path = _save_upload(cleaned_file, run_dir)
                if path:
                    st.session_state["path_cleaned"] = str(path)
                    try:
                        df = load_dataframe(path)
                        st.session_state["df_cleaned"] = df
                        st.session_state["df_preview"] = df  # Main working df
                        st.success(f"Loaded {len(df)} rows.")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
        
        with c2:
            st.subheader("Raw Dataset (Optional)")
            raw_file = st.file_uploader("Raw Data (for reference)", key="upl_raw_pg")
            if raw_file:
                path = _save_upload(raw_file, run_dir)
                if path:
                    st.session_state["path_raw"] = str(path)
                    try:
                        df = load_dataframe(path)
                        st.session_state["df_raw"] = df
                        st.success(f"Loaded {len(df)} rows.")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")

    else:
        # Train / Test split
        with c1:
            st.subheader("Train Set")
            tr_file = st.file_uploader("Train Data", key="upl_train_pg")
            if tr_file:
                path = _save_upload(tr_file, run_dir)
                if path:
                    st.session_state["path_train"] = str(path)
                    try:
                        df = load_dataframe(path)
                        st.session_state["df_train"] = df
                        st.session_state["df_preview"] = df # Default for target picking
                        st.success(f"Loaded {len(df)} rows.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with c2:
            st.subheader("Test Set")
            te_file = st.file_uploader("Test Data", key="upl_test_pg")
            if te_file:
                path = _save_upload(te_file, run_dir)
                if path:
                    st.session_state["path_test"] = str(path)
                    try:
                        df = load_dataframe(path)
                        st.session_state["df_test"] = df
                        st.success(f"Loaded {len(df)} rows.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            st.subheader("Raw Dataset (Global)")
            raw_file = st.file_uploader("Raw Data", key="upl_raw_global_pg")
            if raw_file:
                path = _save_upload(raw_file, run_dir)
                if path:
                    st.session_state["path_raw"] = str(path)
                    st.session_state["df_raw"] = load_dataframe(path)

    # Common Logic: Target Selection (Persisted)
    df_preview = st.session_state.get("df_preview") or st.session_state.get("df_cleaned") or st.session_state.get("df_train")
    
    if df_preview is not None:
        st.divider()
        st.subheader("Variable Configuration")
        cols = list(df_preview.columns)
        
        # Try to persist selection
        tgt_idx = 0
        current_tgt = st.session_state.get("target_col")
        if current_tgt and current_tgt in cols:
            tgt_idx = cols.index(current_tgt)
        else:
            def_tgt = _pick_target(df_preview)
            if def_tgt in cols: tgt_idx = cols.index(def_tgt)
            
        target = st.selectbox("Target Column", cols, index=tgt_idx)
        st.session_state["target_col"] = target
        
        # Features
        current_feats = st.session_state.get("feature_cols", [])
        default_feats = [c for c in cols if c != target]
        # Filter existing selection
        valid_feats = [c for c in current_feats if c in cols and c != target]
        if not valid_feats: valid_feats = default_feats
        
        features = st.multiselect("Features", [c for c in cols if c != target], default=valid_feats)
        st.session_state["feature_cols"] = features
        
        st.info(f"Configuration saved! Target: **{target}**, Features: **{len(features)}**")


def render_profiling_page(df_key, title, run_dir):
    st.header(title)
    df = st.session_state.get(df_key)
    if df is None:
        st.warning(f"No data available for {title}. Please upload in 'Home / Setup'.")
        return

    st.write(f"**Shape**: {df.shape[0]} rows, {df.shape[1]} columns")
    if st.button(f"Run {title}", type="primary"):
        with st.spinner("Running extensive EDA..."):
            from tanml.checks.eda import EDACheck
            import matplotlib.pyplot as plt
            
            # Simple wrapper
            target = st.session_state.get("target_col")
            y_data = df[target] if target and target in df.columns else None
            X_data = df.drop(columns=[target] if target else [], errors='ignore')
            
            subdir = "eda_raw" if "raw" in df_key else "eda_cleaned"
            
            eda = EDACheck(
                cleaned_data=df, 
                X_train=X_data if X_data is not None else df, 
                y_train=y_data if y_data is not None else pd.Series(dtype=float),
                rule_config={"EDACheck": {"max_plots": 10}},
                output_dir=run_dir / "artifacts" / subdir
            )
            res = eda.run()
            
            # Render
            metrics = res.get("health_metrics", {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{metrics.get('n_rows',0):,}")
            m2.metric("Columns", metrics.get('n_cols',0))
            m3.metric("Missing %", f"{metrics.get('pct_missing',0):.1%}")
            m4.metric("Duplicates %", f"{metrics.get('pct_duplicates',0):.1%}")
            
            vis = res.get("visualizations", [])
            plots = res.get("plots", {})
            
            # Target
            if plots.get("target_analysis"):
                st.image(plots["target_analysis"], caption="Target Analysis")
            
            # PCA
            if plots.get("pca_2d"):
                st.image(plots["pca_2d"], caption="PCA 2D Projection")
            
            # Others
            if vis:
                st.subheader("Distributions")
                c1, c2, c3 = st.columns(3)
                for i, v in enumerate(vis[:6]):
                    if os.path.exists(v):
                        (c1 if i%3==0 else c2 if i%3==1 else c3).image(v, caption=os.path.basename(v))

def render_model_validation_page(run_dir):
    st.header("Model Validation")
    
    # Retrieve State
    ds = st.session_state.get("data_source", "Single")
    target = st.session_state.get("target_col")
    features = st.session_state.get("feature_cols", [])
    
    # Check readiness
    ready = False
    if ds.startswith("Single") and st.session_state.get("df_cleaned") is not None:
        ready = True
    elif st.session_state.get("df_train") is not None and st.session_state.get("df_test") is not None:
        ready = True
        
    if not ready or not target:
        st.error("Please configure Data, Target, and Features in 'Home / Setup' first.")
        return

    # Sidebar Config (Local to this page or global? Let's keep local sidebar logic here)
    # We need to replicate the model form
    seed_global = st.session_state.get("seed_global", 42)
    
    # Need y_for_task for task inference
    y_sample = None
    if ds.startswith("Single"):
        df = st.session_state["df_cleaned"]
        if target in df.columns: y_sample = df[target]
    else:
        df = st.session_state["df_train"]
        if target in df.columns: y_sample = df[target]
        
    if y_sample is None:
        st.error("Target column missing from data.")
        return

    library_selected, algo_selected, user_hp, task_selected = render_model_form(y_sample, seed_global)
    
    # Sidebar Thresholds
    with st.sidebar:
        st.divider()
        st.subheader("Validation Settings")
        eval_mode = st.radio(
            "Validation Method",
            ["Quick Run (Single Split)", "Robust Evaluation (Cross-Validation)"],
            index=1,
            help="Robust uses Repeated K-Fold CV."
        )
        
        test_size = 0.3
        cv_folds = 5
        cv_repeats = 1
        
        if eval_mode.startswith("Quick"):
            test_size = st.slider("Test Size", 0.1, 0.5, 0.3)
        else:
            c1, c2 = st.columns(2)
            cv_folds = c1.number_input("Folds (k)", 2, 20, 5)
            cv_repeats = c2.number_input("Repeats", 1, 10, 1)

        # Thresholds
        if task_selected == "classification":
            st.caption("Thresholds")
            auc_min = st.number_input("AUC Min", 0.0, 1.0, 0.60)
            f1_min = st.number_input("F1 Min", 0.0, 1.0, 0.60)
            ks_min = st.number_input("KS Min", 0.0, 1.0, 0.20)
        else:
            auc_min, f1_min, ks_min = 0.0, 0.0, 0.0
            
        # Options
        with st.expander("Advanced Check Options"):
            eda_enabled = st.checkbox("Basic EDA Checks", True)
            shap_enabled = st.checkbox("SHAP (Explainability)", True)
            corr_enabled = st.checkbox("Correlation Checks", True)
            stress_enabled = st.checkbox("Stress Testing", False)
            cluster_enabled = st.checkbox("Cluster Consistency", False)
            vif_enabled = st.checkbox("VIF Checks", False)
            
            shap_bg = 100
            shap_test = 50
            if shap_enabled:
               shap_bg = st.number_input("SHAP Background", 10, 500, 50)
               shap_test = st.number_input("SHAP Test Samples", 10, 500, 50)
               
    # --- The Run Button and Logic ---
    st.divider()
    if st.button("▶️ Start Validation", type="primary"):
        # Reconstruct variables for processing
        # Load dfs from paths to be safe or usage session
        # For simplicity, use session dfs
        
        # 1. Prepare Data
        if ds.startswith("Single"):
            cleaned_df = st.session_state["df_cleaned"]
            X = cleaned_df[features].copy()
            y = cleaned_df[target].copy()
            
            # Split
            if eval_mode.startswith("Quick"):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed_global)
                df_checks = pd.concat([X_train, y_train], axis=1)
                split_strategy = "random"
            else:
                # CV
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed_global)
                df_checks = pd.concat([X, y], axis=1)
                split_strategy = "random"
            
            saved_raw_ = st.session_state.get("path_raw")
            
        else:
            # Train/Test
            train_df = st.session_state["df_train"]
            test_df = st.session_state["df_test"]
            
            # Align
            te_aligned, err = _schema_align_or_error(train_df[features+[target]], test_df[features+[target]])
            if err:
                st.error(err)
                return
            
            X_train = train_df[features]; y_train = train_df[target]
            X_test = te_aligned[features]; y_test = te_aligned[target]
            df_checks = pd.concat([X_train, y_train], axis=1)
            split_strategy = "supplied"
            saved_raw_ = st.session_state.get("path_raw")

        # 2. Build Model
        try:
            model = build_estimator(library_selected, algo_selected, user_hp)
        except Exception as e:
            st.error(f"Model build failed: {e}")
            return

        # 3. CV Loop
        cv_results_summary = None
        if eval_mode.startswith("Robust"):
            with st.spinner(f"Running Robust CV ({cv_folds}x{cv_repeats})..."):
                from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
                from sklearn.base import clone
                from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, mean_squared_error, mean_absolute_error, r2_score
                from tanml.utils.stats import MetricAggregator
                import numpy as np

                full_X = pd.concat([X_train, X_test]) if split_strategy == "random" else X_train
                full_y = pd.concat([y_train, y_test]) if split_strategy == "random" else y_train
                # A bit simplified: for CV we ideally use the full dataset if single split.
                # If train/test supplied, usually we CV on train only or merge?
                # Let's stick to CV on "Train" set for now if supplied, or "Full" if single.
                
                # Actually, if Single: "X" and "y" defined above are full.
                if ds.startswith("Single"):
                     X_cv, y_cv = X, y 
                else: 
                     X_cv, y_cv = X_train, y_train

                is_cls = (task_selected == "classification")
                
                if is_cls:
                    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=seed_global)
                else:
                    cv = RepeatedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=seed_global)
                
                scores = []
                prog = st.progress(0.0)
                tot = cv_folds * cv_repeats
                
                for i, (tr, te) in enumerate(cv.split(X_cv, y_cv)):
                    Xt, Xv = X_cv.iloc[tr], X_cv.iloc[te]
                    yt, yv = y_cv.iloc[tr], y_cv.iloc[te]
                    
                    m = clone(model)
                    m.fit(Xt, yt)
                    yp = m.predict(Xv)
                    
                    res = {}
                    if is_cls:
                        try:
                            pp = m.predict_proba(Xv)[:,1]
                            res["auc"] = roc_auc_score(yv, pp)
                        except: pass
                        res["accuracy"] = accuracy_score(yv, yp)
                        res["f1"] = f1_score(yv, yp, average='binary', zero_division=0)
                    else:
                        res["rmse"] = float(np.sqrt(mean_squared_error(yv, yp)))
                        res["mae"] = mean_absolute_error(yv, yp)
                        res["r2"] = r2_score(yv, yp)
                    scores.append(res)
                    prog.progress((i+1)/tot)
                
                agg = MetricAggregator(scores)
                cv_results_summary = agg.get_formatted_report()
                st.success("CV Complete")

        # 4. Engine Run
        with st.status("Running Validation Engine...", expanded=True):
            model.fit(X_train, y_train)
            
            # Config construction (Simplified for brevity, could use helper)
            rule_cfg = _build_rule_cfg(
                saved_raw=saved_raw_,
                auc_min=auc_min, f1_min=f1_min, ks_min=ks_min,
                eda_enabled=eda_enabled, eda_max_plots=10,
                corr_enabled=corr_enabled, vif_enabled=vif_enabled,
                raw_data_check_enabled=bool(saved_raw_),
                model_meta_enabled=True,
                stress_enabled=stress_enabled, stress_epsilon=0.1, stress_perturb_fraction=0.1,
                cluster_enabled=cluster_enabled, cluster_k=5, cluster_max_k=10,
                shap_enabled=shap_enabled, shap_bg_size=shap_bg, shap_test_size=shap_test,
                artifacts_dir=run_dir / "artifacts",
                split_strategy=split_strategy,
                test_size=test_size,
                seed_global=seed_global,
                component_seeds={},
                in_scope_cols=features + [target]
            )
            
            # Correlation legacy fix
            rule_cfg.setdefault("CorrelationCheck", {})["enabled"] = corr_enabled
            rule_cfg["CorrelationCheck"].update({"method": "pearson", "high_corr_threshold": 0.8})
            
            # SHAP Output path
            rule_cfg.setdefault("SHAPCheck", {})["out_dir"] = str(run_dir / "artifacts")
            
            engine = ValidationEngine(
                model, X_train, X_test, y_train, y_test, rule_cfg, df_checks,
                raw_df=load_dataframe(saved_raw_) if saved_raw_ else None, 
                ctx=st.session_state
            )
            
            results = _try_run_engine(engine)
            
            # Inject CV
            if cv_results_summary:
                results.setdefault("summary", {}).update(cv_results_summary)
                results["summary"]["validation_strategy"] = f"Robust CV ({cv_folds}x{cv_repeats})"
            else:
                results.setdefault("summary", {})["validation_strategy"] = "Single Split"

            # Task Type Stamp
            results["task_type"] = task_selected
            results.setdefault("summary", {})["task_type"] = task_selected
            
            # Save Report
            template = _choose_report_template(task_selected)
            report_path = run_dir / f"tanml_report_{int(time.time())}.docx"
            ReportBuilder(results, str(template), report_path).build()
            
            st.session_state["last_results"] = results
            st.session_state["last_report"] = report_path
            st.write("Done!")
            
    # Render Output if available
    if "last_results" in st.session_state:
        st.divider()
        st.subheader("Results")
        tvr_render_extras("refit") # Reuse existing render logic if we populate "extras"
        # Or Just render directly
        res = st.session_state["last_results"]
        
        # Summary
        st.json(res.get("summary", {}))
        
        # Download
        rp = st.session_state.get("last_report")
        if rp and rp.exists():
            with open(rp, "rb") as f:
                st.download_button("Download Report", f, file_name=rp.name)


def main():
    st.set_page_config(page_title="TanML", layout="wide")
    run_dir = _session_dir()
    
    with st.sidebar:
        st.title("TanML")
        
        # Sidebar global config
        st.session_state.setdefault("seed_global", 42)
        st.number_input("Random Seed", value=st.session_state["seed_global"], key="seed_global")
        
        nav = st.radio("Navigation", [
            "Home / Setup", 
            "Data Profiling (Raw)", 
            "Data Profiling (Cleaned)", 
            "Model Validation",
            "Report Preview & Export",
            "Run History/Artifacts"
        ])
    
    if nav == "Home / Setup":
        render_setup_page(run_dir)
    elif nav == "Data Profiling (Raw)":
        render_profiling_page("df_raw", "Data Profiling (Raw)", run_dir)
    elif nav == "Data Profiling (Cleaned)":
        render_profiling_page("df_cleaned", "Data Profiling (Cleaned)", run_dir)
    elif nav == "Model Validation":
        render_model_validation_page(run_dir)
    elif nav.startswith("Report"):
        st.header("Report Preview")
        if "last_report" in st.session_state:
            st.success(f"Report Ready: {st.session_state['last_report'].name}")
        else:
            st.info("Run validation to generate a report.")
    elif nav.startswith("Run History"):
        st.header("History")
        st.write("Run history not yet persisted.")

if __name__ == "__main__":
    main()
'''

with open("tanml/ui/app.py", "w") as f:
    f.writelines(header_lines)
    f.write(new_code)

print("Refactor complete.")
