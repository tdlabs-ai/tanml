# tanml/ui/pages/evaluation/tabs/benchmark.py
"""
Benchmark Tab - Compares model against baselines.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tanml.ui.pages.evaluation.tabs import register_tab


@register_tab(name="Benchmarking", order=50, key="tab_bench")
def render(context):
    """Render the benchmarking tab."""
    
    st.markdown("### Benchmarking: Compare Your Model vs Baseline Models")
    st.caption("Select one or more baseline models to compare against your trained model.")
    
    # Model selection based on task type
    if context.task_type == "classification":
        available_models = {
            "Logistic Regression (statsmodels)": "sm_logit",
            "Logistic Regression (sklearn)": "sk_logistic",
            "Random Forest": "sk_rf",
            "Decision Tree": "sk_dt",
            "Naive Bayes": "sk_nb",
            "Dummy Classifier (Most Frequent)": "sk_dummy"
        }
    else:  # regression
        available_models = {
            "OLS Regression (statsmodels)": "sm_ols",
            "Linear Regression (sklearn)": "sk_linear",
            "Ridge Regression": "sk_ridge",
            "Random Forest": "sk_rf",
            "Decision Tree": "sk_dt",
            "Dummy Regressor (Mean)": "sk_dummy"
        }
    
    selected_models = st.multiselect(
        "Select Baseline Models to Compare",
        options=list(available_models.keys()),
        default=[list(available_models.keys())[0]],
        help="Choose one or more models to benchmark against"
    )
    
    if st.button("🔬 Run Benchmark Comparison", type="secondary", key="btn_benchmark"):
        if not selected_models:
            st.warning("Please select at least one baseline model.")
        else:
            with st.spinner("Training baseline models..."):
                try:
                    import statsmodels.api as sm
                    from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
                    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
                    from sklearn.naive_bayes import GaussianNB
                    from sklearn.dummy import DummyClassifier, DummyRegressor
                    from sklearn.metrics import (
                        roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
                        mean_squared_error, mean_absolute_error, r2_score
                    )
                    
                    # Prepare data
                    X_tr_num = context.X_train.select_dtypes(include=np.number).fillna(0)
                    X_te_num = context.X_test.select_dtypes(include=np.number).fillna(0)
                    
                    benchmark_results = {"your_model": {}, "baselines": {}}
                    
                    # Your model's metrics (try to get from context, else use simplified dict)
                    metrics_test = context.results.get("metrics_test", {})
                    
                    if context.task_type == "classification":
                        benchmark_results["your_model"] = {
                            "roc_auc": metrics_test.get("AUC", 0),
                            "f1": metrics_test.get("F1", 0),
                            "accuracy": metrics_test.get("Accuracy", 0),
                            "precision": metrics_test.get("Precision", 0),
                            "recall": metrics_test.get("Recall", 0)
                        }
                        higher_better = ["roc_auc", "f1", "accuracy", "precision", "recall"]
                    else:
                        benchmark_results["your_model"] = {
                            "rmse": metrics_test.get("RMSE", 0),
                            "mae": metrics_test.get("MAE", 0),
                            "r2": metrics_test.get("R2", 0)
                        }
                        higher_better = ["r2"]
                    
                    # Train each selected baseline
                    for model_name in selected_models:
                        model_key = available_models[model_name]
                        try:
                            if context.task_type == "classification":
                                # Initialize model
                                if model_key == "sm_logit":
                                    X_sm = sm.add_constant(X_tr_num)
                                    X_sm_te = sm.add_constant(X_te_num)
                                    # Use a simple try/except for singular matrix issues
                                    try:
                                        mdl = sm.Logit(context.y_train, X_sm).fit(disp=0, maxiter=100)
                                        pred_prob = mdl.predict(X_sm_te)
                                    except:
                                        # Fallback if statsmodels fails
                                        mdl = LogisticRegression()
                                        mdl.fit(X_tr_num, context.y_train)
                                        pred_prob = mdl.predict_proba(X_te_num)[:, 1]

                                    pred = (pred_prob >= 0.5).astype(int)
                                elif model_key == "sk_logistic":
                                    mdl = LogisticRegression(max_iter=200, random_state=42)
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                    pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                elif model_key == "sk_rf":
                                    mdl = RandomForestClassifier(n_estimators=50, random_state=42)
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                    pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                elif model_key == "sk_dt":
                                    mdl = DecisionTreeClassifier(random_state=42)
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                    pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                elif model_key == "sk_nb":
                                    mdl = GaussianNB()
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                    pred_prob = mdl.predict_proba(X_te_num)[:, 1]
                                elif model_key == "sk_dummy":
                                    mdl = DummyClassifier(strategy="most_frequent")
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                    pred_prob = np.full(len(context.y_test), context.y_train.mean())
                                
                                # Calculate metrics
                                benchmark_results["baselines"][model_name] = {
                                    "roc_auc": roc_auc_score(context.y_test, pred_prob) if len(np.unique(pred_prob)) > 1 else 0.5,
                                    "f1": f1_score(context.y_test, pred),
                                    "accuracy": accuracy_score(context.y_test, pred),
                                    "precision": precision_score(context.y_test, pred, zero_division=0),
                                    "recall": recall_score(context.y_test, pred, zero_division=0)
                                }
                            else:  # Regression
                                if model_key == "sm_ols":
                                    X_sm = sm.add_constant(X_tr_num)
                                    X_sm_te = sm.add_constant(X_te_num)
                                    mdl = sm.OLS(context.y_train, X_sm).fit()
                                    pred = mdl.predict(X_sm_te)
                                elif model_key == "sk_linear":
                                    mdl = LinearRegression()
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                elif model_key == "sk_ridge":
                                    mdl = Ridge()
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                elif model_key == "sk_rf":
                                    mdl = RandomForestRegressor(n_estimators=50, random_state=42)
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                elif model_key == "sk_dt":
                                    mdl = DecisionTreeRegressor(random_state=42)
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                elif model_key == "sk_dummy":
                                    mdl = DummyRegressor(strategy="mean")
                                    mdl.fit(X_tr_num, context.y_train)
                                    pred = mdl.predict(X_te_num)
                                
                                benchmark_results["baselines"][model_name] = {
                                    "rmse": np.sqrt(mean_squared_error(context.y_test, pred)),
                                    "mae": mean_absolute_error(context.y_test, pred),
                                    "r2": r2_score(context.y_test, pred)
                                }
                        except Exception as model_err:
                            st.warning(f"Failed to train {model_name}: {model_err}")
                    
                    benchmark_results["higher_better"] = higher_better
                    benchmark_results["task_type"] = context.task_type
                    st.session_state["benchmark_results"] = benchmark_results
                    st.success(f"✅ Benchmark complete! Compared against {len(benchmark_results['baselines'])} baseline models.")
                    
                except Exception as e:
                    st.error(f"Benchmark failed: {e}")
    
    # Display Results
    if "benchmark_results" in st.session_state:
        br = st.session_state["benchmark_results"]
        
        if br.get("baselines"):
            st.write("#### Test Set Comparison")
            
            # Build comparison table
            metrics = list(br["your_model"].keys())
            table_data = []
            
            for metric in metrics:
                row = {"Metric": metric.upper(), "Your Model": br["your_model"][metric]}
                for baseline_name, baseline_metrics in br["baselines"].items():
                    row[baseline_name] = baseline_metrics.get(metric, 0)
                table_data.append(row)
            
            df_compare = pd.DataFrame(table_data)
            
            # Formatting
            if not df_compare.empty:
                format_dict = {col: "{:.4f}" for col in df_compare.columns if col != "Metric"}
                st.dataframe(df_compare.style.format(format_dict), use_container_width=True, hide_index=True)
            
            # Bar Chart Comparison
            st.write("#### Performance Comparison Chart")
            
            fig_bench, axes = plt.subplots(1, len(metrics), figsize=(max(4, 4 * len(metrics)), 5))
            if len(metrics) == 1:
                axes = [axes]
            
            all_models = ["Your Model"] + list(br["baselines"].keys())
            colors = plt.cm.Set2(np.linspace(0, 1, len(all_models)))
            
            proportion_metrics = ["roc_auc", "f1", "accuracy", "precision", "recall", "r2"]
            high_better = br.get("higher_better", [])
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx]
                values = [br["your_model"][metric]]
                values += [br["baselines"][m].get(metric, 0) for m in br["baselines"].keys()]
                
                bars = ax.bar(range(len(all_models)), values, color=colors)
                ax.set_xticks(range(len(all_models)))
                ax.set_xticklabels([m[:15] + "..." if len(m) > 15 else m for m in all_models], 
                                  rotation=45, ha='right', fontsize=8)
                ax.set_title(metric.upper(), fontsize=11, fontweight='bold')
                
                # Set appropriate Y-axis limits
                if metric in proportion_metrics:
                    max_val = max(values) if max(values) > 0 else 1
                    ax.set_ylim(0, max(1.0, max_val * 1.1))
                else:
                    if max(values) > 0:
                        ax.set_ylim(0, max(values) * 1.2)
                        
                # Best bar annotation
                if metric in high_better:
                    best_idx = values.index(max(values))
                else:
                    best_idx = values.index(min(values))
                    
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(2)
            
            plt.tight_layout()
            st.pyplot(fig_bench)
            
            # Save chart for report
            buf_bench = io.BytesIO()
            fig_bench.savefig(buf_bench, format='png', bbox_inches='tight', dpi=150)
            buf_bench.seek(0)
            context.images["benchmark_comparison"] = buf_bench.read()
            plt.close(fig_bench)
            
            # Save results to context
            context.results["benchmark_results"] = br
            st.toast("Benchmark saved to Report!", icon="📊")
