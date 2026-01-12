# tanml/ui/pages/evaluation/tabs/benchmark.py
"""
Benchmark Tab - Compares model against baselines.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from tanml.ui.views.evaluation.tabs import register_tab


@register_tab(name="Benchmarking", order=50, key="tab_bench")
def render(context):
    import io
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns
    """Render the benchmarking tab."""
    from tanml.models.registry import list_models, build_estimator, ui_schema_for, get_spec
    
    st.markdown("### Benchmarking: Compare Your Model vs Baseline Models")
    st.caption("Select one or more baseline models to compare against your trained model.")
    
    # 1. Fetch Available Models
    available_specs = list_models(context.task_type)
    # Format options as "Library: Algorithm"
    model_options = [f"{lib}: {algo}" for (lib, algo) in available_specs.keys()]
    
    selected_model_strs = st.multiselect(
        "Select Baseline Models to Compare",
        options=model_options,
        default=[model_options[0]] if model_options else None,
        help="Choose one or more models to benchmark against"
    )
    
    # 2. Dynamic Hyperparameters
    bench_configs = {} # Store (lib, algo, params) for each selected model
    
    if selected_model_strs:
        with st.expander("Advanced Hyperparameters", expanded=False):
            st.caption("Adjust hyperparameters for selected models.")
            
            for m_str in selected_model_strs:
                lib, algo = m_str.split(": ")
                st.markdown(f"**{algo}** ({lib})")
                
                # Fetch schema and defaults
                spec = get_spec(lib, algo)
                schema = spec.ui_schema
                defaults = spec.defaults
                
                # Container for this model's params
                params = {} 
                
                # Render Form (Simplified version of forms.py)
                c1, c2 = st.columns(2)
                seed_keys = [k for k in ("random_state", "seed", "random_seed") if k in defaults]
                
                # Filter out seed keys (we'll handle globally or ignore for simplicity in benchmark)
                valid_items = {k: v for k, v in schema.items() if k not in seed_keys}
                
                for i, (name, (typ, choices, helptext)) in enumerate(valid_items.items()):
                    with (c1 if i % 2 == 0 else c2):
                        default_val = defaults.get(name)
                        key_widget = f"bench_{lib}_{algo}_{name}"
                        
                        if typ == "choice":
                            opts = list(choices) if choices else []
                            show = ["None" if o is None else o for o in opts]
                            idx = 0
                            if default_val in show: idx = show.index(default_val)
                            elif default_val is None and "None" in show: idx = show.index("None")
                            
                            sel = st.selectbox(name, show, index=idx, key=key_widget, help=helptext)
                            params[name] = None if sel == "None" else sel
                            
                        elif typ == "bool":
                            params[name] = st.checkbox(name, value=bool(default_val), key=key_widget, help=helptext)
                            
                        elif typ == "int":
                            # Handle None default for int (rare but possible for 'max_depth')
                            val = int(default_val) if default_val is not None else 0
                            params[name] = int(st.number_input(name, value=val, step=1, key=key_widget, help=helptext))
                            
                        elif typ == "float":
                            params[name] = float(st.number_input(name, value=float(default_val or 0.0), key=key_widget, help=helptext))
                        
                        else: # str
                            params[name] = st.text_input(name, value=str(default_val or ""), key=key_widget, help=helptext)

                # Store config
                bench_configs[m_str] = {"lib": lib, "algo": algo, "params": params}
                st.divider()

    if st.button("🔬 Run Benchmark Comparison", type="secondary", key="btn_benchmark"):
        if not selected_model_strs:
            st.warning("Please select at least one baseline model.")
        else:
            with st.spinner("Training baseline models..."):
                try:
                    from sklearn.metrics import (
                        roc_auc_score, f1_score, accuracy_score, precision_score, recall_score,
                        mean_squared_error, mean_absolute_error, r2_score
                    )
                    
                    # Prepare data
                    # Registry models expect numeric inputs generally, but some handle cats.
                    # For safety in this benchmark, we'll stick to numeric features similar to original
                    # BUT ideally we should pass what the model allows. For now, keep simple numeric select.
                    X_tr = context.X_train.select_dtypes(include=np.number).fillna(0)
                    X_te = context.X_test.select_dtypes(include=np.number).fillna(0)
                    
                    benchmark_results = {"your_model": {}, "baselines": {}}
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
                    for m_str in selected_model_strs:
                        cfg = bench_configs[m_str]
                        try:
                            # Build and Train
                            mdl = build_estimator(cfg["lib"], cfg["algo"], cfg["params"])
                            mdl.fit(X_tr, context.y_train)
                            
                            pred = mdl.predict(X_te)
                            
                            # Metrics
                            if context.task_type == "classification":
                                try:
                                    pred_prob = mdl.predict_proba(X_te)[:, 1]
                                except:
                                    pred_prob = np.zeros(len(pred)) # Fallback
                                
                                benchmark_results["baselines"][m_str] = {
                                    "roc_auc": roc_auc_score(context.y_test, pred_prob) if len(np.unique(pred_prob)) > 1 else 0.5,
                                    "f1": f1_score(context.y_test, pred),
                                    "accuracy": accuracy_score(context.y_test, pred),
                                    "precision": precision_score(context.y_test, pred, zero_division=0),
                                    "recall": recall_score(context.y_test, pred, zero_division=0)
                                }
                            else:
                                benchmark_results["baselines"][m_str] = {
                                    "rmse": np.sqrt(mean_squared_error(context.y_test, pred)),
                                    "mae": mean_absolute_error(context.y_test, pred),
                                    "r2": r2_score(context.y_test, pred)
                                }
                        
                        except Exception as model_err:
                            st.warning(f"Failed to train {m_str}: {model_err}")
                    
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
                st.dataframe(df_compare.style.format(format_dict), width="stretch", hide_index=True)
            
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
