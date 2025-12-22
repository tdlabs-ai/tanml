# tanml/ui/pages/evaluation/tabs/stress.py
"""
Stress Testing Tab - Robustness checks.
"""

import streamlit as st

from tanml.ui.pages.evaluation.tabs import register_tab


@register_tab(name="Stress Testing", order=60, key="tab_stress")
def render(context):
    """Render the stress testing tab."""
    
    st.markdown("### Stress Testing (Robustness)")
    st.caption("Evaluates how model performance degrades when **Testing Data** is perturbed with noise (simulating poor data quality).")

    col1, col2 = st.columns(2)
    with col1:
        epsilon = st.slider("Perturbation Magnitude", 0.001, 0.5, 0.01, step=0.001, format="%.3f", help="Magnitude of noise to add (e.g. 0.01 = 1%)")
    with col2:
        fraction = st.slider("Affected Data Fraction", 0.1, 1.0, 0.2, step=0.1, help="Fraction of rows to perturb")
    
    if st.button("Run Stress Test", type="secondary"):
        with st.spinner("Running Stress Checks..."):
            try:
                from tanml.checks.stress_test import StressTestCheck
                
                # Initialize and Run check on TEST set
                stress_check = StressTestCheck(
                    context.model, 
                    context.X_test, 
                    context.y_test, 
                    epsilon=epsilon, 
                    perturb_fraction=fraction
                )
                df_stress = stress_check.run()
                
                st.write(f"**Stress Test Results (Perturbation +/- {epsilon*100:.1f}%)**")
                
                # Highlight drops
                if "delta_accuracy" in df_stress.columns:
                    st.dataframe(
                        df_stress.style.format({
                            "accuracy": "{:.4f}", "auc": "{:.4f}", 
                            "delta_accuracy": "{:.4f}", "delta_auc": "{:.4f}"
                        }).bar(subset=["delta_accuracy"], color=['#ffcccc', '#ccffcc'], align='zero')
                    )
                else:
                    st.dataframe(
                        df_stress.style.format({
                            "rmse": "{:.4f}", "r2": "{:.4f}",
                            "delta_rmse": "{:.4f}", "delta_r2": "{:.4f}"
                        })
                    )
                
                # Store results for report
                context.results["stress"] = df_stress.to_dict(orient="records")
                st.toast("Stress Test saved to Report!", icon="💥")
                
            except Exception as e:
                st.error(f"Stress Test Failed: {e}")
