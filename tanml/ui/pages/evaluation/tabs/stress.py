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
    
    if st.button("Run Stress Test", type="secondary"):
        with st.spinner("Running Stress Checks..."):
            try:
                from tanml.checks.stress_test import StressTestCheck
                
                # Initialize and Run check on TEST set
                stress_check = StressTestCheck(
                    context.model, 
                    context.X_test, 
                    context.y_test, 
                    epsilon=0.01, 
                    perturb_fraction=0.2
                )
                df_stress = stress_check.run()
                
                st.write("**Stress Test Results (Perturbation +/- 1%)**")
                
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
