# tanml/ui/pages/evaluation/tabs/shap.py
"""
Explainability Tab - SHAP analysis.
"""

import pandas as pd
import streamlit as st

from tanml.ui.pages.evaluation.tabs import register_tab


@register_tab(name="Explainability (SHAP)", order=70, key="tab_exp")
def render(context):
    """Render the SHAP analysis tab."""
    
    st.markdown("### Model Explainability (SHAP)")
    st.caption("Understand feature importance and how features drive the model's predictions.")

    with st.expander("ℹ️ Interpreting SHAP Plots"):
        st.markdown("""
        **Beeswarm Plot (Global Info)**
        - Shows **feature impact** (x-axis) and **feature value** (color).
        - **Red dots**: High feature values. **Blue dots**: Low feature values.
        - If red dots are on the right, high values *increase* the prediction.
        
        **Bar Plot (Feature Importance)**
        - Shows the average magnitude of impact for each feature.
        - Longer bars mean the feature is more important overall.
        """)
    
    
    col1, col2 = st.columns(2)
    with col1:
        bg_size = st.slider("Background Samples (K-Means)", 10, 500, 50, step=10, help="Number of samples from training data to use as baseline reference.")
    with col2:
        test_size = st.slider("Test Samples to Explain", 10, 1000, 100, step=10, help="Number of samples from test data to calculate SHAP values for.")

    if st.button("🚀 Start Analysis", type="primary"):
        with st.spinner("Running SHAP Check (this may take a minute)..."):
            try:
                import importlib
                import tanml.checks.explainability.shap_check
                importlib.reload(tanml.checks.explainability.shap_check)
                from tanml.checks.explainability.shap_check import SHAPCheck
                
                # Config with user values - Enforce run output directory
                artifacts_dir = context.run_dir / "artifacts"
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                
                scog = {
                     "explainability": {
                          "shap": {
                               "background_sample_size": bg_size, 
                               "test_sample_size": test_size,
                               "out_dir": str(artifacts_dir)
                          }
                     }
                }
                
                # Note: SHAPCheck expects X_train, X_test, y_train, y_test
                shap_check = SHAPCheck(
                    context.model, 
                    context.X_train, 
                    context.X_test, 
                    context.y_train, 
                    context.y_test, 
                    rule_config=scog
                )
                res = shap_check.run()
                
                if res.get("status") == "ok":
                    plots = res.get("plots", {})
                    
                    c_s1, c_s2 = st.columns(2)
                    
                    with c_s1:
                        if "beeswarm" in plots:
                            st.image(plots["beeswarm"], caption="SHAP Beeswarm Plot (Global Info)")
                            # Save to context for potential report use (if needed)
                    
                    with c_s2:
                        if "bar" in plots:
                            st.image(plots["bar"], caption="SHAP Bar Plot (Feature Importance)")
                            
                    # Top Features Table
                    if "top_features" in res:
                        st.write("**Top Feature Impacts**")
                        st.dataframe(pd.DataFrame(res["top_features"]))
                    
                    # Store results for report
                    context.results["explainability"] = res
                    st.toast("SHAP Results saved to Report!", icon="🧠")
                    
                else:
                    st.error(f"SHAP Analysis Error: {res.get('status')}")
                    
            except Exception as e:
                st.error(f"SHAP Failed: {e}")
