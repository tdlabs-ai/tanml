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
    
    if st.button("Run SHAP Analysis", type="secondary"):
        with st.spinner("Running SHAP Check (this may take a minute)..."):
            try:
                from tanml.checks.explainability.shap_check import SHAPCheck
                
                # We use 100 bg samples and 100 test samples by default for speed
                scog = {"explainability": {"shap": {"background_sample_size": 50, "test_sample_size": 100}}}
                
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
