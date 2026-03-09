# tanml/ui/pages/evaluation/tabs/inference.py
"""
Statistical Inference Tab - Renders specialized statsmodels tables.
"""

import numpy as np
import pandas as pd
import streamlit as st

from tanml.ui.views.evaluation.tabs import register_tab


@register_tab(name="Statistical Inference", order=15, key="tab_inference")
def render(context):
    """Render the statistical inference tab, only for statsmodels."""
    
    # Only render if the model is our StatsModelsWrapper
    # The wrapper is created in main.py if library == 'statsmodels'
    if not hasattr(context.model, "sm_model"):
        st.info("Statistical inference is only available when using the 'statsmodels' library.")
        return

    st.markdown("### 🔬 Statistical Inference")
    
    sm_model = context.model.sm_model
    algo = context.model.algo
    features = context.model.feats
    
    # Define tabs based on algorithm
    if algo == "Logit":
        t_coef, t_summ, t_odds = st.tabs(["Coefficients", "Model Summary", "Odds Ratios"])
    else:
        t_coef, t_summ = st.tabs(["Coefficients", "Model Summary"])

    # Extract Summary Data
    sm_res = sm_model
    summary = sm_res.summary2()

    with t_coef:
        st.write("**Model Coefficients**")
        df_coef = summary.tables[1].copy()
        try:
            df_coef.columns = ["Coef", "Std.Err.", "t/z", "P>|t|", "[0.025", "0.975]"]
        except Exception:
            pass
        def color_p_value(val):
            """Color specific significance thresholds."""
            if val < 0.01:
                return 'background-color: rgba(34, 197, 94, 0.2); color: #15803d;' # Significant (Strong)
            if val < 0.05:
                return 'background-color: rgba(34, 197, 94, 0.1); color: #166534;' # Significant
            if val < 0.1:
                return 'background-color: rgba(234, 179, 8, 0.1); color: #854d0e;' # Borderline
            return ''

        st.dataframe(df_coef.style.format("{:.4f}").applymap(
            color_p_value, subset=["P>|t|"] if "P>|t|" in df_coef.columns else []
        ))
        
        # Calculate p-value power ranking
        if "P>|t|" in df_coef.columns:
            st.divider()
            c_p1, c_p2 = st.columns([2, 1])
            with c_p1:
                st.write("**Feature Power Ranking (by P-value)**")
                # Exclude constant
                if "const" in df_coef.index:
                    df_p = df_coef.drop("const").copy()
                else:
                    df_p = df_coef.copy()
                
                df_p = df_p.sort_values(by="P>|t|", ascending=True)
                df_p = df_p[["P>|t|", "Coef"]]
                st.dataframe(df_p.style.format("{:.4f}").applymap(
                    color_p_value, subset=["P>|t|"]
                ))
            
            with c_p2:
                st.info("Significance: Strong Green (< 0.01), Green (< 0.05), Yellow (< 0.10). "
                       "The smaller the p-value, the stronger the evidence that the feature is meaningful.")

    with t_summ:
        st.write("**Model Fit Summary**")
        
        # Build dictionary from statsmodels properties
        summ_data = {
            "Algorithm": "Logistic Regression" if algo == "Logit" else "Ordinary Least Squares",
            "Observations (No.)": int(sm_res.nobs),
            "Df Residuals": int(sm_res.df_resid),
            "Df Model": int(sm_res.df_model),
            "AIC": sm_res.aic,
            "BIC": sm_res.bic,
        }
        
        if algo == "OLS":
            summ_data["R-squared"] = sm_res.rsquared
            summ_data["Adj. R-squared"] = sm_res.rsquared_adj
            summ_data["F-statistic"] = sm_res.fvalue
            summ_data["Prob (F-statistic)"] = sm_res.f_pvalue
        elif algo == "Logit":
            summ_data["Pseudo R-squared"] = sm_res.prsquared
            summ_data["Log-Likelihood"] = sm_res.llf
            summ_data["LL-Null"] = sm_res.llnull
            summ_data["LLR p-value"] = sm_res.llr_pvalue
        
        st.json(summ_data)

    if algo == "Logit":
        with t_odds:
            st.write("**Odds Ratios (exp(Coef))**")
            or_df = pd.DataFrame({
                "Odds Ratio": np.exp(sm_res.params),
                "Lower 95%": np.exp(sm_res.conf_int()[0]),
                "Upper 95%": np.exp(sm_res.conf_int()[1])
            })
            st.dataframe(or_df.style.format("{:.4f}"))
            st.info("Odds Ratio interpretation: A unit increase in X increases the odds of Y by a factor of the Odds Ratio, holding all other variables constant.")

    # Save payload to context for report generator
    sm_payload = {
        "coef_table": df_coef.to_dict(orient="index"),
        "summary_stats": summ_data,
    }
    if algo == "Logit" and "or_df" in locals():
        sm_payload["odds_ratios"] = or_df.to_dict(orient="index")
        
    context.results["statsmodels_inference"] = sm_payload
