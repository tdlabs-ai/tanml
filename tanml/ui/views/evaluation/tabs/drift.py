# tanml/ui/pages/evaluation/tabs/drift.py
"""
Drift Analysis Tab - PSI and KS drift detection.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st


from tanml.ui.views.evaluation.tabs import register_tab


@register_tab(name="Drift Analysis (PSI/KS)", order=30, key="tab_drift")
def render(context):
    import io
    import numpy as np
    import pandas as pd
    import streamlit as st
    import matplotlib.pyplot as plt
    import seaborn as sns

    from tanml.analysis.drift import calculate_psi, calculate_ks
    """Render the drift analysis tab."""
    
    st.markdown("### Feature Drift Analysis")
    st.caption("Measures the shift in feature distributions between **Training** (Expected) and **Testing** (Actual) datasets.")


    
    drift_results = []
    
    # Calculate PSI and KS for all numeric features
    for col in context.features:
        if pd.api.types.is_numeric_dtype(context.X_train[col]):
            train_vals = context.X_train[col].dropna()
            test_vals = context.X_test[col].dropna()
            
            psi = calculate_psi(train_vals, test_vals)
            ks_stat, ks_pval = calculate_ks(train_vals, test_vals)
            
            # PSI Status
            psi_status = "🟢 Stable"
            if psi > 0.2: psi_status = "🔴 Critical"
            elif psi > 0.1: psi_status = "🟠 Moderate"
            
            # KS Status
            ks_status = "🟢 Stable"
            if ks_stat > 0.3: ks_status = "🔴 Critical"
            elif ks_stat > 0.2: ks_status = "🟠 Moderate"
            elif ks_stat > 0.1: ks_status = "🟡 Minor"
            
            drift_results.append({
                "Feature": col,
                "PSI": psi,
                "PSI Status": psi_status,
                "KS Stat": ks_stat,
                "KS p-value": ks_pval,
                "KS Status": ks_status
            })
    
    if drift_results:
        df_drift = pd.DataFrame(drift_results).sort_values("KS Stat", ascending=False)
        
        # Legend
        st.markdown("""
        **Thresholds:**
        - **PSI**: 🟢 < 0.1 (Stable), 🟠 0.1-0.2 (Moderate), 🔴 > 0.2 (Critical)
        - **KS**: 🟢 < 0.1 (Stable), 🟡 0.1-0.2 (Minor), 🟠 0.2-0.3 (Moderate), 🔴 > 0.3 (Critical)
        """)
        
        st.dataframe(
            df_drift.style.format({"PSI": "{:.4f}", "KS Stat": "{:.4f}", "KS p-value": "{:.4f}"})
        )
        
        # Plot top drift feature
        st.write("#### Top Drifting Feature Distribution")
        top_drift = df_drift.iloc[0]["Feature"]
        
        fig_drift, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Train Distribution
        axes[0].hist(context.X_train[top_drift].dropna(), bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0].set_title(f"Train: {top_drift}")
        axes[0].set_xlabel("Value")
        axes[0].set_ylabel("Frequency")
        
        # Test Distribution
        axes[1].hist(context.X_test[top_drift].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
        axes[1].set_title(f"Test: {top_drift}")
        axes[1].set_xlabel("Value")
        
        # CDF Comparison
        train_sorted = np.sort(context.X_train[top_drift].dropna())
        test_sorted = np.sort(context.X_test[top_drift].dropna())
        train_cdf = np.arange(1, len(train_sorted)+1) / len(train_sorted)
        test_cdf = np.arange(1, len(test_sorted)+1) / len(test_sorted)
        
        axes[2].plot(train_sorted, train_cdf, 'b-', lw=2, label='Train CDF')
        axes[2].plot(test_sorted, test_cdf, 'orange', lw=2, label='Test CDF')
        axes[2].set_title(f"{top_drift}: CDF (KS={df_drift.iloc[0]['KS Stat']:.3f})")
        axes[2].legend()
        axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig_drift)
        
        # Save for report
        buf = io.BytesIO()
        fig_drift.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        context.images["drift_distribution"] = buf.read()
        plt.close(fig_drift)
        
        # Store results
        context.results["drift"] = drift_results
        st.toast("Drift Analysis saved to Report!", icon="🌊")
    else:
        st.warning("No numeric features available for Drift Analysis.")
