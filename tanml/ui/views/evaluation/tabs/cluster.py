# tanml/ui/pages/evaluation/tabs/cluster.py
"""
Input Cluster Coverage Tab - Analyzes input space coverage.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from tanml.ui.views.evaluation.tabs import register_tab
from tanml.analysis.clustering import analyze_cluster_coverage


@register_tab(name="Input Cluster Coverage Check", order=40, key="tab_cluster")
def render(context):
    """Render the cluster coverage tab."""
    
    st.markdown("### Input Cluster Coverage Check")
    st.caption("Evaluates how well the **Testing Data** covers the input space defined by the **Training Data** clusters. Low coverage may indicate the model is being applied to out-of-distribution (OOD) samples.")
    
    # User selects number of clusters
    col_clust_opt, col_clust_btn = st.columns([1, 2])
    with col_clust_opt:
        n_clusters_input = st.slider("Number of Clusters", min_value=2, max_value=20, value=5, 
                                    help="Choose the number of K-Means clusters to partition the training data into.")
    
    with col_clust_btn:
        run_cluster = st.button("🎯 Run Cluster Coverage Check", type="secondary", key="btn_cluster_cov")
    
    if run_cluster:
        with st.spinner("Running K-Means Clustering..."):
            try:
                # Use the analysis module for clustering
                cluster_results_raw = analyze_cluster_coverage(
                    X_train=context.X_train,
                    X_test=context.X_test,
                    n_clusters=n_clusters_input,
                )
                
                # Build UI-friendly results from analysis module output
                n_clusters = cluster_results_raw["n_clusters"]
                coverage_pct = cluster_results_raw["coverage_pct"]
                uncovered_count = cluster_results_raw["uncovered_count"]
                cluster_dist = cluster_results_raw["cluster_distribution"]
                
                # Calculate OOD metrics
                ood_pct = 100 - coverage_pct
                ood_indices = cluster_results_raw["uncovered_indices"]
                
                # Build cluster summary table
                cluster_summary = []
                for c in range(n_clusters):
                    if c in cluster_dist:
                        train_c = cluster_dist[c]["train_count"]
                        test_c = cluster_dist[c]["test_count"]
                    else:
                        train_c, test_c = 0, 0
                    cluster_summary.append({
                        "Cluster": c,
                        "Train Count": int(train_c),
                        "Train %": f"{train_c / len(context.X_train) * 100:.1f}%",
                        "Test Count": int(test_c),
                        "Test %": f"{test_c / len(context.X_test) * 100:.1f}%",
                        "Status": "✓ Covered" if test_c > 0 else "✗ Uncovered"
                    })
                
                cluster_results = {
                    "n_clusters": n_clusters,
                    "coverage_pct": coverage_pct,
                    "covered_clusters": n_clusters - len([c for c in cluster_dist.values() if c["test_count"] == 0]),
                    "uncovered_clusters": len([c for c in cluster_dist.values() if c["test_count"] == 0]),
                    "ood_pct": ood_pct,
                    "ood_count": uncovered_count,
                    "ood_indices": ood_indices,
                    "cluster_summary": cluster_summary
                }
                
                # Store OOD samples data for download
                if uncovered_count > 0 and ood_indices:
                    ood_df = context.X_test.iloc[ood_indices].copy()
                    st.session_state["eval_ood_samples"] = ood_df
                
                # Store PCA data for visualization (from analysis module)
                train_pca = np.array(cluster_results_raw.get("train_pca", []))
                test_pca = np.array(cluster_results_raw.get("test_pca", []))
                centers_pca = np.array(cluster_results_raw.get("cluster_centers_pca", []))
                
                st.session_state["eval_cluster_pca"] = {
                    "train_pca": train_pca,
                    "test_pca": test_pca,
                    "centers_pca": centers_pca,
                    "train_labels": cluster_results_raw.get("train_labels", []),
                    "test_labels": cluster_results_raw.get("test_labels", []),
                    "n_clusters": n_clusters,
                    "total_train": len(context.X_train),
                    "total_test": len(context.X_test)
                }
                
                # Save to session state
                st.session_state["eval_cluster_results"] = cluster_results
                
                st.success(f"Cluster Coverage Check Complete!")
                
            except Exception as e:
                st.error(f"Cluster Coverage Check Failed: {e}")
    
    # Display Results if available
    if "eval_cluster_results" in st.session_state:
        cr = st.session_state["eval_cluster_results"]
        
        # Key Metrics
        st.write("#### Coverage Summary")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Clusters", cr["n_clusters"])
        k2.metric("Coverage", f"{cr['coverage_pct']:.1f}%", 
                 delta=None if cr['coverage_pct'] >= 90 else f"-{100 - cr['coverage_pct']:.1f}%")
        k3.metric("Uncovered Clusters", cr["uncovered_clusters"],
                 delta_color="inverse" if cr["uncovered_clusters"] > 0 else "off")
        k4.metric("Potential OOD Samples", f"{cr['ood_pct']:.1f}%",
                 delta=f"{cr['ood_count']} samples" if cr['ood_count'] > 0 else None,
                 delta_color="inverse" if cr['ood_count'] > 0 else "off")
        
        # Interpretation
        st.write("")
        if cr['coverage_pct'] >= 95:
            st.success("✅ **Excellent Coverage**: Test data covers nearly all training input space clusters.")
        elif cr['coverage_pct'] >= 80:
            st.warning("⚠️ **Good Coverage**: Most clusters are covered, but some regions of the input space are not represented in testing.")
        else:
            st.error("🚨 **Poor Coverage**: Test data does not adequately cover the training input space. Model may be applied to unfamiliar data patterns.")
        
        if cr['ood_pct'] > 10:
            st.warning(f"⚠️ **OOD Alert**: {cr['ood_pct']:.1f}% of test samples are far from any training cluster center (potential out-of-distribution data).")
        
        # OOD Samples Viewer & Download
        if cr['ood_count'] > 0 and "eval_ood_samples" in st.session_state:
            with st.expander(f"📋 View & Download OOD Samples ({cr['ood_count']} samples)", expanded=False):
                ood_df = st.session_state["eval_ood_samples"]
                
                st.write("**Out-of-Distribution Samples**")
                st.caption("These test samples are far from any training cluster center, suggesting they may represent data patterns not seen during training.")
                
                # Show dataframe (sort by first column if _ood_distance not available)
                if '_ood_distance' in ood_df.columns:
                    display_df = ood_df.sort_values('_ood_distance', ascending=False).reset_index(drop=True)
                else:
                    display_df = ood_df.reset_index(drop=True)
                st.dataframe(display_df, width="stretch", height=300)
                
                # Download button
                csv_data = display_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Download OOD Samples (CSV)",
                    data=csv_data,
                    file_name="ood_samples.csv",
                    mime="text/csv",
                    type="primary"
                )
        
        # Cluster Summary Table
        st.write("#### Cluster Distribution")
        st.dataframe(
            pd.DataFrame(cr["cluster_summary"]).style
            .map(lambda v: "color: green" if "Covered" in str(v) else "color: red" if "Uncovered" in str(v) else "", subset=["Status"])
        )
        
        # Visualization
        st.write("#### Train vs Test Cluster Distribution (Share of Records)")
        
        fig_clust, ax_clust = plt.subplots(figsize=(12, 5))
        x_pos = np.arange(cr["n_clusters"])
        width = 0.35
        
        # Calculate percentages instead of counts
        train_counts = [d["Train Count"] for d in cr["cluster_summary"]]
        test_counts = [d["Test Count"] for d in cr["cluster_summary"]]
        total_train = sum(train_counts)
        total_test = sum(test_counts)
        
        train_pcts = [c / total_train * 100 if total_train > 0 else 0 for c in train_counts]
        test_pcts = [c / total_test * 100 if total_test > 0 else 0 for c in test_counts]
        
        # Create bars
        bars_train = ax_clust.bar(x_pos - width/2, train_pcts, width, label='Train', color='steelblue', alpha=0.8)
        bars_test = ax_clust.bar(x_pos + width/2, test_pcts, width, label='Test', color='darkorange', alpha=0.8)
        
        # Add annotations
        for bar in bars_train:
            height = bar.get_height()
            ax_clust.annotate(f'{height:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold', color='steelblue')
        
        for bar in bars_test:
            height = bar.get_height()
            ax_clust.annotate(f'{height:.1f}%',
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkorange')
        
        ax_clust.set_xlabel('Cluster ID', fontsize=10)
        ax_clust.set_ylabel('Share of Records (%)', fontsize=10)
        ax_clust.set_title('Cluster Distribution: Train vs Test (Percentage)', fontsize=12)
        ax_clust.set_xticks(x_pos)
        ax_clust.set_xticklabels([f'C{i}' for i in x_pos])
        ax_clust.legend(loc='upper right')
        ax_clust.grid(axis='y', alpha=0.3)
        ax_clust.set_ylim(0, max(max(train_pcts), max(test_pcts)) * 1.2)
        
        st.pyplot(fig_clust)
        
        # Save image for report
        buf_clust = io.BytesIO()
        fig_clust.savefig(buf_clust, format='png', bbox_inches='tight')
        buf_clust.seek(0)
        context.images["cluster_distribution"] = buf_clust.read()
        plt.close(fig_clust)
        
        # 2nd Diagram: PCA Scatter Plot
        st.write("#### Cluster Space Visualization (2D Projection)")
        st.caption("💡 **How to read**: Gray = Training data. Colored squares = Test data. Each centroid shows **Train% / Test% (ratio)**.")
        
        if "eval_cluster_pca" in st.session_state:
            pca_data = st.session_state["eval_cluster_pca"]
            train_pca = np.array(pca_data["train_pca"])
            test_pca = np.array(pca_data["test_pca"])
            centers_pca = np.array(pca_data["centers_pca"])
            test_labels = np.array(pca_data["test_labels"])
            n_clusters = pca_data["n_clusters"]
            
            fig_pca, ax_pca = plt.subplots(figsize=(12, 8))
            
            # Color palette
            cluster_colors = plt.cm.Set1(np.linspace(0, 1, max(n_clusters, 9)))[:n_clusters]
            
            # Plot training points in gray
            ax_pca.scatter(train_pca[:, 0], train_pca[:, 1], 
                          c='lightgray', alpha=0.25, s=10, label='Train Data', zorder=1)
            
            # Plot test points
            for c in range(n_clusters):
                mask = test_labels == c
                if mask.sum() > 0:
                    ax_pca.scatter(test_pca[mask, 0], test_pca[mask, 1],
                                  c=[cluster_colors[c]], alpha=0.85, s=60, marker='s', 
                                  edgecolors='black', linewidths=0.5, zorder=3)
            
            # Plot centers
            for i, (cx, cy) in enumerate(centers_pca):
                ax_pca.scatter(cx, cy, c=[cluster_colors[i]], s=400, marker='X', 
                              edgecolors='black', linewidths=2, zorder=5)
            
            ax_pca.set_title("Input Data Projection (PCA)", fontsize=14)
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
            ax_pca.grid(alpha=0.2)
            
            ax_pca.set_title("Input Data Projection (PCA)", fontsize=14)
            ax_pca.set_xlabel("PC1")
            ax_pca.set_ylabel("PC2")
            ax_pca.grid(alpha=0.2)
            
            st.pyplot(fig_pca)

            # Save PCA image for report
            buf_pca = io.BytesIO()
            fig_pca.savefig(buf_pca, format='png', bbox_inches='tight')
            buf_pca.seek(0)
            context.images["cluster_pca"] = buf_pca.read()
            plt.close(fig_pca)

            # Save results to context
            context.results["cluster_coverage"] = cr
            st.toast("Cluster Coverage saved to Report!", icon="🎯")
