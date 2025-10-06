# -----------------------------------------------------------------------------
# File: tanml/checks/input_cluster.py
# -----------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class InputClusterCoverageCheck:
    """Cluster the model inputs and report their distribution.

    On run() this class will:
    • auto‑select k (or respect the YAML‑supplied one)
    • save a cluster_distribution.csv  and a input_cluster_plot.png
    • return a result‑dict consumed by the report builder.
    """

    def __init__(
        self,
        cleaned_df: pd.DataFrame,
        feature_names: list[str],
        rule_config: dict | None = None,
    ) -> None:
        self.cleaned_df = cleaned_df
        self.feature_names = feature_names

        cfg = (rule_config or {}).get("InputClusterCoverageCheck", {})
        self.n_clusters: int | None = cfg.get("n_clusters")  # None → auto
        self.max_k: int = cfg.get("max_k", 10)  # elbow upper bound

 
    def _auto_select_k(self, X_scaled: pd.DataFrame) -> int:
        """Simple elbow‑method heuristic."""
        inertias: list[float] = []
        for k in range(1, self.max_k + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)

        deltas = pd.Series(inertias).diff().iloc[1:]
        second_delta = deltas.diff().iloc[1:]
        if second_delta.empty:
            return 2  # fall‑back if data are too small
        return second_delta.idxmax() + 1  # +1 because diff() shifts index

 
    def run(self) -> dict:
        missing = set(self.feature_names) - set(self.cleaned_df.columns)
        if missing:
            raise ValueError(f"Features not in cleaned_df: {missing}")

        X = self.cleaned_df[self.feature_names]
        X_scaled = StandardScaler().fit_transform(X)

        if self.n_clusters is None:
            self.n_clusters = self._auto_select_k(X_scaled)

        km = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)

        dfc = X.copy()
        dfc["cluster"] = labels

        summary = (
            dfc["cluster"].value_counts().sort_index().reset_index(name="Count")
        )
        summary.columns = ["Cluster", "Count"]
        total = len(dfc)
        summary["Percent"] = (summary["Count"] / total * 100).round(2)

        Path("reports/clusters").mkdir(parents=True, exist_ok=True)
        Path("reports/images").mkdir(parents=True, exist_ok=True)

        csv_path = "reports/clusters/cluster_distribution.csv"
        plot_path = "reports/images/input_cluster_plot.png"

        summary.to_csv(csv_path, index=False)
        self._save_bar_chart(summary, plot_path)

        return {
            "cluster_table": summary.to_dict(orient="records"),
            "cluster_csv": csv_path,
            "cluster_plot_img": plot_path,
            "n_clusters": self.n_clusters,
        }

    @staticmethod
    def _save_bar_chart(summary: pd.DataFrame, plot_path: str) -> None:
        """Helper used internally and by tests."""
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.barh(y=summary["Cluster"].astype(str), width=summary["Count"])
        ax.set_xlabel("Count")
        ax.set_ylabel("Cluster")
        ax.set_title("Input Cluster Distribution")
        max_count = summary["Count"].max() or 0
        for bar in bars:
            w = bar.get_width()
            ax.text(
                w + max_count * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{int(w)}",
                va="center",
            )
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
