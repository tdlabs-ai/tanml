# tanml/ui/glossary.py
"""
Glossary of technical terms for TanML reports.

Provides definitions for metrics, techniques, and concepts
used in model validation reports.
"""

from __future__ import annotations

GLOSSARY: dict[str, str] = {
    # General Concepts
    "Cross-Validation": (
        "A technique to assess model performance by training and testing on "
        "multiple different subsets of data. Ensures the model generalizes well."
    ),
    "Overfitting": (
        "When a model performs well on training data but poorly on new, unseen data. "
        "Like memorizing answers vs understanding concepts."
    ),
    "Regularization": (
        "Techniques that prevent overfitting by penalizing model complexity. "
        "Helps models generalize to new data."
    ),
    # Classification Metrics
    "ROC AUC": (
        "Area Under the ROC Curve. Measures the model's ability to distinguish "
        "between classes. 1.0 is perfect, 0.5 is random."
    ),
    "KS Statistic": (
        "Kolmogorov-Smirnov statistic. Maximum separation between cumulative "
        "distributions of positive and negative classes. Higher is better."
    ),
    "Gini Coefficient": (
        "Related to AUC: Gini = 2*AUC - 1. Measures inequality in predictions. "
        "1.0 is perfect separation, 0 is random."
    ),
    "F1 Score": (
        "Harmonic mean of Precision and Recall. Balances both metrics, useful "
        "when classes are imbalanced."
    ),
    "PR AUC": (
        "Area Under the Precision-Recall Curve. Focuses on performance on the "
        "positive class, critical for rare events."
    ),
    "Log Loss": (
        "Measures discrimination and confidence. Heavily penalizes confident "
        "wrong answers. Lower is better."
    ),
    "Brier Score": (
        "Mean squared difference between predicted probabilities and actual "
        "outcomes. Lower is better."
    ),
    "Precision": (
        "Of all positive predictions, how many were correct? "
        "Important when false positives are costly."
    ),
    "Recall": (
        "Of all actual positives, how many were found? "
        "Important when false negatives are costly (e.g., fraud detection)."
    ),
    "Accuracy": (
        "Overall correctness: (TP + TN) / Total. Can be misleading for imbalanced datasets."
    ),
    # Regression Metrics
    "RMSE": (
        "Root Mean Squared Error. Average prediction error in target units. "
        "Penalizes large errors heavily. Lower is better."
    ),
    "MAE": (
        "Mean Absolute Error. Average prediction error in target units. "
        "More robust to outliers than RMSE. Lower is better."
    ),
    "R² Score": (
        "Coefficient of determination. Proportion of variance explained by "
        "the model. 1.0 is perfect, 0 is baseline."
    ),
    "Median Absolute Error": (
        "Median of absolute prediction errors. Very robust to outliers. Lower is better."
    ),
    # Diagnostic Features
    "VIF": (
        "Variance Inflation Factor. Measures multicollinearity between features. "
        "VIF > 5-10 indicates concerning correlation with other predictors."
    ),
    "Correlation Heatmap": (
        "Visualizes pairwise correlations between features. High correlations "
        "(>0.8) may indicate redundant features."
    ),
    "PSI": (
        "Population Stability Index. Measures distribution shift between two "
        "datasets. PSI > 0.25 suggests significant drift."
    ),
    # Validation Techniques
    "Stress Test": (
        "Tests model robustness by adding noise to data. Small performance "
        "drops indicate a stable, reliable model."
    ),
    "SHAP": (
        "SHapley Additive exPlanations. Explains *why* a model made a specific "
        "prediction by attributing impact to each feature."
    ),
    "Cluster Coverage": (
        "Measures how well test data covers the training data's input space "
        "using K-Means clustering. Low coverage suggests test data may be "
        "out-of-distribution."
    ),
    # Data Quality
    "Missing Values": (
        "Data points with no recorded value. Can bias models if not handled "
        "properly (imputation, removal, etc.)."
    ),
    "Duplicate Rows": (
        "Identical records in the dataset. May cause data leakage if same row "
        "appears in both train and test sets."
    ),
    "Constant Columns": (
        "Features with only one unique value. Provide no predictive power and should be removed."
    ),
    "Power Score": (
        "A normalized metric (0-100) representing the relative importance of a feature "
        "for predicting the target. 100 is the most important feature."
    ),
}


def get_definition(term: str) -> str:
    """
    Get the definition for a term.

    Args:
        term: Technical term to look up

    Returns:
        Definition string or empty string if not found
    """
    return GLOSSARY.get(term, "")


def get_all_terms() -> list[str]:
    """
    Get all terms in the glossary.

    Returns:
        List of term names
    """
    return list(GLOSSARY.keys())


def format_glossary_html() -> str:
    """
    Format glossary as HTML for display.

    Returns:
        HTML string with formatted glossary
    """
    html_parts = ['<div class="tanml-glossary">']
    for term, definition in sorted(GLOSSARY.items()):
        html_parts.append(f"<p><strong>{term}</strong>: {definition}</p>")
    html_parts.append("</div>")
    return "\n".join(html_parts)
