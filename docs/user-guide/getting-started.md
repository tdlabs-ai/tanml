# Getting Started

This guide walks you through the end-to-end lifecycle of a model validation project using TanML.

## Prerequisites

TanML requires **Python 3.10 or higher**. We strongly recommend using a virtual environment:

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows

# Install TanML
pip install tanml
```

## The Validation Lifecycle

Launch the dashboard to begin:

```bash
tanml ui
```

### Step 1: Understand Your Data (Data Profiling)

#### Overview  
Start by uploading your raw dataset. TanML supports a wide range of file formats, including:

- **Standard formats:** CSV, TSV, Excel (.xlsx, .xls), JSON, TXT  
- **High-performance formats:** Parquet  
- **Statistical/legacy formats:** Stata (.dta), SPSS (.sav), SAS (.sas7bdat)  

TanML performs automated exploratory data analysis directly within the interface to assess data quality and statistical structure. This step is **target-agnostic**, meaning it helps you identify issues across the entire dataset before you even decide what you intend to predict.

---

#### Functionalities  

**Missing Value Analysis**  
TanML computes the percentage of missing values for each feature and provides visual bar charts to highlight gaps. This helps identify features that require imputation or removal in Step 2.

**Distribution Analysis**  
The tool visualizes the distribution of numerical variables using **Histograms** and **Box Plots**. This allows users to detect skewness, median shifts, and the spread of data.

**Outlier Identification**  
TanML uses statistical IQR thresholds to surface extreme observations. Users can also set **custom domain constraints** (Min/Max) to identify data points that violate business logic.

**Duplicate Detection**  
Users can identify identical records based on a custom subset of columns. This is critical for ensuring data integrity and preventing "leakage" before the data is split.

**Automated Risk Flags**  
TanML automatically flags columns with high cardinality (too many unique categories) or near-constant values, warning you of features that might cause problems for machine learning models later.

---

#### Outputs  
- **Interactive Visualizations**: Histograms, Box Plots, and Correlation Heatmaps.  
- **Summary Statistics**: Automated calculation of Mean, Median, Std Dev, and unique counts.  
- **Downloadable Reports**: Export specific lists of identified **Outliers** or **Duplicates** as CSV files for manual review.

---

#### Why This Step Matters  
This step establishes a "Health Check" for your data. It enables you to catch "Garbage In, Garbage Out" scenarios early, ensuring that your preprocessing decisions in the next step are based on a solid statistical understanding of your raw data. 

### Step 2: Clean & Split (Data Preprocessing)

#### Overview  
This step prepares the dataset for modeling by handling missing values, encoding categorical variables, and creating separate training and testing datasets. TanML ensures that every transformation is logged and reproducible.

---

#### Functionalities  

**Advanced Imputation**  
TanML provides several strategies for handling gaps in your data:
- **Numeric**: Choose from `Mean`, `Median`, `Zero`, or the advanced **KNN (K-Nearest Neighbors)** imputer.
- **Categorical**: Apply `Mode` (Most Frequent) or create an explicit **"Missing_Label"** category.
- **Row Deletion**: Use the **"Drop Rows"** option to remove records with missing data entirely.

**Categorical Encoding**  
Convert text columns into numeric formats that ML algorithms can process:
- **One-Hot Encoding**: Best for nominal data (e.g., "Color").
- **Label Encoding**: Best for ordinal data or high-cardinality features.

**Data Splitting & Stratification**  
Divide your dataset into a **Training Set** and a **Testing Set**.  
- **Stratify**: Ensures that the class balance (e.g., default rate) remains identical in both sets.
- **Shuffle & Seed**: Control the randomness to ensure your experiment is reproducible.
---

#### Outputs  
- **Cleaned Data**: The full dataset after imputation and encoding.
- **Train/Test Sets**: Separate files ready for Step 3 and Step 4.
- **Multi-Format Export**: Download your work in `CSV`, `Parquet`, `Excel`, `JSON`, `Pickle`, `Feather`, `Stata (.dta)`, `SPSS (.sav)`, or `SAS (.xpt)`.

---

#### Why This Step Matters  
This step ensures your model receives clean, consistent inputs. By using **Stratified Splitting**, you prevent "unlucky" data distributions from ruining your validation. 

### Step 3: Analyze Features (Feature Power Ranking)

#### Overview  
This step evaluates the predictive strength of each feature in the dataset. By identifying which variables contribute meaningfully to the target, you can drop "noisy" data, simplify your model, and meet regulatory requirements for feature selection.

---

#### Functionalities  

**Multi-Method Ranking**  
Choose how TanML calculates the "Power" of your features:
- **Statistical Correlation**: Best for linear relationships.
- **XGBoost / Decision Trees**: Best for capturing complex, non-linear interactions.

**The Metrics Board (Leaderboard)**  
TanML provides a multi-dimensional view of feature quality:
- **Power Score (0–100)**: A normalized measure of predictive strength.
- **Univariate p-values**: Statistically independent significance tests (calculated via Statsmodels) to prove a variable is mathematically relevant.
- **Information Value (IV) & Gini**: Estimated scores specifically for classification tasks.

**Visual Analysis & "Sanity Checks"**  
- **Distribution Overlay**: Visualize how a feature distributes relative to the target (e.g., density plots for default vs. non-default groups).
- **Correlation Heatmap**: Identify **Multicollinearity** (redundant features) to ensure your model stays stable.

**Automated Word (.docx) Reporting**  
One-click generation of a professional **Power Ranking Report** that includes the full metrics board and all diagnostic charts.

---

#### Outputs  
- **Interactive Leaderboard**: Sortable table of scores, p-values, and missing rates.
- **Power Chart**: A visual ranking of your most important variables.
- **Audit-Ready Report**: A downloadable DOCX document for your project archives.

---

#### Why This Step Matters  
This step moves you beyond "guesswork." It provides the statistical proof needed to justify why you kept or dropped a specific feature. By removing redundant variables using the **Heatmap**, you ensure your final model is robust, interpretable, and faster to train.

### Step 4: Build & Iterate (Model Development)

#### Overview  
This step enables users to train machine learning models. 

---

#### Functionalities  

**Multi-Library Modeling**  
Build models using the world’s most popular libraries:
- **Scikit-learn**: For standard, robust machine learning.
- **XGBoost, LightGBM, CatBoost**: For state-of-the-art predictive power on tabular data.
- **Statsmodels**: For deep statistical inference (OLS/Logistic Regression) with coefficients and p-values.

**Robust Cross-Validation (CV) Engine**  
TanML ensures your model's results aren't a "fluke" by using:
- **Repeated Stratified K-Fold**: For classification, maintaining class balance across every slice.
- **Repeated K-Fold**: For regression.
- **Stability Monitoring**: Automatically flags models where validation scores and cross-validation averages diverge, identifying potential instability.

**Comprehensive Metric Dashboard**  
- **Classification**: Go beyond Accuracy with **ROC-AUC**, **MCC**, **Brier Score**, **Gini**, and the industry-standard **KS (Kolmogorov-Smirnov) statistic**.
- **Regression**: Analyze fit using **RMSE**, **R²**, **MAE**, and **Median Absolute Error**.

**Interactive Visual Diagnostics**  
- **Spaghetti Plots**: Visualize ROC and Precision-Recall curves for **every CV fold** to assess consistency.
- **Decision Support**: Use the **F1 vs. Threshold** plot to choose the optimal probability cutoff for your business case.
- **Residual Suite (Regression)**: Detect systematic errors via **Residual vs. Predicted**, **Error Histograms**, and **Q-Q Plots**.

**Statistical Deep-Dive (Statsmodels)**  
For users in regulated industries, TanML provides full transparency:
- **Coefficient Analysis**: Exact weights, p-values, and Standard Errors.
- **Odds Ratios**: Clear interpretation of feature impact on outcome probability.
- **Information Criteria**: AIC, BIC, and Log-Likelihood for comparing model complexity.

**Audit-Ready Development Report (DOCX)**  
Generate a professional, multi-page Word document capturing the configuration, metrics, and all diagnostic plots. The report includes an automated **Glossary** to help stakeholders understand the results.

---

#### Outputs  
- **Performance Summaries**: High-level dashboards of model health.
- **Stability Logs**: CV vs. Final model consistency checks.
- **Development Report**: A complete, portable record of the model's creation.

---

#### Why This Step Matters  
This step moves you from "training a model" to "validating an asset." By providing **Spaghetti Plots** for stability and **Statsmodels** for inference, TanML ensures your model is not only predictive but also statistically defensible and ready for a professional audit.


### Step 5: Validate Performance (Model Evaluation)

#### Overview  
This step evaluates the final model on unseen data and validates its readiness for real-world deployment. TanML focuses on robustness, interpretability, and risk detection to ensure the model performs reliably outside the training environment.

---

#### Functionalities  

**Data Drift Monitoring (PSI & KS)**  
TanML automatically detects "Model Decay" by comparing your Training and Testing data:
- **Population Stability Index (PSI)**: Measures shifts in the data distribution.
- **Kolmogorov-Smirnov (KS) Test**: Statistically identifies changes between datasets.
- **Alert System**: Green/Yellow/Red status flags tell you exactly which features have become unstable.

**Explainability (SHAP / XAI)**  
- **Beeswarm Plots**: A global view showing both feature importance and the directional impact (positive or negative) of feature values on predictions.
- **Bar Plots**: A clear visualization of average magnitude of impact for each feature.

**Robustness & Stress Testing**  
TanML "attacks" your model with controlled Gaussian noise to see if its predictions remain stable. This simulates real-world "messy data" and tests the fragility of your model.

**Input Space Coverage Analysis**  
Uses K-Means clustering to analyze how well your training data covers the input space and identifies **Out-of-Distribution (OOD)** samples in your test set. This highlights regions where the model has "blind spots."

**Baseline Benchmarking**  
Model performance is compared against simple baseline approaches (like predicting the mean). This ensures that your chosen model provides a meaningful, statistically significant improvement over a trivial solution.

**Audit-Ready Evaluation Report (DOCX)**  
One-click generation of the final, comprehensive **Model Evaluation Document**. This captures the metrics, Drift analysis, Stress Tests, Coverage, and SHAP plots in a portable format for risk management approval.

---

#### Outputs  
- **Generalization Dashboard**: Evaluation metrics on unseen data.
- **Explainability Suite**: SHAP Beeswarm and Bar visualizations.
- **Risk Assessment**: Drift, robustness, and coverage metrics.
- **Final Evaluation Report (DOCX)**: The definitive artifact for production sign-off.

---

#### Why This Step Matters  
This is the final check that proves your model is safe, interpretable, and robust under real-world conditions. It helps prevent the deployment of models that may fail due to data drift, lack of coverage, or instability, and provides the exact documentation required for governance approval.

---

## Advanced Configuration

### Remote Access
Running TanML on a powerful office server? Use the public flag:
```bash
tanml ui --public --port 8501
```
*Note: Ensure you are on a secure VPN/network when using `--public`.*

### Headless Mode
Running on a server without a monitor?
```bash
tanml ui --headless
```
This starts the backend without trying to open a local browser. You can then access it from your laptop using the server's IP address.
