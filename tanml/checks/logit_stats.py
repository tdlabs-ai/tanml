# tanml/checks/logit_stats.py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from contextlib import suppress

# ---------- helpers: winsorize + standardize (no sklearn) ----------
def _winsorize_df(df: pd.DataFrame, low_q=0.005, high_q=0.995) -> pd.DataFrame:
    with suppress(Exception):
        q_low = df.quantile(low_q)
        q_hi = df.quantile(high_q)
        df = df.clip(lower=q_low, upper=q_hi, axis=1)
    return df

def _standardize_df(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean()
    sd = df.std(ddof=0).replace(0, np.nan)
    z = (df - mu) / sd
    return z.fillna(0.0)

def _prep_design_matrix_df(
    X: pd.DataFrame,
    ref_columns: pd.Index | None = None,
    add_const: bool = True
) -> pd.DataFrame:
    """
    Consistent design prep for statsmodels:
      - ensure DataFrame
      - one-hot encode categoricals (drop_first=True)
      - align to ref_columns (if provided; excludes 'const')
      - coerce to numeric, replace inf, fillna(0)
      - drop zero-variance columns
      - winsorize tails, standardize
      - optionally add intercept
    """
    Xd = pd.DataFrame(X).copy()
    Xd = pd.get_dummies(Xd, drop_first=True)

    if ref_columns is not None:
        ref_wo_const = [c for c in ref_columns if c != "const"]
        Xd = Xd.reindex(columns=ref_wo_const, fill_value=0.0)

    # numeric + clean
    for c in Xd.columns:
        Xd[c] = pd.to_numeric(Xd[c], errors="coerce")
    Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # drop zero-variance columns (except 'const' which isn't present yet)
    var = Xd.var(numeric_only=True)
    drop_cols = list(var.index[var == 0])
    if drop_cols:
        Xd = Xd.drop(columns=drop_cols, errors="ignore")

    # stabilize logits: winsorize + standardize
    Xd = _winsorize_df(Xd, 0.005, 0.995)
    Xd = _standardize_df(Xd)

    if add_const:
        Xd = sm.add_constant(Xd, has_constant="add")
    return Xd

def _encode_binary_target(y) -> pd.Series:
    yb = pd.Series(y)
    uniq = pd.unique(yb.dropna())
    try:
        cats = pd.Series(uniq).astype("category").cat.categories
        if set(map(int, cats)) == {0, 1}:
            return yb.astype(int)
    except Exception:
        pass
    if len(uniq) != 2:
        raise ValueError("Logit requires a binary target with exactly two classes.")
    # map majority -> 0, minority -> 1 (stable ordering)
    counts = yb.value_counts().sort_values(ascending=False).index.tolist()
    return yb.map({counts[0]: 0, counts[1]: 1}).astype(int)

def _fit_stage1_ridge(yb: pd.Series, Xd: pd.DataFrame, alpha_grid=(1.0, 0.3, 0.1, 0.03, 0.01)):
    """
    Stage-1: ridge-regularized logit for stability. Returns best result and alpha.
    """
    best = None
    for a in alpha_grid:
        try:
            res = sm.Logit(yb, Xd).fit_regularized(alpha=a, L1_wt=0.0, maxiter=2000, disp=False)
            coef = res.params.values
            if not np.all(np.isfinite(coef)):
                continue
            score = float(getattr(res, "llf", -np.inf))
            cand = {"alpha": a, "res": res, "score": score}
            if best is None or cand["score"] > best["score"] + 1e-9 or \
               (abs(cand["score"] - best["score"]) <= 1e-9 and a < best["alpha"]):
                best = cand
        except Exception:
            continue
    if best is None:
        # stronger ridge last-ditch
        res = sm.Logit(yb, Xd).fit_regularized(alpha=3.0, L1_wt=0.0, maxiter=2000, disp=False)
        best = {"alpha": 3.0, "res": res, "score": float(getattr(res, "llf", -np.inf))}
    return best

def _fit_stage2_inference(yb: pd.Series, Xd: pd.DataFrame, start_params: np.ndarray):
    """
    Stage-2: try unpenalized MLE (for classic p-values). If it fails, fall back to GLM Binomial with robust SEs.
    Returns (model_tag, results).
    """
    # Unpenalized MLE with good starts
    try:
        mle = sm.Logit(yb, Xd).fit(start_params=start_params, method="lbfgs", maxiter=5000, disp=False)
        return ("logit_mle", mle)
    except Exception:
        pass

    # Robust GLM fallback
    glm = sm.GLM(yb, Xd, family=sm.families.Binomial())
    glm_res = glm.fit(cov_type="HC3")  # robust/sandwich SEs
    return ("glm_robust", glm_res)

def compute_logit_stats(X: pd.DataFrame, y) -> dict:
    """
    Train-only stats summary (Sections a & b):
      - robust design prep (winsorize + z-score)
      - Stage-1 ridge to stabilize
      - Stage-2 unpenalized MLE for inference (fallback to GLM robust)
      - returns summary text and a tidy coef table
    """
    # design + target
    Xd = _prep_design_matrix_df(X, ref_columns=None, add_const=True)
    yb = _encode_binary_target(y)

    # Stage-1: ridge (stability)
    stg1 = _fit_stage1_ridge(yb, Xd, alpha_grid=(1.0, 0.3, 0.1, 0.03, 0.01))
    start = stg1["res"].params.values

    # Stage-2: inference
    model_tag, res = _fit_stage2_inference(yb, Xd, start_params=start)

    # summary text
    with suppress(Exception):
        # summary2() is prettier when available
        summary_text = res.summary2().as_text()
    if 'summary_text' not in locals():
        summary_text = str(res.summary())

    # coef table
    params = res.params
    bse = res.bse
    # z or t values depending on object; label as z
    if hasattr(res, "tvalues"):
        stat_vals = res.tvalues
    elif hasattr(res, "zvalues"):
        stat_vals = res.zvalues
    else:
        stat_vals = params / bse.replace(0, np.nan)

    pvals = res.pvalues
    ci = res.conf_int(alpha=0.05)
    ci.columns = ["ci_low", "ci_high"]

    coef_df = pd.DataFrame({
        "feature": params.index,
        "coef": params.values,
        "std err": bse.values,
        "z": stat_vals.values,
        "P>|z|": pvals.values,
        "ci_low": ci["ci_low"].values,
        "ci_high": ci["ci_high"].values,
    })

    # const first
    if "const" in coef_df["feature"].values:
        coef_df = pd.concat(
            [coef_df.loc[coef_df["feature"] == "const"],
             coef_df.loc[coef_df["feature"] != "const"].sort_values("feature")],
            ignore_index=True
        )

    # round to 6 decimals so small effects don't show up as 0.000000
    for c in ["coef","std err","z","P>|z|","ci_low","ci_high"]:
        coef_df[c] = pd.to_numeric(coef_df[c], errors="coerce").round(6)

    # annotate header line to reflect fallback if used
    if model_tag == "glm_robust":
        summary_text = "Results: GLM Binomial (robust SE)\n" + summary_text
    else:
        summary_text = "Results: Logit (unpenalized MLE)\n" + summary_text

    return {
        "summary_text": summary_text,
        "coef_table_headers": ["feature","coef","std err","z","P>|z|","ci_low","ci_high"],
        "coef_table_rows": coef_df.to_dict(orient="records"),
    }
