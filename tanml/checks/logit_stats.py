# tanml/checks/logit_stats.py

import statsmodels.api as sm

class LogisticStatsCheck:
    def __init__(self, model, X, y, config):
        self.model = model
        self.X = X
        self.y = y
        self.config = config or {}

    def run(self):
        try:
            # 1) Add constant and fit the statsmodels Logit
            Xc = sm.add_constant(self.X, has_constant='add')
            res = sm.Logit(self.y, Xc).fit(disp=False)

            # 2) Extract coefficient table
            coef    = res.params
            stderr  = res.bse
            zscore  = coef / stderr
            pvals   = res.pvalues

            table = []
            for feat in coef.index:
                label = "Intercept" if feat.lower() == "const" else feat
                table.append({
                    "feature":     label,
                    "coefficient": float(coef[feat]),
                    "std_error":   float(stderr[feat]),
                    "z_score":     float(zscore[feat]),
                    "p_value":     float(pvals[feat]),
                })

            # 3) Fit statistics
            fit = {
                "log_lik":   float(res.llf),
                "aic":       float(res.aic),
                "bic":       float(res.bic),
                "pseudo_r2": float(res.prsquared),
            }

            # 4) Full summary text
            summary = res.summary().as_text()

            return {
                "table":   table,
                "fit":     fit,
                "summary": summary,
                "object":  res
            }

        except Exception as e:
            return {
                "table":   [],
                "fit":     {},
                "summary": f"LogisticStatsCheck failed: {e}",
                "object":  None
            }
