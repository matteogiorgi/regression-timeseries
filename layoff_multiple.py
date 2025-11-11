import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)
n = 100

# ============= 1) COSTRUISCO I DATI =============
ai_adoption = np.random.uniform(0, 100, n)
revenue_growth = np.random.normal(5, 10, n)
merger = np.random.binomial(1, 0.2, n)
market_cap = np.random.lognormal(mean=2, sigma=0.7, size=n)
taxes = np.random.normal(50, 15, n)
fdi = np.random.poisson(3, n)
sector = np.random.choice(["Tech", "Finance", "Manufacturing"], size=n)

layoffs = (
    20
    + 1.4 * ai_adoption
    - 0.3 * revenue_growth
    + 10 * merger
    + 0.8 * market_cap
    + 0.2 * taxes
    + 3 * fdi
    + np.random.normal(0, 15, n)
)

df = pd.DataFrame(
    {
        "Layoffs": layoffs,
        "AI_Adoption": ai_adoption,
        "RevenueGrowth": revenue_growth,
        "Merger": merger,
        "MarketCap": market_cap,
        "Taxes": taxes,
        "FDI": fdi,
        "Sector": sector,
    }
)

# ============= 2) DUMMY PER IL SETTORE =============
df = pd.get_dummies(df, columns=["Sector"], drop_first=True)

# ============= 3) MODELLO OLS COMPLETO =============
y = df["Layoffs"]
X = df.drop(columns=["Layoffs"])
X = X.astype(float)
X = sm.add_constant(X)
model_full = sm.OLS(y, X).fit()
print(model_full.summary())

# ======================================================
# 4) PLOT PER IL MODELLO COMPLETO
# ======================================================
y_pred_full = model_full.fittedvalues
resid_full = model_full.resid

# a) Observed vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y, y_pred_full, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", lw=2)
plt.xlabel("Observed layoffs")
plt.ylabel("Predicted layoffs")
plt.title("Full model: observed vs predicted")
plt.tight_layout()
plt.show()

# b) Residual plot
plt.figure(figsize=(6, 5))
plt.scatter(y_pred_full, resid_full, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Full model: residuals vs fitted")
plt.tight_layout()
plt.show()

# ======================================================
# 5) MODELLO RIDOTTO (solo variabili significative)
# ======================================================
X_reduced = df[["AI_Adoption", "Merger", "MarketCap", "Taxes", "FDI"]]
X_reduced = sm.add_constant(X_reduced)
model_reduced = sm.OLS(y, X_reduced).fit()
print(model_reduced.summary())

# ======================================================
# 6) PLOT PER IL MODELLO RIDOTTO
# ======================================================
y_pred_red = model_reduced.fittedvalues
resid_red = model_reduced.resid

# a) Observed vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y, y_pred_red, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", lw=2)
plt.xlabel("Observed layoffs")
plt.ylabel("Predicted layoffs")
plt.title("Reduced model: observed vs predicted")
plt.tight_layout()
plt.show()

# b) Residual plot
plt.figure(figsize=(6, 5))
plt.scatter(y_pred_red, resid_red, alpha=0.7)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Reduced model: residuals vs fitted")
plt.tight_layout()
plt.show()

# ======================================================
# 7) CONFRONTO METRICHE
# ======================================================
print(
    "\n================ COMPARISON BETWEEN FULL AND REDUCED MODELS ================\n"
)
print("AIC (full):   ", model_full.aic)
print("AIC (reduced):", model_reduced.aic)
print("BIC (full):   ", model_full.bic)
print("BIC (reduced):", model_reduced.bic)
print("Adj. R² (full):   ", model_full.rsquared_adj)
print("Adj. R² (reduced):", model_reduced.rsquared_adj)

# ======================================================
# 8) F-TEST PER MODELLI ANNIDATI
# ======================================================
from statsmodels.stats.anova import anova_lm

anova_results = anova_lm(model_reduced, model_full)
print("\n================ F-TEST BETWEEN MODELS ================\n")
print(anova_results)
