import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Carica il dataset AmesHousing dal percorso locale
ames = pd.read_csv(
    "/home/matteo/Documents/regression-timeseries/assignment_regression/AmesHousing.csv"
)

# Rinomina le variabili come nel file R originale,
# così da poter riutilizzare esattamente la stessa formula
ames.rename(
    columns={
        "Gr.Liv.Area": "GrLivArea",
        "X1st.Flr.SF": "FirstFlrSF",
        "Total.Bsmt.SF": "TotalBsmtSF",
        "Lot.Area": "LotArea",
        "Full.Bath": "FullBath",
        "Garage.Area": "GarageArea",
        "Garage.Cars": "GarageCars",
        "Garage.Yr.Blt": "GarageYrBlt",
        "Year.Built": "YearBuilt",
        "Year.Remod.Add": "YearRemodAdd",
        "Overall.Qual": "OverallQual",
        "Kitchen.Qual": "KitchenQual",
        "Exter.Qual": "ExterQual",
        "Bsmt.Qual": "BsmtQual",
        "MS.Zoning": "MSZoning",
        "Utilities": "Utilities",
        "Yr.Sold": "YrSold",
    },
    inplace=True,
)

# Ordina le osservazioni in base all'anno di vendita
ames_ordered = ames.sort_values(by="YrSold")

# Definisce l'anno di potenziale break strutturale (crisi 2008)
break_year = 2008

# Sotto-campione prima del 2008
ames_pre = ames_ordered[ames_ordered["YrSold"] < break_year]

# Sotto-campione dal 2008 in poi
ames_post = ames_ordered[ames_ordered["YrSold"] >= break_year]

# Specifica del modello: prezzo di vendita in funzione delle caratteristiche dell'immobile
formula = (
    "SalePrice ~ "
    "GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea + FullBath + "
    "GarageArea + GarageCars + GarageYrBlt + YearBuilt + YearRemodAdd + "
    "OverallQual + KitchenQual + ExterQual + BsmtQual + Neighborhood + "
    "MSZoning + Utilities"
)

# Stima del modello unico su tutto il campione
model_combined = ols(formula, data=ames_ordered).fit()

# Stima del modello solo sul periodo pre‑crisi
model_pre = ols(formula, data=ames_pre).fit()

# Stima del modello solo sul periodo post‑crisi
model_post = ols(formula, data=ames_post).fit()

# Somma dei quadrati dei residui (Residual Sum of Squares) per ciascun modello
rss_combined = sum(model_combined.resid**2)
rss_pre = sum(model_pre.resid**2)
rss_post = sum(model_post.resid**2)

# Numero di osservazioni nei due sotto‑campioni
n_pre = model_pre.nobs
n_post = model_post.nobs

# Numero di parametri stimati (k): df_model + 1 per includere l'intercetta
k = model_combined.df_model + 1

# Statistica F del test di Chow:
# H0: non c'è break strutturale (stessi coefficienti pre e post)
F_chow = ((rss_combined - (rss_pre + rss_post)) / k) / (
    (rss_pre + rss_post) / (n_pre + n_post - 2 * k)
)

# Stampa a video il valore della statistica di Chow
print("Chow test statistic:", F_chow)
