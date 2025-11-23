import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

ames = pd.read_csv(
    "/home/matteo/Documents/regression-timeseries/assignment_regression/AmesHousing.csv"
)

# Rinomina le variabili come in R
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

# Ordina per anno di vendita
ames_ordered = ames.sort_values(by="YrSold")
break_year = 2008
ames_pre = ames_ordered[ames_ordered["YrSold"] < break_year]
ames_post = ames_ordered[ames_ordered["YrSold"] >= break_year]
formula = "SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea + FullBath + GarageArea + GarageCars + GarageYrBlt + YearBuilt + YearRemodAdd + OverallQual + KitchenQual + ExterQual + BsmtQual + Neighborhood + MSZoning + Utilities"
model_combined = ols(formula, data=ames_ordered).fit()
model_pre = ols(formula, data=ames_pre).fit()
model_post = ols(formula, data=ames_post).fit()
rss_combined = sum(model_combined.resid**2)
rss_pre = sum(model_pre.resid**2)
rss_post = sum(model_post.resid**2)

n_pre = model_pre.nobs
n_post = model_post.nobs
k = model_combined.df_model + 1  # +1 per l'intercetta

F_chow = ((rss_combined - (rss_pre + rss_post)) / k) / (
    (rss_pre + rss_post) / (n_pre + n_post - 2 * k)
)
print("Chow test statistic:", F_chow)
