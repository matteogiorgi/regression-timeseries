setwd("~/Documents/regression-timeseries/assignment_regression")
ames <- read.csv("AmesHousing.csv")

# First, we rename variables with spaces into more convenient names
library(dplyr)

ames_clean <- ames %>%
  rename(
    GrLivArea     = `Gr.Liv.Area`,
    FirstFlrSF    = `X1st.Flr.SF`,
    TotalBsmtSF   = `Total.Bsmt.SF`,
    LotArea       = `Lot.Area`,
    FullBath      = `Full.Bath`,
    GarageArea    = `Garage.Area`,
    GarageCars    = `Garage.Cars`,
    GarageYrBlt   = `Garage.Yr.Blt`,
    YearBuilt     = `Year.Built`,
    YearRemodAdd  = `Year.Remod.Add`,
    OverallQual   = `Overall.Qual`,
    KitchenQual   = `Kitchen.Qual`,
    ExterQual     = `Exter.Qual`,
    BsmtQual      = `Bsmt.Qual`,
    Neighborhood  = Neighborhood
  )

# Linear regression
model <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood,
  data = ames_clean
)

# Results
summary(model)





#Diagnostica
#1:near multicollinearity
# Installazione (se non già presente)
#install.packages("car")
#library(car)
### Calcolo del VIF
# vif_values <- vif(model)
# print(vif_values)

library(performance)
vif_values <- check_collinearity(model)
print(vif_values)

#multicollinearità perfetta
ames_multicollineareperfetto <- ames %>%
  rename(
    GrLivArea = `Gr.Liv.Area`,
    FirstFlrSF = `X1st.Flr.SF`,
    SecondFlrSF = `X2nd.Flr.SF`,
    LowQualFinSF = `Low.Qual.Fin.SF`,
    TotalBsmtSF = `Total.Bsmt.SF`,
    LotArea = `Lot.Area`,
    FullBath = `Full.Bath`,
    GarageArea = `Garage.Area`,
    GarageCars = `Garage.Cars`,
    GarageYrBlt = `Garage.Yr.Blt`,
    YearBuilt = `Year.Built`,
    YearRemodAdd = `Year.Remod.Add`,
    OverallQual = `Overall.Qual`,
    KitchenQual = `Kitchen.Qual`,
    ExterQual = `Exter.Qual`,
    BsmtQual = `Bsmt.Qual`,
    Neighborhood = `Neighborhood`,
    MSZoning = `MS.Zoning`,
    Utilities = `Utilities`,
  )
#creo modello con multicollinearità perfetta
modelmulticollperf <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + SecondFlrSF + LowQualFinSF +
    TotalBsmtSF + LotArea + FullBath + GarageArea + GarageCars +
    GarageYrBlt + YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood + MSZoning + Utilities,
  data = ames_multicollineareperfetto
)

summary(modelmulticollperf)
