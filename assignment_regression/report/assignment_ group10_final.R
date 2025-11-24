
setwd("C:/Users/enric/OneDrive/Documenti/Bank accounting")
ames <- read.csv("Amespug.csv")

# First we load the dataset and rename
#the variables into more convenient names 
install.packages("dplyr")
library(dplyr)
ames_clean <- ames %>%
  rename(
    GrLivArea = `Gr.Liv.Area`,
    FirstFlrSF = `X1st.Flr.SF`,
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
)

# linear regression
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

#Diagnostic analysis
#1:near collinearity

install.packages("car")
library(car)
###  VIF calculation
vif_values <- vif(model)
#display results
print(vif_values)

install.packages("performance")
library(performance)

vif_values <- check_collinearity(model)
print(vif_values)


#Perfect collinearity test
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
#Linear model with potential perfect collinearity
modelmulticollperf <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + SecondFlrSF + LowQualFinSF +
    TotalBsmtSF + LotArea + FullBath + GarageArea + GarageCars +
    GarageYrBlt + YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood + MSZoning + Utilities,
  data = ames_multicollineareperfetto
)

summary(modelmulticollperf)

#2: Chow-type structural stability test
install.packages("strucchange")
library(sandwich)
library(strucchange)

#Ensure ordered factors for quality variables
ames_clean$OverallQual <- ordered(ames_clean$OverallQual, levels = 1:10)

ames_clean$KitchenQual <- ordered(
  ames_clean$KitchenQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

ames_clean$ExterQual <- ordered(
  ames_clean$ExterQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

ames_clean$BsmtQual <- ordered(
  ames_clean$BsmtQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

#order data by year of sale
ames_ordered <- ames_clean[order(ames_clean$Yr.Sold), ]

#baseline model for structural break test(without Neighborhood)
modelordered <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual ,
  data = ames_ordered
)

#Break point: first observation with Yr.Sold==2008
break_point <- min(which(ames_ordered$Yr.Sold == 2008))

#Chow-type test at the given break point
library(strucchange)
chow_test <- sctest(modelordered, type = "Chow", point = break_point)
print(chow_test)

#see if intercept changes
#separate models pre and post 2008(including Neighborhood)
# Model pre-2008
model_pre2008 <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood ,
  data = subset(ames_clean, Yr.Sold <= 2008)
)

#model post-2008
model_post2008 <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood ,
  data = subset(ames_clean, Yr.Sold > 2008)
)

summary(model_pre2008)
summary(model_post2008)

#3:test reset
install.packages("lmtest")
library(lmtest)
# Trasform ordinal variables
ames_clean$OverallQual <- ordered(ames_clean$OverallQual, levels = 1:10)

ames_clean$KitchenQual <- ordered(
  ames_clean$KitchenQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

ames_clean$ExterQual <- ordered(
  ames_clean$ExterQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

ames_clean$BsmtQual <- ordered(
  ames_clean$BsmtQual,
  levels = c("Po", "Fa", "TA", "Gd", "Ex")
)

#Baseline model
model <- lm(
  SalePrice ~ GrLivArea + FirstFlrSF + TotalBsmtSF + LotArea +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood,
  data = ames_clean
)

resettest(model)

#log transformation
model_log <- lm(
  log(SalePrice) ~ log(GrLivArea) + log(FirstFlrSF) + log(TotalBsmtSF) + log(LotArea) +
    FullBath + GarageArea + GarageCars + GarageYrBlt +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood,
  data = ames_clean
)

resettest(model_log)

#log-model with interactions
model_transformato_reset <- lm(
  log(SalePrice) ~ 
    log(GrLivArea) +
    log(FirstFlrSF) +
    log(TotalBsmtSF) +
    log(LotArea) +
    FullBath + GarageArea + GarageCars +
    YearBuilt + YearRemodAdd + OverallQual +
    KitchenQual + ExterQual + BsmtQual +
    Neighborhood +
    GrLivArea:OverallQual,
  data = ames_clean
)


resettest(model_transformato_reset)


#4:heteroskedasticity and independence of residuals
#Heteroskedasticity
install.packages("whitestrap")
library(whitestrap)
white_test(model_transformato_reset)


#independence
#try to create a new dataset more suitable for DW Test
library(dplyr)
ames_period <- subset(ames_clean, Yr.Sold >= 2006 & Yr.Sold <= 2010)
pid_counts <- ames_period %>%
  group_by(PID) %>%
  summarise(n_sales = n_distinct(Yr.Sold))  

pid_multi <- pid_counts %>%
  filter(n_sales >= 2) %>%
  pull(PID)

#create new dataset
ames_multi_sales <- ames_period %>%
  filter(PID %in% pid_multi)

ames_clean %>%
  count(PID) %>%
  filter(n > 1)


ames_clean %>%
  group_by(PID) %>%
  summarise(anni_vendita = paste(unique(Yr.Sold), collapse = ", "),
            n_anni = n_distinct(Yr.Sold)) %>%
  filter(n_anni >= 2)

ames_multi_sales <- ames_clean %>%
  group_by(PID) %>%
  filter(n_distinct(Yr.Sold) >= 2) %>%
  ungroup()

#DW TEST
dwtest(model_transformato_reset)

#using HAC estimator for standard errors
install.packages("sandwich")
install.packages("lmtest")

library(sandwich)
library(lmtest)
vcov_hac <- vcovHAC(model)
coeftest(model_transformato_reset, vcov. = vcov_hac)

summary(model_transformato_reset)


#normality of residuals
install.packages("tseries")
library(tseries)
jarque.bera.test(residuals(model_transformato_reset ))
