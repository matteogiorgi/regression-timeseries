import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. Load the dataset
df = pd.read_csv("arma12_seasonal_controlled_15series.csv")

# 2. Select the tenth series
series_10 = df["series_10"]

# 3. Perform the Augmented Dickey-Fuller test
# The adfuller function returns a tuple of statistics
result = adfuller(series_10)

# 4. Extract and print the results
adf_stat = result[0]
p_value = result[1]
usedlag = result[2]
nobs = result[3]
critical_values = result[4]

print(f"ADF Statistic: {adf_stat}")
print(f"p-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"   {key}: {value}")

# 5. Check stationarity (commonly using p < 0.05)
if p_value < 0.05:
    print("Result: The series is Stationary")
else:
    print("Result: The series is Non-Stationary")


# 1. Load the data
df = pd.read_csv("arma12_seasonal_controlled_15series.csv")
series_10 = df["series_10"]

# 2. Initialize the figure with 3 subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

# 3. Plot the Time Series
ax1.plot(df["t"], series_10)
ax1.set_title("Time Series 10")
ax1.set_xlabel("Time")
ax1.set_ylabel("Value")
ax1.grid(True)

# 4. Plot Autocorrelation (ACF)
# lags=40 is optional, but often good for visualization
plot_acf(series_10, ax=ax2, lags=40, title="Autocorrelation Function (ACF)")

# 5. Plot Partial Autocorrelation (PACF)
plot_pacf(series_10, ax=ax3, lags=40, title="Partial Autocorrelation Function (PACF)")

# 6. Show the plot
plt.tight_layout()
plt.show()
