import requests
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ============= PARAMETRI =============
LAT = 45.406  # Padova
LON = 11.876
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
YF_TICKER = "ELEC.F"  # proxy

# ============= 1) PREZZI DA YAHOO =============
print("Scarico dati da Yahoo Finance...")
px = yf.download(YF_TICKER, start=START_DATE, end=END_DATE)

# appiattimento eventuale
if isinstance(px.columns, pd.MultiIndex):
    px.columns = ["_".join([c for c in col if c]).strip() for col in px.columns.values]

px = px.reset_index()

# qui prendiamo qualsiasi colonna che inizi con "Close"
price_cols = [c for c in px.columns if c.lower().startswith("close")]
if not price_cols:
    raise ValueError(f"Non trovo colonne di prezzo, colonne trovate: {px.columns}")
price_col = price_cols[0]  # es. "Close_ELEC.F"

px.rename(columns={"Date": "date", price_col: "price"}, inplace=True)
px["date"] = pd.to_datetime(px["date"]).dt.date

# ============= 2) IRRADIAZIONE DA NASA POWER =============
print("Scarico dati da NASA POWER...")
nasa_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": "ALLSKY_SFC_SW_DWN",
    "start": START_DATE.replace("-", ""),
    "end": END_DATE.replace("-", ""),
    "latitude": LAT,
    "longitude": LON,
    "community": "RE",
    "format": "JSON",
}
resp = requests.get(nasa_url, params=params)
resp.raise_for_status()
data_json = resp.json()
irr_dict = data_json["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
irr_df = pd.DataFrame(list(irr_dict.items()), columns=["date", "ghi"])
irr_df["date"] = pd.to_datetime(irr_df["date"]).dt.date

# ============= 3) MERGE =============
print("Faccio il merge dei dati...")
df = pd.merge(px[["date", "price"]], irr_df, on="date", how="inner").sort_values("date")

df = df.dropna(subset=["price", "ghi"])
print("Osservazioni disponibili:", len(df))
if df.empty:
    raise ValueError("Nessuna osservazione dopo il merge – controlla parametri.")

# ============= 4) OLS =============
X = sm.add_constant(df["ghi"])
y = df["price"]
model = sm.OLS(y, X).fit()

print("\n===== PRIME RIGHE =====")
print(df.head())

print("\n===== RISULTATI OLS =====")
print(model.summary())

# ============= 5) PLOT SCATTER + LINEA ============
print("Creo il plot …")
plt.figure(figsize=(10, 6))
plt.scatter(df["ghi"], df["price"], alpha=0.6, label="Dati")

# per la linea ordiniamo per ghi
df_sorted = df.sort_values("ghi")
y_hat = model.params["const"] + model.params["ghi"] * df_sorted["ghi"]

plt.plot(df_sorted["ghi"], y_hat, label="Regressione OLS", linewidth=2)

plt.xlabel("Irradiazione giornaliera (kWh/m²/day)")
plt.ylabel("Prezzo elettricità (proxy Yahoo)")
plt.title("Prezzo vs Irradiazione solare (Italia, Padova) – OLS")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
