!pip install -r requirements.txt

# Import data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df = pd.read_csv("superstore_final_dataset.csv", encoding="cp1250")

"""# Overview"""

# Check the number of rows and columns
df.shape

# Check initial columns types
df.dtypes

# Change the type of columns
df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
df["Ship_Date"] = pd.to_datetime(df["Ship_Date"], dayfirst=True, errors="coerce")
df['Postal_Code'] = df['Postal_Code'].astype('Int64').astype(str)

# Structure table
structure_table = pd.DataFrame({
    "Typ danych": df.dtypes.astype(str),
    "Liczba braków": df.isnull().sum().values,
    "Przykładowa wartość": [df[col].dropna().iloc[0] for col in df.columns]
})

print(structure_table)

# Count unique values
df.nunique()

# List and count unique values

print("Region:", df["Region"].unique())
print("Unique values:", df["Region"].nunique())

print("\nSegment:", df["Segment"].unique())
print("Unique values:", df["Segment"].nunique())

print("\nCategory:", df["Category"].unique())
print("Unique values:", df["Category"].nunique())

print("\nSub_Category:", df["Sub_Category"].unique())
print("Unique values:", df["Sub_Category"].nunique())

# Show max and min

print("Order_Date")
print("Min:", df["Order_Date"].min())
print("Max:", df["Order_Date"].max())

print("\nSales")
print("Min:", df["Sales"].min())
print("Max:", df["Sales"].max())

"""# Data exploration

## Seasonality and time trends
"""

# Number of orders per month

orders_per_month = df.groupby(df["Order_Date"].dt.to_period("M"))["Order_ID"].count()
ax =  orders_per_month.plot(kind="line", figsize=(12,6))
plt.ylabel("Number of orders")
plt.xlabel("")
for x, y in zip(orders_per_month.index.astype(str), orders_per_month.values):
    ax.text(x, y, str(y), ha="center", va="bottom", fontsize=8)
plt.show()

# Orders value per year

sales_per_year = df.groupby(df["Order_Date"].dt.year)["Sales"].sum()
ax = sales_per_year.plot(kind="bar", figsize=(12,6), edgecolor="black")
plt.ylabel("Sales value")
plt.xlabel("")

for i, v in enumerate(sales_per_year.values):
    ax.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

plt.show()

"""## Customers structure"""

# Share of customer types (orders)
share = df["Segment"].value_counts(normalize=True) * 100
share.plot(kind="pie", autopct="%.1f%%", figsize=(5,5), startangle=90)
plt.title("Share of customer types (by orders)")
plt.ylabel("")
plt.show()

# Share of customer types (sales value)
share_value = df.groupby("Segment")["Sales"].sum()
share_value = share_value / share_value.sum() * 100
share_value.plot(kind="pie", autopct="%.1f%%", figsize=(5,5), startangle=90)
plt.title("Share of customer types (by sales value)")
plt.ylabel("")
plt.show()

"""## Geographic structure"""

# States by sales value (top 10)
sales_per_state = df.groupby("State")["Sales"].sum().reset_index()

us_state_to_abbrev = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
}

state_coords = {
    "Alabama": (32.8, -86.6), "Alaska": (61.4, -152.3), "Arizona": (34.2, -111.6),
    "Arkansas": (34.9, -92.4), "California": (37.3, -119.7), "Colorado": (39.0, -105.5),
    "Connecticut": (41.6, -72.7), "Delaware": (39.0, -75.5), "Florida": (27.8, -81.5),
    "Georgia": (32.6, -83.4), "Hawaii": (20.8, -156.3), "Idaho": (44.1, -114.6),
    "Illinois": (40.0, -89.2), "Indiana": (40.0, -86.3), "Iowa": (42.0, -93.5),
    "Kansas": (38.5, -98.0), "Kentucky": (37.8, -85.8), "Louisiana": (30.9, -91.9),
    "Maine": (45.3, -69.2), "Maryland": (39.0, -76.7), "Massachusetts": (42.4, -71.5),
    "Michigan": (44.2, -85.6), "Minnesota": (46.7, -94.6), "Mississippi": (32.7, -89.7),
    "Missouri": (38.5, -92.5), "Montana": (46.9, -110.4), "Nebraska": (41.5, -99.8),
    "Nevada": (39.5, -116.9), "New Hampshire": (43.9, -71.6), "New Jersey": (40.1, -74.7),
    "New Mexico": (34.4, -106.1), "New York": (43.0, -75.0),
    "North Carolina": (35.5, -79.4), "North Dakota": (47.5, -100.5),
    "Ohio": (40.3, -82.8), "Oklahoma": (35.6, -97.5), "Oregon": (43.9, -120.6),
    "Pennsylvania": (41.0, -77.6), "Rhode Island": (41.7, -71.6),
    "South Carolina": (33.9, -80.9), "South Dakota": (44.4, -100.2),
    "Tennessee": (35.9, -86.3), "Texas": (31.0, -99.9), "Utah": (39.3, -111.7),
    "Vermont": (44.0, -72.7), "Virginia": (37.5, -78.8), "Washington": (47.4, -120.5),
    "West Virginia": (38.6, -80.6), "Wisconsin": (44.5, -89.5), "Wyoming": (43.0, -107.6),
    "District of Columbia": (38.9, -77.0)
}

sales_per_state["StateCode"] = sales_per_state["State"].map(us_state_to_abbrev)

top10 = sales_per_state.sort_values("Sales", ascending=False).head(10)
top10["Lat"] = top10["State"].map(lambda x: state_coords[x][0])
top10["Lon"] = top10["State"].map(lambda x: state_coords[x][1])

color_scale = ["#FFFFFF", "#9ecae1", "#3182bd"]

fig = px.choropleth(
    sales_per_state,
    locations="StateCode",
    locationmode="USA-states",
    color="Sales",
    scope="usa",
    color_continuous_scale=color_scale
)

fig.add_trace(go.Scattergeo(
    locationmode="USA-states",
    lon=top10["Lon"],
    lat=top10["Lat"],
    text=[f"{code}<br>${val:,.0f}" for code, val in zip(top10["StateCode"], top10["Sales"])],
    mode="text",
    showlegend=False
))

fig.update_layout(width=1200, height=800)
fig.show()

"""## Products structure"""

# Share of products categories (sales value)
sales_by_category = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

plt.figure(figsize=(5,5))
plt.pie(
    sales_by_category,
    labels=sales_by_category.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=plt.cm.tab20.colors
)
plt.show()

# Sales value of su-categories

top_subcat = (
    df.groupby(["Sub_Category", "Category"])["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12,6))
bars = plt.bar(
    [f"{sub} ({cat})" for sub, cat in top_subcat.index],
    top_subcat.values,
    color=plt.cm.tab20.colors[:10]
)

plt.ylabel("Sales ($)")
plt.xticks(rotation=45, ha='right')

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1000, f"${yval:,.0f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

"""# Segmentation"""

# Removing extra columns and aggregating sales value
df_small = df[["Order_Date", "Order_ID", "Customer_ID", "Sales"]]
df_grouped = df_small.groupby(["Order_Date", "Order_ID", "Customer_ID"], as_index=False)["Sales"].sum()

print(df_grouped.head())

"""## RFM"""

# R-F-M values
rfm = df_grouped.groupby("Customer_ID").agg({
    "Order_Date": lambda x: (df_grouped["Order_Date"].max() - x.max()).days,
    "Order_ID": "nunique",
    "Sales": "sum"
}).reset_index()

rfm.rename(columns={
    "Order_Date": "Recency",
    "Order_ID": "Frequency",
    "Sales": "Monetary"
}, inplace=True)

rfm["Monetary"] = rfm["Monetary"].round(0)

rfm.head()

# Medians
R_median = rfm["Recency"].median()
F_median = rfm["Frequency"].median()
M_median = rfm["Monetary"].median()

print(R_median)
print(F_median)
print(M_median)

# R-F-M labels
rfm["Recency_lvl"] = rfm["Recency"].apply(lambda x: "High" if x > R_median else "Low")
rfm["Frequency_lvl"] = rfm["Frequency"].apply(lambda x: "High" if x > F_median else "Low")
rfm["Monetary_lvl"] = rfm["Monetary"].apply(lambda x: "High" if x > M_median else "Low")

rfm

# Segments lables
def assign_segment(row):
    if row["Recency_lvl"] == "Low" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "Low":
        return "Nowy klient"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "High" and row["Monetary_lvl"] == "Low":
        return "Regularny klient niskoobrotowy"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "High" and row["Monetary_lvl"] == "High":
        return "Regularny klient wysokoobrotowy"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "High":
        return "Idealny klient"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "Low":
        return "Utracony klient"
    else:
        return "Inny"

rfm["Segment_name"] = rfm.apply(assign_segment, axis=1)
print(rfm[["Customer_ID", "Recency_lvl", "Frequency_lvl", "Monetary_lvl", "Segment_name"]].head())

# RFM chart
segment_counts = rfm["Segment_name"].value_counts(normalize=True) * 100

plt.figure(figsize=(12,6))
sns.barplot(x=segment_counts.index, y=segment_counts.values)

plt.ylabel("Share [%]")
plt.xlabel(" ")

for i, v in enumerate(segment_counts.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.xticks(rotation=45, ha="right")
plt.show()

# "Inny" category
other_customers = rfm[rfm["Segment_name"] == "Inny"]

other_combinations = other_customers.groupby(["Recency_lvl", "Frequency_lvl", "Monetary_lvl"]).size().reset_index(name="Count")

other_combinations_sorted = other_combinations.sort_values("Count", ascending=False)

print(other_combinations_sorted)

# New segments labels
def assign_segment_new(row):
    if row["Recency_lvl"] == "Low" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "Low":
        return "Nowy klient"
    elif row["Frequency_lvl"] == "High" and row["Monetary_lvl"] == "Low":
        return "Regularny klient niskoobrotowy"
    elif row["Frequency_lvl"] == "High" and row["Monetary_lvl"] == "High":
        return "Regularny klient wysokoobrotowy"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "High":
        return "Idealny klient"
    elif row["Recency_lvl"] == "High" and row["Frequency_lvl"] == "Low" and row["Monetary_lvl"] == "Low":
        return "Utracony klient"
    else:
        return "Inny"

rfm["Segment_name"] = rfm.apply(assign_segment_new, axis=1)
print(rfm[["Customer_ID", "Recency_lvl", "Frequency_lvl", "Monetary_lvl", "Segment_name"]].head())

# New RFM chart
segment_counts_new = rfm["Segment_name"].value_counts(normalize=True) * 100

plt.figure(figsize=(12,6))
sns.barplot(x=segment_counts_new.index, y=segment_counts_new.values)

plt.ylabel("Share [%]")
plt.xlabel(" ")

for i, v in enumerate(segment_counts_new.values):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)

plt.xticks(rotation=45, ha="right")
plt.show()

"""## K-means"""

# Clustering
rfm_km = df.groupby("Customer_ID").agg({
    "Order_Date": lambda x: (df["Order_Date"].max() - x.max()).days,
    "Order_ID": "nunique",
    "Sales": "sum"
}).reset_index()

rfm_km.columns = ["Customer_ID", "Recency", "Frequency", "Monetary"]

rfm_km["Monetary"] = rfm_km["Monetary"].round(0)

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_km[["Recency", "Frequency", "Monetary"]])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
rfm_km["Cluster"] = kmeans.fit_predict(rfm_scaled)

rfm_km.head()

# Centroids and number of customers
cluster_summary = (
    rfm_km.groupby("Cluster")[["Recency", "Frequency", "Monetary"]]
    .mean()
    .round(2)
    .rename(columns={
        "Recency": "Recency_centroid",
        "Frequency": "Frequency_centroid",
        "Monetary": "Monetary_centroid"
    })
)

cluster_counts = rfm_km["Cluster"].value_counts().sort_index()
cluster_percent = (cluster_counts / len(rfm_km) * 100).round(2)

cluster_summary["Clients"] = cluster_counts.values
cluster_summary["Share_%"] = cluster_percent.values

print(cluster_summary)

"""## Comparison"""

# Comparison RFM and k-means
merged = rfm[["Customer_ID", "Segment_name"]].merge(
    rfm_km[["Customer_ID", "Cluster"]],
    on="Customer_ID"
)

pd.crosstab(merged["Segment_name"], merged["Cluster"], normalize="index").round(2) * 100
