import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Marketing Campaign Dashboard", layout="wide")

@st.cache_data
def load_data(path: str):
    df = pd.read_csv(path, sep=";")
    # Feature engineering
    if "Dt_Customer" in df.columns:
        df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce")
    if "Year_Birth" in df.columns:
        df["Age"] = 2015 - df["Year_Birth"]
    spend_cols = [c for c in df.columns if c.startswith("Mnt")]
    buy_cols   = [c for c in df.columns if c.startswith("Num") and "Visits" not in c]
    if spend_cols:
        df["TotalSpent"] = df[spend_cols].sum(axis=1)
    if buy_cols:
        df["TotalPurchases"] = df[buy_cols].sum(axis=1)
    if "Kidhome" in df.columns and "Teenhome" in df.columns:
        df["Children"] = df["Kidhome"] + df["Teenhome"]
    return df, spend_cols, buy_cols

# Sidebar
st.sidebar.title("⚙️ Filtres")
data_path = st.sidebar.text_input("Chemin du CSV", value="Camp_Market.csv")
df, spend_cols, buy_cols = load_data(data_path)

# Filters
if "Age" in df.columns:
    age_range = st.sidebar.slider("Âge", int(df.Age.min()), int(df.Age.max()),
                                  (int(df.Age.min()), int(df.Age.max())))
else:
    age_range = None

if "Income" in df.columns:
    income_range = st.sidebar.slider("Revenu", int(df.Income.min()), int(df.Income.max()),
                                     (int(df.Income.min()), int(df.Income.max())))
else:
    income_range = None

marital_sel = st.sidebar.multiselect("Statut marital",
                                     sorted(df["Marital_Status"].dropna().unique())) \
             if "Marital_Status" in df.columns else []
edu_sel = st.sidebar.multiselect("Niveau d'études",
                                 sorted(df["Education"].dropna().unique())) \
          if "Education" in df.columns else []

children_sel = None
if "Children" in df.columns:
    children_sel = st.sidebar.slider("Enfants (Kid+Teen)", int(df.Children.min()),
                                     int(df.Children.max()),
                                     (int(df.Children.min()), int(df.Children.max())))

recency_sel = None
if "Recency" in df.columns:
    recency_sel = st.sidebar.slider("Récence (jours)", int(df.Recency.min()),
                                    int(df.Recency.max()),
                                    (int(df.Recency.min()), int(df.Recency.max())))

flt = pd.Series(True, index=df.index)
if age_range and "Age" in df.columns:
    flt &= df["Age"].between(age_range[0], age_range[1])
if income_range and "Income" in df.columns:
    flt &= df["Income"].between(income_range[0], income_range[1])
if marital_sel and "Marital_Status" in df.columns:
    flt &= df["Marital_Status"].isin(marital_sel)
if edu_sel and "Education" in df.columns:
    flt &= df["Education"].isin(edu_sel)
if children_sel and "Children" in df.columns:
    flt &= df["Children"].between(children_sel[0], children_sel[1])
if recency_sel and "Recency" in df.columns:
    flt &= df["Recency"].between(recency_sel[0], recency_sel[1])

df_f = df[flt].copy()

# KPIs
st.title("📊 Marketing Campaign Dashboard")
st.caption(f"Dataset : **{data_path}** | Lignes filtrées : **{len(df_f)} / {len(df)}**")

k1, k2, k3, k4 = st.columns(4)
if "Response" in df_f.columns:
    k1.metric("Taux de réponse", f"{df_f['Response'].mean():.1%}")
else:
    k1.metric("Lignes", f"{len(df_f)}")
if "TotalSpent" in df_f.columns:
    k2.metric("Dépense moyenne", f"{df_f['TotalSpent'].mean():,.0f}".replace(",", " "))
if "Income" in df_f.columns:
    k3.metric("Revenu moyen", f"{df_f['Income'].mean():,.0f}".replace(",", " "))
if "NumWebVisitsMonth" in df_f.columns:
    k4.metric("Visites web / mois", f"{df_f['NumWebVisitsMonth'].mean():.2f}")

st.markdown("---")

# Charts
c1, c2 = st.columns(2)
if spend_cols:
    spend_long = df_f.melt(value_vars=spend_cols, var_name="Catégorie", value_name="Montant")
    spend_sum = spend_long.groupby("Catégorie", as_index=False)["Montant"].sum().sort_values("Montant", ascending=False)
    c1.plotly_chart(px.bar(spend_sum, x="Catégorie", y="Montant", title="Dépenses par catégorie (somme)"),
                    use_container_width=True)
if buy_cols:
    buy_long = df_f.melt(value_vars=buy_cols, var_name="Canal", value_name="Achats")
    buy_sum = buy_long.groupby("Canal", as_index=False)["Achats"].sum().sort_values("Achats", ascending=False)
    c2.plotly_chart(px.bar(buy_sum, x="Canal", y="Achats", title="Achats par canal (somme)"),
                    use_container_width=True)

c3, c4 = st.columns(2)
if "Response" in df_f.columns and "Age" in df_f.columns:
    bins = [0,25,35,45,55,65,120]
    df_f["_age_bin"] = pd.cut(df_f["Age"], bins=bins, right=False)
    resp_age = df_f.groupby("_age_bin")["Response"].mean().reset_index()
    c3.plotly_chart(px.line(resp_age, x="_age_bin", y="Response", markers=True,
                            title="Taux de réponse par tranche d'âge"), use_container_width=True)
if "Response" in df_f.columns and "Income" in df_f.columns:
    inc_q = pd.qcut(df_f["Income"], q=5, duplicates="drop")
    resp_inc = df_f.groupby(inc_q)["Response"].mean().reset_index()
    resp_inc.rename(columns={"Income":"Quantile revenu"}, inplace=True)
    c4.plotly_chart(px.bar(resp_inc, x="Quantile revenu", y="Response",
                            title="Taux de réponse par quantile de revenu"), use_container_width=True)

c5, c6 = st.columns(2)
if "Response" in df_f.columns and "Education" in df_f.columns:
    edu_resp = df_f.groupby("Education")["Response"].mean().reset_index().sort_values("Response", ascending=False)
    c5.plotly_chart(px.bar(edu_resp, x="Education", y="Response",
                            title="Taux de réponse par niveau d'études"), use_container_width=True)
if "Response" in df_f.columns and "Marital_Status" in df_f.columns:
    mar_resp = df_f.groupby("Marital_Status")["Response"].mean().reset_index().sort_values("Response", ascending=False)
    c6.plotly_chart(px.bar(mar_resp, x="Marital_Status", y="Response",
                            title="Taux de réponse par statut marital"), use_container_width=True)

st.markdown("### 🔎 Détail (top 500 lignes filtrées)")
st.dataframe(df_f.head(500))
