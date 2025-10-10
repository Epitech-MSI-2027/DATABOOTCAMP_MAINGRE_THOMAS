import streamlit as st, pandas as pd, numpy as np
import plotly.express as px
from pathlib import Path
import calendar
import streamlit.components.v1 as components

# ----------------------------- CONFIG ---------------------------------
st.set_page_config(page_title="Analytics Pharma", layout="wide")
CODES = ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"]

# ----------------------------- UTILS ----------------------------------
def fmt(n, decimals=0):
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "—"
    s = f"{n:,.{decimals}f}"
    return s.replace(",", " ").replace(".", ",")

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if "datum" in df.columns:
        df["datum"] = pd.to_datetime(df["datum"], errors="coerce")
        if "Year" not in df.columns:  df["Year"]  = df["datum"].dt.year
        if "Month" not in df.columns: df["Month"] = df["datum"].dt.month
    return df

def melt_long(df: pd.DataFrame) -> pd.DataFrame:
    return df.melt(
        id_vars=[c for c in df.columns if c not in CODES],
        value_vars=CODES, var_name="Code_ATC", value_name="Ventes"
    )

# --------- Détection de tendances ----------
def _trend_metrics_one(series: pd.Series, smooth: bool, win: int):
    s = series.sort_index().dropna().astype(float)
    if s.empty:
        return None, np.nan, np.nan, np.nan, np.nan
    if smooth and len(s) >= max(3, win):
        s = s.rolling(win, min_periods=max(2, win//2)).mean().dropna()
        if s.empty:
            return None, np.nan, np.nan, np.nan, np.nan
    y = s.values
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return y, np.nan, np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    y_hat = m * x + b
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return y, m, r2, float(y.mean()), float(y.std(ddof=0))

def compute_trends(df_long: pd.DataFrame,
                   smooth: bool = True,
                   window: int = 4,
                   min_points: int = 6,
                   slope_threshold_pct: float = 0.8,
                   cv_threshold_pct: float = 15.0):
    out_rows = []
    if df_long.empty:
        return pd.DataFrame(columns=["Code_ATC","pente_%_par_periode","R2","CV_%","points","class"])

    for code, g in df_long.groupby("Code_ATC"):
        if "datum" not in g.columns:
            continue
        s = g.set_index("datum")["Ventes"].sort_index()
        s = s.groupby(level=0).sum()

        npts = int(s.dropna().shape[0])
        if npts < min_points:
            out_rows.append([code, np.nan, np.nan, np.nan, npts, "insuffisant"])
            continue

        y, slope, r2, mean_, std_ = _trend_metrics_one(s, smooth, window)
        if y is None or np.isnan(slope) or mean_ <= 0:
            out_rows.append([code, np.nan, r2, np.nan, int(len(y) if y is not None else 0), "insuffisant"])
            continue

        slope_pct = (slope / (mean_ + 1e-9)) * 100.0
        cv_pct    = (std_ / (mean_ + 1e-9)) * 100.0

        if abs(slope_pct) <= slope_threshold_pct and cv_pct <= cv_threshold_pct:
            klass = "stable"
        elif slope_pct > slope_threshold_pct:
            klass = "hausse"
        elif slope_pct < -slope_threshold_pct:
            klass = "baisse"
        else:
            klass = "stable"

        out_rows.append([code, slope_pct, r2, cv_pct, int(len(y)), klass])

    res = pd.DataFrame(out_rows, columns=["Code_ATC","pente_%_par_periode","R2","CV_%","points","class"])
    res = res.sort_values("pente_%_par_periode", ascending=False, na_position="last")
    return res

# --------- Paramètres figés tendances ----------
TREND_DEFAULTS = {
    "Daily":   {"smooth": True, "window": 4, "slope": 0.8, "cv": 15.0},
    "Weekly":  {"smooth": True, "window": 4, "slope": 0.8, "cv": 15.0},
    "Monthly": {"smooth": True, "window": 3, "slope": 1.2, "cv": 15.0},
}

def show_trends_fixed(long_df: pd.DataFrame, granularity: str, titre_suffix: str):
    params = TREND_DEFAULTS[granularity]
    st.markdown(f"###  Détecter les tendances — {titre_suffix}")
    st.caption(
        f"Lissage: **{'ON' if params['smooth'] else 'OFF'}**, "
        f"fenêtre: **{params['window']}**, "
        f"seuil pente: **{params['slope']} %/période**, "
        f"seuil stabilité (CV): **{params['cv']} %**"
    )

    trends = compute_trends(
        long_df,
        smooth=params["smooth"],
        window=int(params["window"]),
        min_points=6,
        slope_threshold_pct=float(params["slope"]),
        cv_threshold_pct=float(params["cv"])
    )

    if trends.empty:
        st.info("Pas de données suffisantes pour détecter une tendance.")
        return

    col_up, col_flat, col_down = st.columns(3)
    df_up   = trends[trends["class"]=="hausse"][["Code_ATC","pente_%_par_periode","R2","CV_%","points"]]
    df_flat = trends[trends["class"]=="stable"][["Code_ATC","pente_%_par_periode","R2","CV_%","points"]]
    df_down = trends[trends["class"]=="baisse"][["Code_ATC","pente_%_par_periode","R2","CV_%","points"]]

    with col_up:
        st.markdown("**📈 En hausse**")
        st.dataframe(df_up, use_container_width=True, height=340)
    with col_flat:
        st.markdown("**➖ Stables**")
        st.dataframe(df_flat, use_container_width=True, height=340)
    with col_down:
        st.markdown("**📉 En baisse**")
        st.dataframe(df_down.sort_values("pente_%_par_periode"), use_container_width=True, height=340)

# ----------------------------- UI BLOCKS -------------------------------
def apercu_scroll_10(df: pd.DataFrame):
    st.subheader("Aperçu des données :")
    ROW_H, HEAD_H = 36, 58
    st.dataframe(df, use_container_width=True, height=HEAD_H + ROW_H * 10)

def kpi_block(df_long: pd.DataFrame, titre: str, unite: str):
    total = df_long["Ventes"].sum()
    moy = df_long.groupby("datum")["Ventes"].sum().mean() if "datum" in df_long.columns else np.nan
    top = df_long.groupby("Code_ATC")["Ventes"].sum().sort_values(ascending=False).head(1)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{titre} — Ventes totales", fmt(total))
    c2.metric(f"Moyenne / {unite}", fmt(moy))
    if not top.empty:
        c3.metric("Code ATC principal", f"{top.index[0]} ({fmt(top.iloc[0])})")

# ---- Carte HTML fiable (components.html) avec meilleur & pire code ----
def card_html2(title: str, value: str, date_str: str,
               best_code: str, best_value: str,
               worst_code: str, worst_value: str,
               accent: str) -> str:
    if accent == "green":
        grad = "linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%)"
        ring = "#22c55e"
        worst_bg = "rgba(239,68,68,.10)"; worst_bd = "rgba(239,68,68,.25)"; worst_col = "#ef4444"
        best_bg  = "rgba(14,165,233,.08)"; best_bd  = "rgba(14,165,233,.25)"; best_col  = "#0ea5e9"
        val_bg   = "rgba(34,197,94,.08)";  val_bd   = "rgba(34,197,94,.25)";  val_col   = "#22c55e"
    else:
        grad = "linear-gradient(135deg, #f97316 0%, #ef4444 100%)"
        ring = "#ef4444"
        worst_bg = "rgba(239,68,68,.10)"; worst_bd = "rgba(239,68,68,.25)"; worst_col = "#ef4444"
        best_bg  = "rgba(14,165,233,.08)"; best_bd  = "rgba(14,165,233,.25)"; best_col  = "#0ea5e9"
        val_bg   = "rgba(34,197,94,.08)";  val_bd   = "rgba(34,197,94,.25)";  val_col   = "#22c55e"

    return f"""
<!doctype html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:transparent;">
  <div style="background:#0b0f16;border:1px solid rgba(255,255,255,.08);
              border-radius:18px;padding:18px 20px;box-shadow:0 8px 30px rgba(0,0,0,.35);
              font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Helvetica,Arial,sans-serif;">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;color:#e5e7eb;">
      <div style="width:12px;height:12px;border-radius:999px;background:{ring};
                  box-shadow:0 0 0 6px rgba(255,255,255,.06)"></div>
      <div style="font-weight:700;font-size:18px;letter-spacing:.2px">{title}</div>
    </div>

    <div style="display:flex;align-items:baseline;gap:14px;flex-wrap:wrap">
      <div style="font-size:56px;line-height:1;font-weight:800;background:{grad};
                  -webkit-background-clip:text;background-clip:text;color:transparent;">
        {value}
      </div>
    </div>

    <div style="display:flex;align-items:center;gap:10px;margin-top:14px;flex-wrap:wrap">
      <div style="font-size:14px;color:#cbd5e1;background:rgba(148,163,184,.1);
                  border:1px solid rgba(148,163,184,.25);padding:6px 10px;border-radius:999px;">
        {date_str}
      </div>

      <div style="font-size:14px;color:{best_col};background:{best_bg};
                  border:1px solid {best_bd};padding:6px 10px;border-radius:999px;">
        {best_code}
      </div>
      <div style="font-size:14px;color:{val_col};background:{val_bg};
                  border:1px solid {val_bd};padding:6px 10px;border-radius:999px;">
        {best_value}
      </div>

      <div style="font-size:14px;color:{worst_col};background:{worst_bg};
                  border:1px solid {worst_bd};padding:6px 10px;border-radius:999px;margin-left:6px;">
        pire&nbsp;: {worst_code} • {worst_value}
      </div>
    </div>
  </div>
</body></html>
    """.strip()

def top_bottom_block(df_long: pd.DataFrame, lib_periode: str):
    if df_long.empty or "datum" not in df_long.columns:
        return

    by_t = df_long.groupby("datum")["Ventes"].sum()
    if by_t.empty:
        return

    t_max, v_max = by_t.idxmax(), float(by_t.max())
    t_min, v_min = by_t.idxmin(), float(by_t.min())

    # Meilleur point : meilleur & pire codes ce jour-là
    g_max = (df_long[df_long["datum"] == t_max]
             .groupby("Code_ATC")["Ventes"].sum()
             .sort_values(ascending=False))
    max_code = g_max.index[0] if not g_max.empty else "—"
    max_code_val = float(g_max.iloc[0]) if not g_max.empty else float("nan")
    max_code_min = g_max.index[-1] if g_max.shape[0] else "—"
    max_code_min_val = float(g_max.iloc[-1]) if g_max.shape[0] else float("nan")

    # Pire point : meilleur & pire codes ce jour-là
    g_min = (df_long[df_long["datum"] == t_min]
             .groupby("Code_ATC")["Ventes"].sum()
             .sort_values(ascending=False))
    min_code = g_min.index[0] if not g_min.empty else "—"
    min_code_val = float(g_min.iloc[0]) if not g_min.empty else float("nan")
    min_code_min = g_min.index[-1] if g_min.shape[0] else "—"
    min_code_min_val = float(g_min.iloc[-1]) if g_min.shape[0] else float("nan")

    st.markdown("### Pic & creux sur la période filtrée")
    c1, c2 = st.columns(2)

    with c1:
        html = card_html2(
            title=f"Meilleure {lib_periode}",
            value=fmt(v_max),
            date_str=t_max.date().isoformat(),
            best_code=max_code, best_value=fmt(max_code_val),
            worst_code=max_code_min, worst_value=fmt(max_code_min_val),
            accent="green",
        )
        components.html(html, height=240, scrolling=False)

    with c2:
        html = card_html2(
            title=f"Pire {lib_periode}",
            value=fmt(v_min),
            date_str=t_min.date().isoformat(),
            best_code=min_code, best_value=fmt(min_code_val),
            worst_code=min_code_min, worst_value=fmt(min_code_min_val),
            accent="red",
        )
        components.html(html, height=240, scrolling=False)

def courbe_temporelle(df_long: pd.DataFrame, titre: str):
    fig = px.line(df_long, x="datum", y="Ventes", color="Code_ATC")
    fig.update_layout(title=titre, legend_title="Code ATC", yaxis_title="Ventes", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

def barres_par_code(df_long: pd.DataFrame, titre: str):
    totaux = df_long.groupby("Code_ATC")["Ventes"].sum().reset_index().sort_values("Ventes", ascending=False)
    st.plotly_chart(px.bar(totaux, x="Code_ATC", y="Ventes", title=titre), use_container_width=True)

# ----------------------------- FILTRES ---------------------------------
def filtre_daily_weekly(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if "datum" not in df.columns or df["datum"].dropna().empty:
        return df
    dmin = df["datum"].min().date()
    dmax = df["datum"].max().date()
    c1, c2 = st.columns(2)
    with c1:
        d_from = st.date_input(f"{label} — Date de début", value=dmin, min_value=dmin, max_value=dmax, key=f"{label}_from")
    with c2:
        d_to   = st.date_input(f"{label} — Date de fin",   value=dmax, min_value=dmin, max_value=dmax, key=f"{label}_to")
    if d_from > d_to:
        st.warning("⚠️ La date de début est après la date de fin. Filtre ignoré.")
        return df
    return df[(df["datum"].dt.date >= d_from) & (df["datum"].dt.date <= d_to)]

def filtre_monthly(df: pd.DataFrame):
    years = sorted(df["Year"].dropna().astype(int).unique())
    months = list(range(1, 13))
    nom_mois = {m: calendar.month_name[m] for m in months}

    c1, c2 = st.columns(2)
    with c1:
        year_opt = ["Toutes les années"] + [str(y) for y in years]
        year_sel = st.selectbox("Année", options=year_opt, index=0)
        year_val = None if year_sel == "Toutes les années" else int(year_sel)
    with c2:
        month_opt = ["Tous les mois"] + [f"{m:02d} — {nom_mois[m]}" for m in months]
        month_sel = st.selectbox("Mois", options=month_opt, index=0)
        month_val = None if month_sel == "Tous les mois" else int(month_sel.split(" — ")[0])

    out = df.copy()
    if year_val is not None:
        out = out[out["Year"] == year_val]
    if month_val is not None:
        out = out[out["Month"] == month_val]
    return out, year_val, month_val, nom_mois

# ----------------------------- APP ------------------------------------
st.title("Analytics Pharma")

defaults = {
    "Daily": "Pharma_Ventes_Daily.csv",
    "Weekly": "Pharma_Ventes_Weekly.csv",
    "Monthly": "Pharma_Ventes_Monthly.csv",
}
paths = {}
c1, c2, c3 = st.columns(3)
with c1: paths["Daily"]   = st.text_input("Chemin CSV — Journalier (Daily)",    value=defaults["Daily"])
with c2: paths["Weekly"]  = st.text_input("Chemin CSV — Hebdomadaire (Weekly)", value=defaults["Weekly"])
with c3: paths["Monthly"] = st.text_input("Chemin CSV — Mensuel (Monthly)",     value=defaults["Monthly"])

tab_daily, tab_weekly, tab_monthly = st.tabs(["Journalier","Hebdomadaire","Mensuel"])

# ============================ DAILY ====================================
with tab_daily:
    p = paths["Daily"]
    if Path(p).exists():
        df = load_csv(p)
        df_f = filtre_daily_weekly(df, "Journalier")
        apercu_scroll_10(df_f)
        long = melt_long(df_f)
        codes = st.multiselect("Sélection des codes ATC (Daily)", CODES, default=CODES, key="daily_codes")
        long = long[long["Code_ATC"].isin(codes)]

        if long.empty:
            st.info("Aucune donnée pour le filtre choisi.")
        else:
            kpi_block(long, "Journalier", "jour")
            show_trends_fixed(long, "Daily", "Daily")
            top_bottom_block(long, "journée")
            courbe_temporelle(long, "Journalier — Série temporelle")
            barres_par_code(long, "Journalier — Totaux par code ATC")
    else:
        st.info("Indique un chemin valide vers le CSV Daily.")

# ============================ WEEKLY ===================================
with tab_weekly:
    p = paths["Weekly"]
    if Path(p).exists():
        df = load_csv(p)
        df_f = filtre_daily_weekly(df, "Hebdomadaire")
        apercu_scroll_10(df_f)
        long = melt_long(df_f)
        codes = st.multiselect("Sélection des codes ATC (Weekly)", CODES, default=CODES, key="weekly_codes")
        long = long[long["Code_ATC"].isin(codes)]

        if long.empty:
            st.info("Aucune donnée pour le filtre choisi.")
        else:
            kpi_block(long, "Hebdomadaire", "semaine")
            show_trends_fixed(long, "Weekly", "Weekly")
            top_bottom_block(long, "semaine")
            courbe_temporelle(long, "Hebdomadaire — Série temporelle")
            barres_par_code(long, "Hebdomadaire — Totaux par code ATC")
    else:
        st.info("Indique un chemin valide vers le CSV Weekly.")

# ============================ MONTHLY ==================================
with tab_monthly:
    p = paths["Monthly"]
    if Path(p).exists():
        df = load_csv(p)
        df_f, year_val, month_val, nom_mois = filtre_monthly(df)
        if not df_f.empty:
            if year_val is None and month_val is None:
                st.markdown("**Filtre actif : toutes les années — tous les mois**")
            elif year_val is None and month_val is not None:
                st.markdown(f"**Filtre actif : tous {nom_mois[month_val]} (toutes années)**")
            elif year_val is not None and month_val is None:
                st.markdown(f"**Filtre actif : année {year_val} (tous mois)**")
            else:
                st.markdown(f"**Filtre actif : {nom_mois[month_val]} {year_val}**")
        apercu_scroll_10(df_f)
        long = melt_long(df_f)
        codes = st.multiselect("Sélection des codes ATC (Monthly)", CODES, default=CODES, key="monthly_codes")
        long = long[long["Code_ATC"].isin(codes)]

        if long.empty:
            st.info("Aucune donnée pour le filtre choisi.")
        else:
            kpi_block(long, "Mensuel", "mois")
            show_trends_fixed(long, "Monthly", "Monthly")
            top_bottom_block(long, "mois")
            courbe_temporelle(long, "Mensuel — Série temporelle")
            barres_par_code(long, "Mensuel — Totaux par code ATC")
    else:
        st.info("Indique un chemin valide vers le CSV Monthly.")
