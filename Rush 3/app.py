
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Modeling
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)
# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt

st.set_page_config(page_title="Credit Risk — Logistic Regression (with CV/SMOTE/GridSearch)", layout="wide")

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.replace("\ufeff", "") for c in df.columns]
    return df

def describe_target(df, target):
    counts = df[target].value_counts().sort_index()
    st.metric("Observations", len(df))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Clients 'mauvais' (1)", int(counts.get(1, 0)))
    with col2:
        st.metric("Clients 'bons' (0)", int(counts.get(0, 0)))
    st.bar_chart(counts, height=160)

def make_filters(df: pd.DataFrame):
    st.sidebar.header("Filtres")
    month = st.sidebar.multiselect("Mois", sorted(df["month"].unique().tolist()))
    region = st.sidebar.multiselect("Région", sorted(df["region"].unique().tolist()))
    prod = st.sidebar.multiselect("Type de produit", sorted(df["product_type"].unique().tolist()))
    sex = st.sidebar.multiselect("Sexe", sorted(df["sex"].unique().tolist()))
    only_clients = st.sidebar.selectbox("Is client (filtre)", ["Tous", "Clients existants (1)", "Nouveaux (0)"])

    df2 = df.copy()
    if month: df2 = df2[df2["month"].isin(month)]
    if region: df2 = df2[df2["region"].isin(region)]
    if prod: df2 = df2[df2["product_type"].isin(prod)]
    if sex: df2 = df2[df2["sex"].isin(sex)]
    if only_clients != "Tous":
        df2 = df2[df2["is_client"] == (1 if "1" in only_clients else 0)]
    st.sidebar.write(f"**Jeu filtré**: {len(df2)} lignes")
    return df2

def eda(df):
    st.subheader("Aperçu")
    st.dataframe(df.head(20))
    st.subheader("Statistiques descriptives (numériques)")
    st.dataframe(df.select_dtypes(include=[np.number]).describe().T)

    st.subheader("Distribution par variables clés")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("Montant du crédit")
        fig, ax = plt.subplots(); ax.hist(df['credit_amount'].dropna(), bins=30); st.pyplot(fig, use_container_width=True)
    with c2:
        st.write("Âge")
        fig, ax = plt.subplots(); ax.hist(df['age'].dropna(), bins=30); st.pyplot(fig, use_container_width=True)
    with c3:
        st.write("Revenu")
        fig, ax = plt.subplots(); ax.hist(df['income'].dropna(), bins=30); st.pyplot(fig, use_container_width=True)

    st.subheader("Répartition catégorielles")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.bar_chart(df["sex"].value_counts())
    with c2: st.bar_chart(df["education"].value_counts())
    with c3: st.bar_chart(df["product_type"].value_counts())
    with c4: st.bar_chart(df["family_status"].value_counts())

    # --- Nuage de points intégré ---
    st.subheader("Nuage de points — Revenu vs Montant du crédit")
    limit_outliers = st.checkbox("Limiter aux quantiles 1%–99% (réduit l'effet des valeurs extrêmes)", value=True, key="scatter_outliers")
    dff = df.copy()
    if limit_outliers:
        q = dff[["income","credit_amount"]].quantile([0.01, 0.99])
        dff = dff[
            dff["income"].between(q.loc[0.01, "income"], q.loc[0.99, "income"]) &
            dff["credit_amount"].between(q.loc[0.01, "credit_amount"], q.loc[0.99, "credit_amount"])
        ]
    fig, ax = plt.subplots()
    ax.scatter(dff["income"], dff["credit_amount"], alpha=0.5)
    ax.set_xlabel("Income")
    ax.set_ylabel("Credit Amount")
    st.pyplot(fig, use_container_width=True)

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols

def train_model(df, target, test_size, C, class_weight, max_iter, threshold):
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    clf = LogisticRegression(C=C, class_weight=class_weight, max_iter=max_iter, solver="lbfgs")
    pipe = Pipeline(steps=[("pre", preprocessor), ("lr", clf)])

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    pipe.fit(Xtrain, ytrain)

    yproba = pipe.predict_proba(Xtest)[:, 1]
    ypred = (yproba >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(ytest, ypred),
        "Precision": precision_score(ytest, ypred, zero_division=0),
        "Recall": recall_score(ytest, ypred, zero_division=0),
        "F1": f1_score(ytest, ypred, zero_division=0),
        "ROC-AUC": roc_auc_score(ytest, yproba),
    }
    cm = confusion_matrix(ytest, ypred)

    return pipe, (Xtest, ytest, ypred, yproba), metrics, cm

def plot_roc_pr(ytest, yproba):
    st.write("**Courbes ROC & PR**")
    fig1, ax1 = plt.subplots()
    RocCurveDisplay.from_predictions(ytest, yproba, ax=ax1)
    st.pyplot(fig1, use_container_width=True)

    fig2, ax2 = plt.subplots()
    PrecisionRecallDisplay.from_predictions(ytest, yproba, ax=ax2)
    st.pyplot(fig2, use_container_width=True)

def export_predictions(Xtest, ytest, ypred, yproba):
    out = Xtest.copy()
    out["y_true"] = ytest.values
    out["y_pred"] = ypred
    out["proba_bad_client"] = yproba
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Télécharger les prédictions (CSV)", csv, "predictions.csv", "text/csv")

# --- App ---
st.title("Crédit — Régression logistique (Rush 3) + CV / SMOTE / GridSearch")
data_path = st.sidebar.text_input("Chemin du fichier CSV", "./Credit Data_Fichier Clients(in).csv")
df = load_data(data_path)

target_col = "bad_client_target"
if target_col not in df.columns:
    st.error(f"Colonne cible '{target_col}' introuvable.")
    st.stop()

df_f = make_filters(df)

tab1, tab2, tab3, tab4 = st.tabs(["🔎 EDA", "🧠 Modélisation", "🧮 Prédiction individuelle", "🔧 Validation avancée"])

with tab1:
    describe_target(df_f, target_col)
    eda(df_f)

with tab2:
    st.subheader("Paramètres du modèle")
    colA, colB, colC = st.columns(3)
    with colA:
        test_size = st.slider("Taille test", 0.1, 0.5, 0.2, 0.05)
        threshold = st.slider("Seuil de classification", 0.1, 0.9, 0.5, 0.01)
    with colB:
        C = st.number_input("C (inverse de la régularisation)", min_value=0.001, max_value=10.0, value=1.0, step=0.05)
        max_iter = st.number_input("max_iter", min_value=100, max_value=2000, value=500, step=50)
    with colC:
        cw = st.selectbox("Class weight", ["None", "balanced"])
        class_weight = None if cw == "None" else "balanced"

    pipe, eval_data, metrics, cm = train_model(
        df_f, target_col, test_size, C, class_weight, int(max_iter), threshold
    )

    Xtest, ytest, ypred, yproba = eval_data

    st.subheader("Scores")
    st.write(pd.DataFrame([metrics]).style.format({k: "{:.3f}" for k in metrics.keys()}))

    st.subheader("Matrice de confusion")
    cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
    st.dataframe(cm_df)

    plot_roc_pr(ytest, yproba)

    st.subheader("Coefficients du modèle")
    pre = pipe.named_steps["pre"]
    num_features = pre.transformers_[0][2]
    cat_encoder = pre.transformers_[1][1]
    cat_features = cat_encoder.get_feature_names_out(pre.transformers_[1][2])

    feature_names = np.concatenate([num_features, cat_features])
    coefs = pipe.named_steps["lr"].coef_.ravel()

    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("coef", ascending=False)
    st.dataframe(coef_df)

    export_predictions(Xtest, ytest, ypred, yproba)

    # --- Analyse d'erreurs (ajout notebook collègue) ---
    st.subheader("Analyse d'erreurs (échantillon)")
    errors = Xtest.copy()
    errors["y_true"] = ytest.values
    errors["y_pred"] = ypred
    errors["proba_bad"] = yproba
    err_view = errors[errors["y_true"] != errors["y_pred"]]
    st.write(f"{len(err_view)} erreurs sur {len(errors)} prédictions test.")
    cols_to_show = ["age","income","credit_amount","product_type","y_true","y_pred","proba_bad"]
    available = [c for c in cols_to_show if c in err_view.columns]
    st.dataframe(err_view[available].sort_values("proba_bad", ascending=False).head(25))

    buf = BytesIO(); pd.to_pickle(pipe, buf); buf.seek(0)
    st.download_button("📦 Télécharger le modèle (pickle)", data=buf, file_name="logreg_pipeline.pkl")

with tab3:
    st.subheader("Saisir un client pour prédire")
    X = df_f.drop(columns=[target_col])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    inputs = {}
    nc1, nc2, nc3 = st.columns(3)
    for i, col in enumerate(num_cols):
        col_container = [nc1, nc2, nc3][i % 3]
        val = float(col_container.number_input(col, value=float(X[col].median())))
        inputs[col] = val

    cc1, cc2 = st.columns(2)
    for i, col in enumerate(cat_cols):
        col_container = [cc1, cc2][i % 2]
        val = col_container.selectbox(col, sorted(X[col].dropna().unique().tolist()))
        inputs[col] = val

    if st.button("Prédire"):
        xdf = pd.DataFrame([inputs])
        proba = pipe.predict_proba(xdf)[0, 1]
        pred = int(proba >= 0.5)
        st.metric("Proba 'mauvais client'", f"{proba:.3f}")
        st.metric("Prédiction", pred)

with tab4:
    st.subheader("Validation avancée (CV / SMOTE / GridSearch)")
    colX, colY, colZ = st.columns(3)
    with colX:
        use_smote = st.checkbox("Activer SMOTE (train uniquement)", value=False)
    with colY:
        n_splits = st.slider("K-fold (CV)", 3, 10, 5, 1)
    with colZ:
        do_grid = st.checkbox("GridSearch (C)", value=False)

    y_cv = df_f[target_col].astype(int)
    X_cv = df_f.drop(columns=[target_col])

    preprocessor_cv, _, _ = build_preprocessor(X_cv)

    if use_smote:
        base = ImbPipeline(steps=[
            ("pre", preprocessor_cv),
            ("smote", SMOTE(random_state=42)),
            ("lr", LogisticRegression(max_iter=800, class_weight="balanced", solver="lbfgs")),
        ])
    else:
        base = Pipeline(steps=[
            ("pre", preprocessor_cv),
            ("lr", LogisticRegression(max_iter=800, class_weight="balanced", solver="lbfgs")),
        ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    if do_grid:
        grid = {"lr__C": [0.1, 0.5, 1.0, 2.0, 5.0]}
        with st.spinner("GridSearch en cours ..."):
            gs = GridSearchCV(estimator=base, param_grid=grid, scoring="roc_auc", cv=skf, n_jobs=-1, refit=True)
            gs.fit(X_cv, y_cv)
        st.write("**Meilleurs paramètres**:", gs.best_params_)
        st.write("**AUC-ROC (CV, best)**: {:.3f}".format(gs.best_score_))
    else:
        with st.spinner("Cross-validation en cours ..."):
            scores = cross_val_score(base, X_cv, y_cv, scoring="roc_auc", cv=skf, n_jobs=-1)
        st.write("AUC-ROC par fold:", [round(s,3) for s in scores])
        st.write("**AUC-ROC moyen (CV)**: {:.3f} ± {:.3f}".format(scores.mean(), scores.std()))
