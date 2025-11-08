import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt


#chargements de df_final après pré-processing

#df_final : données des top joueurs entre 2010 et 2020
@st.cache_data
def load_data():
    df = pd.read_csv('data/df_final.csv')
    df = df.drop(["dist", "Tm", "AST", "TRB", "FT%", "WS", "PER", "VORP", "X Location", "Y Location"], axis=1)
    df["Game Date"] = pd.to_datetime(df["Game Date"], format="%Y-%m-%d", errors="coerce")
    return df

df = load_data()



@st.cache_data
def load_roc_data():
    return joblib.load('data/roc_data.pkl')


roc_data = load_roc_data()

import pandas as pd
import streamlit as st


# --- Titre et sidebar ---
st.title("Projet des tirs de joueurs NBA")
st.sidebar.title("Sommaire")
pages = ["Jeu de données", "DataVizualization", "Modélisation", "Interprétation des résultats de la Régression Logistique"]
page = st.sidebar.radio("Aller vers", pages)

if page == "Jeu de données":
    st.header("Présentation du jeu de données")

    st.write("Les données concernent le top 25 des meilleurs joueurs NBA du 21ème siècle (selon ESPN), entre les saisons 2010 et 2020.")

    st.subheader("Aperçu des données")
    st.dataframe(df.head()) 

    st.subheader("Dimensions du jeu de données")
    st.write(df.shape)

    st.markdown("---")

    st.write("""
    Voici la liste des principales variables du jeu de données
    """)

    st.markdown("""
    **Player** : (string) nom du joueur  
    **Game Date** : (date) date du match  
    **Shot Made Flag** : (int) 1 si le tir est réussi, 0 sinon  
    **Shot Distance** : (int) distance du tir en feet  
    **Period** : (int) période du match (1 à 4, voire plus pour prolongation)  
    **Action Type** : (string) type d’action du tir (layup, dunk, hook, jump shot et 3 point)  
    **is_home** : (int) 1 si le joueur évolue à domicile, 0 sinon  
    **angle** : (float) angle du tir par rapport à l’axe du panier (en radian)
    **Shot Zone Area** : (string) zone du terrain depuis laquelle le tir est tenté  
    **Shot Type** : (string) type de tir (2PT Field Goal ou 3PT Field Goal)  
    **Season Type** : (string) type de saison (Regular Season ou Playoffs)  
    **Year** : (int) année de la saison  
    **TIME_LEFT** : (int) temps restant dans la période (en secondes)  
    **Pos** : (string) poste du joueur (PG, SG, SF, PF, C)  
    **Age** : (float) âge du joueur au moment du match  
    **G** : (float) nombre total de matchs joués cette saison  
    **MP** : (float) minutes jouées par match  
    **PTS** : (float) points moyens par match  
    **FG%** : (float) pourcentage de tirs réussis  
    **3P%** : (float) pourcentage de tirs à 3 points réussis  
    """)



if page == "Modélisation":
    st.write("### Modélisation")


    df_results = pd.read_csv('data/model_results.csv')

    model_name = st.selectbox("Choisissez le modèle :", df_results["model"].unique())

    df_model = df_results[df_results["model"] == model_name]


   #hyperparamètres
    hyperparams = {}

    if model_name == "Logistic Regression":
        for col in ["C", "solver", "penalty"]:
            if col in df_model.columns:
                hyperparams[col] = st.selectbox(col, sorted(df_model[col].dropna().unique()))
        
        # Filtrer selon la sélection
        df_model_filtered = df_model
        for col, val in hyperparams.items():
            df_model_filtered = df_model_filtered[df_model_filtered[col] == val]

    elif model_name == "XGBoost":
        # Liste de tous les hyperparamètres disponibles dans df_results
        xgb_cols = ["n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"]
        for col in xgb_cols:
            if col in df_model.columns:
                hyperparams[col] = st.selectbox(col, sorted(df_model[col].dropna().unique()))
        
        # Filtrer selon la sélection
        df_model_filtered = df_model
        for col, val in hyperparams.items():
            df_model_filtered = df_model_filtered[df_model_filtered[col] == val]

    # Choix de la métrique ---
    metric = st.radio("Quelle métrique souhaitez-vous afficher ?", ("AUC", "Brier", "LogLoss"))


    # Affichage des métriques ---
    if not df_model_filtered.empty:
        st.write("#### Résultats")
        st.write(f"{metric} : **{df_model_filtered[metric].values[0]:.4f}**")

        # Affichage de la courbe ROC ---
        import plotly.graph_objects as go
        from sklearn.metrics import roc_curve, auc

        # Filtrer les données ROC correspondantes
        filtered_roc = [
            r for r in roc_data
            if r["model"] == model_name and all(r.get(k) == v for k, v in hyperparams.items())
        ]

        if filtered_roc:
            y_true = filtered_roc[0]["y_true"]
            y_proba = filtered_roc[0]["y_proba"]

            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.4f})'
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines', name='Aléatoire', line=dict(dash='dash')
            ))

            fig.update_layout(
                title="Courbe ROC",
                xaxis_title="Taux de faux positifs (FPR)",
                yaxis_title="Taux de vrais positifs (TPR)",
                width=700, height=500,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Données ROC non trouvées pour cette combinaison.")
    else:
        st.warning("Aucune combinaison correspondante trouvée dans les résultats.")



if page == "DataVizualization":
    st.header("Visualisation du jeu de données")

    st.markdown("""
    Cette page présente quelques visualisations pour mieux comprendre 
    notre jeu de données.
    """)

    # variable cible
    st.subheader("Répartition des tirs réussis vs ratés")

    prop = (df["Shot Made Flag"].sum() / df["Shot Made Flag"].count()).round(3)
    st.write(f"**Proportion de tirs réussis :** {prop}")
    st.write("Pas de  classe majoritaire observée.")

    @st.cache_data
    def plot_shot_success_count(df):
        fig, ax = plt.subplots()
        sns.countplot(x="Shot Made Flag", data=df, ax=ax)
        ax.set_title("Répartition des tirs réussis vs ratés")
        return fig

    fig_count = plot_shot_success_count(df)
    st.pyplot(fig_count)
    
    st.markdown("---------------")


    #distance et variable cible
    st.subheader("Vue d’ensemble : distance et réussite")

    @st.cache_data
    def plot_distance_box(df):
        fig, ax = plt.subplots()
        sns.boxplot(x="Shot Made Flag", y="Shot Distance", data=df, ax=ax)
        ax.set_title("Distance moyenne selon réussite du tir")
        return fig


    @st.cache_data
    def plot_shot_distance_distribution(df):
        fig, ax = plt.subplots()
        sns.kdeplot(df[df["Shot Made Flag"]==1]["Shot Distance"], label="Réussi", ax=ax)
        sns.kdeplot(df[df["Shot Made Flag"]==0]["Shot Distance"], label="Raté", ax=ax)
        ax.legend()
        ax.set_title("Distribution de la distance selon le résultat du tir")
        return fig

    col1, col2 = st.columns(2)

    with col1:
        fig_box = plot_distance_box(df)
        st.pyplot(fig_box)    

    with col2:
        fig_distance = plot_shot_distance_distribution(df)
        st.pyplot(fig_distance)

    st.write("En moyenne, on constate que les tirs réussis ont une distance plus proche du panier, autour de 8-9 feets (versus 15-16 feets pour les tirs ratés).")
    st.write("On constate que la plupart des tirs sont effecutés soit proche du panier, soit à 3 points.")

    st.markdown("---------------")



    # taux de réussite selon le temps
    st.subheader("Taux de réussite selon le temps et le quart-temps")

    @st.cache_data
    def plot_success_time(df):
        df["TIME_BIN"] = pd.cut(
            df["TIME_LEFT"],
            bins=[0, 60, 180, 360, 540, 720],
            labels=["0-1 min", "1-3 min", "3-6 min", "6-9 min", "9-12 min"]
        ).cat.reorder_categories(["9-12 min", "6-9 min", "3-6 min", "1-3 min", "0-1 min"], ordered=True)

        fig1, ax1 = plt.subplots()
        sns.barplot(x="TIME_BIN", y="Shot Made Flag", data=df, estimator="mean", ax=ax1, palette="coolwarm")
        ax1.set_title("Taux de réussite selon le temps restant")
        ax1.set_xlabel("Temps restant")
        ax1.set_ylabel("Taux de réussite moyen")

        fig2, ax2 = plt.subplots()
        sns.barplot(x="Period", y="Shot Made Flag", data=df, estimator="mean", ax=ax2, palette="coolwarm")
        ax2.set_title("Taux de réussite selon le quart-temps")

        return fig1, fig2

    fig_time1, fig_time2 = plot_success_time(df)
   


    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig_time1)

    with col2:
        st.pyplot(fig_time2)


    st.write("On constate qu'au fil du match, le taux de réussite au tir diminue progressivement.")

    st.markdown("---------------")



    # taux de réussite en fonction de l'age
    st.subheader("Taux de réussite en fonction de l’âge")

    @st.cache_data
    def plot_success_age(df):
        df_age22 = df[df["Age"] > 22]
        fig, ax = plt.subplots()
        sns.lineplot(x="Age", y="Shot Made Flag", data=df_age22, estimator="mean", ax=ax)
        ax.set_title("Taux de réussite moyen par âge")
        ax.set_xlabel("Âge du joueur")
        ax.set_ylabel("Taux de réussite moyen")
        return fig

    fig_age = plot_success_age(df)
    st.pyplot(fig_age)

    st.write("On constate que les jeunes joueurs NBA ont un taux de réussite qui augmente progressivement durant les 24 et 28 ans, probablement lié au temps d'adaptation à la NBA. Le taux de réussite se stabilise entre 28 et 32 ans d'expérience. A partir de 32 ans, le taux de réussite diminue progressivement")

    st.markdown("---------------")

    #variable catégorielle
    st.subheader("Analyse des variables catégorielles")

    # Action Type
    @st.cache_data
    def plot_action_type(df):
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="Action Type", y="Shot Made Flag", data=df, estimator="mean", ax=ax, palette="Paired")
        ax.set_title("Taux de réussite selon le type de tir")
        plt.xticks(rotation=45, ha="right")
        return fig
    fig_type = plot_action_type(df)
    st.pyplot(fig_type)


    st.write("On constate que les tirs proches du panier ont un taux de réussite moyen plus élevé (layup et dunk).")

    st.markdown("---------------")


    # is_home
    def plot_home_away(df):
        fig, ax = plt.subplots()
        sns.barplot(x="is_home", y="Shot Made Flag", data=df, estimator="mean", ax=ax)
        ax.set_xticklabels(["Extérieur", "Domicile"])
        ax.set_title("Taux de réussite à domicile vs extérieur")
        return fig

    fig_home = plot_home_away(df)
    st.pyplot(fig_home)

    home_mean = df.loc[df["is_home"]==1, "Shot Made Flag"].mean().round(3)
    away_mean = df.loc[df["is_home"]==0, "Shot Made Flag"].mean().round(3) 

    st.write(f"**Taux de réussite à domicile :** {home_mean}")
    st.write(f"**Taux de réussite à l’extérieur :** {away_mean}")
    st.write("On constate que le taux de réussite est légèrement meilleur lorsque le joueur joue à domicile.")



    st.markdown("---------------")

    # Zone de tir
    st.subheader("Zones de tir")

    @st.cache_data
    def plot_shot_zone(df):
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="Shot Zone Area", y="Shot Made Flag", data=df, estimator="mean", ax=ax, palette="Paired")
        ax.set_title("Taux de réussite selon la zone de tir")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return fig
    
    fig_zone = plot_shot_zone(df)
    st.pyplot(fig_zone)

    st.write("On constate que les tirs dans la zone centrale ont taux de réussite plus élevé.")
    center_mean = df.loc[df["Shot Zone Area"]=="Center(C)","Shot Made Flag"].mean().round(3)
    st.write(f"**Taux moyen de réussite depuis la zone centrale :** {center_mean}")


    st.markdown("---------------")


    # Evolution par saison
    st.subheader("Évolution des types de tirs par saison")

    @st.cache_data
    def plot_shot_type_season(df):
        shot_type_season = (
            df.groupby(["Year", "Action Type"])
            .size()
            .reset_index(name="count")
        )
        shot_type_season["proportion"] = shot_type_season.groupby("Year")["count"].transform(lambda x: x / x.sum())
        pivot_data = shot_type_season.pivot_table(
            index="Year", columns="Action Type", values="proportion", fill_value=0
        )

        fig, ax = plt.subplots(figsize=(12,6))
        pivot_data.plot(kind="bar", stacked=True, colormap="tab20", ax=ax)
        ax.set_title("Répartition des types de tir par saison (proportions)")
        ax.set_xlabel("Saison")
        ax.set_ylabel("Proportion des tirs")
        ax.legend(title="Type de tir", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        return fig

    fig_season = plot_shot_type_season(df)
    st.pyplot(fig_season)

if page == "Interprétation des résultats de la Régression Logistique":
    st.header("Interprétation des résultats de la Régression Logistique")


    st.markdown("""
    Cette page présente un exemple d'un modèle de Régression Logistique
    """)

    # Chargement du fichier de sauvegarde top joueur
    data = joblib.load("data/predictions_logit.pkl")

    X_test = data["X_test"]
    y_test = data["y_test"]
    y_pred_logit = data["y_pred_logit"]
    y_proba_logit = data["y_proba_logit"]
    params = data["params"]
    pipe_logit = data["pipe_logit"] 

    # chargements modele Curry 
    data_curry = joblib.load("data/predictions_logit_curry.pkl")

    X_test_curry = data_curry["X_test"]
    y_test_curry = data_curry["y_test"]
    y_pred_logit_curry = data_curry["y_pred_logit"]
    y_proba_logit_curry = data_curry["y_proba_logit"]
    params_curry = data_curry["params"]
    pipe_logit_curry = data_curry["pipe_logit"] 


    # Affichage des paramètres
    st.subheader("Paramètres du modèle de Régression Logistique")
    st.write("Les données ont été entraînées sur le top 25 des meilleurs joueurs.")
    st.dataframe(pd.DataFrame(params.items(), columns=["Paramètre", "Valeur"]))

    st.markdown("---")

    # Analyse sur les joueurs stars
    st.subheader("Analyse des performances sur les joueurs en activité")

    joueurs = [
        "LeBron James", "Stephen Curry", "Nikola Jokic", "Dwyane Wade",
        "Giannis Antetokounmpo", "James Harden", "Chris Paul", "Kawhi Leonard",
        "Anthony Davis", "Draymond Green", "Russell Westbrook", "Luka Doncic"
    ]

    # Filtrer uniquement les joueurs d'intérêt
    X_test_topplayer = X_test[X_test["Player"].isin(joueurs)].copy()

    if X_test_topplayer.empty:
        st.warning("Aucune donnée trouvée pour les joueurs sélectionnés dans le jeu de test.")
    else:
        # Moyenne par type d’action
        st.write("### Moyenne par type d’action de tir")
        mean_action = (
            X_test_topplayer.groupby("Action Type")[["y_test", "proba_logit"]]
            .mean()
            .round(3)
            .sort_values("proba_logit", ascending=False)
        )
        st.dataframe(mean_action)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=mean_action["proba_logit"],
            y=mean_action.index,
            palette="crest",
            ax=ax
        )
        ax.set_title("Probabilité moyenne de réussite selon le type d’action")
        ax.set_xlabel("Probabilité moyenne prédite")
        st.pyplot(fig)

        st.markdown("---")

        # Moyenne par zone de tir 
        st.write("### Moyenne par zone de tir")
        mean_zone = (
            X_test_topplayer.groupby("Shot Zone Area")[["y_test", "proba_logit"]]
            .mean()
            .round(3)
            .sort_values("proba_logit", ascending=False)
        )
        st.dataframe(mean_zone)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(
            x=mean_zone["proba_logit"],
            y=mean_zone.index,
            palette="viridis",
            ax=ax
        )
        ax.set_title("Probabilité moyenne de réussite selon la zone de tir")
        ax.set_xlabel("Probabilité moyenne prédite")
        st.pyplot(fig)

    st.markdown("---")

    # coefficient de la reg log
    
    coefs = pipe_logit.named_steps["clf"].coef_[0]
    features = pipe_logit.named_steps["pre"].get_feature_names_out()

    # Créer un DataFrame des coefficients
    df_coef = pd.DataFrame({
        "Variable": features,
        "Coefficient": coefs,
        "Odds_ratio": np.exp(coefs)
    })

    # Top 6 variables influençant positivement la probabilité
    st.write("#### Top 6 variables influençant positivement la probabilité de réussite")
    df_coef_pos = df_coef.sort_values(by="Odds_ratio", ascending=False).head(6)
    st.dataframe(df_coef_pos.style.format({"Coefficient": "{:.3f}", "Odds_ratio": "{:.3f}"}))

    # Top 8 variables influençant négativement la probabilité ---
    st.write("#### Top 8 variables influençant négativement la probabilité de réussite")
    df_coef_neg = df_coef[df_coef["Coefficient"] < 0].sort_values("Coefficient").head(8)
    st.dataframe(df_coef_neg.style.format({"Coefficient": "{:.3f}", "Odds_ratio": "{:.3f}"}))

    st.write("Les variables influençant positivement le tir sont les tirs proches et au centre du panier. " \
    "Et inversément, les variables influençant négativement le tir sont les tirs éloignés ou les tirs difficiles (Jump Shot). Il y a une exception pour la variable Layup.")


    # Modele Curry
    st.write("#### Comparaison avec un modèle focalisé uniquement sur Stephen Curry")
    # Renommer les colonnes pour Curry
    X_test_curry = X_test_curry.rename(columns={
        "y_test": "y_test_curry",
        "proba_logit": "proba_logit_curry"
    })

    # Probabilité moyenne par type d'action pour Curry
    mean_action_curry = (
        X_test_curry.groupby("Action Type")[["y_test_curry", "proba_logit_curry"]]
        .mean()
        .round(3)
    )

    # Probabilité moyenne par type d'action pour tous les joueurs
    mean_action_all = (
        X_test_topplayer.groupby("Action Type")[["proba_logit"]]
        .mean()
        .round(3)
        .rename(columns={"proba_logit": "proba_logit_AllPlayer"})
    )   

    # Fusionner les deux DataFrames pour affichage côte à côte
    mean_action_combined = mean_action_curry.join(mean_action_all, how="outer").sort_values("proba_logit_curry", ascending=False)

    st.dataframe(mean_action_combined)
    st.write("Entre les prédictions propres à Stephen Curry et celles sur la base de tous les joueurs, il y a un écart de 15% sur les tirs à 3 points.")

    st.markdown("---")


    ### variable explicative de Curry

    coefs_curry = pipe_logit_curry.named_steps["clf"].coef_[0]
    features_curry = pipe_logit_curry.named_steps["pre"].get_feature_names_out()

    # Créer un DataFrame des coefficients
    df_coef_curry = pd.DataFrame({
        "Variable": features_curry,
        "Coefficient": coefs_curry,
        "Odds_ratio": np.exp(coefs_curry)
    }) 

    # Top 6 variables influençant positivement la probabilité de Curry
    st.write("#### Top 6 variables influençant positivement la probabilité de réussite de Stephen Curry")
    df_coef_pos_curry = df_coef_curry.sort_values(by="Odds_ratio", ascending=False).head(6)
    st.dataframe(df_coef_pos_curry.style.format({"Coefficient": "{:.3f}", "Odds_ratio": "{:.3f}"}))

    st.write("Nous constatons des résultats intéressants : le modèle a pu capter que Stephen Curry est un spécialiste des tirs à long distance. " \
    "La variable explicative 'Distance' a une relation positive avec le taux de réussite du tir." \
    " Nous constatons également que 'TIME_LEFT' a une relation postive avec le tir, " \
    "ce qui laisse sugérer que Stephen Curry est un joueur capable de marquer dans un contexte stressant.")

