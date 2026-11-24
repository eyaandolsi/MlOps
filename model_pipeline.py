"""
model_pipeline.py
-----------------
Pipeline ML modulaire : prédiction du traitement antidiabétique (diabetesMed).

Fonctions principales :
    prepare_data()   : charger et prétraiter les données
    train_model()    : entraîner un modèle
    evaluate_model() : évaluer les performances
    save_model()     : sauvegarder le modèle (joblib)
    load_model()     : charger un modèle sauvegardé
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ----------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------
RANDOM_STATE = 42

MEDICAMENTS = [
    "metformin",
    "repaglinide",
    "nateglinide",
    "chlorpropamide",
    "glimepiride",
    "acetohexamide",
    "glipizide",
    "glyburide",
    "tolbutamide",
    "pioglitazone",
    "rosiglitazone",
    "acarbose",
    "miglitol",
    "troglitazone",
    "tolazamide",
    "examide",
    "citoglipton",
    "insulin",
    "glyburide-metformin",
    "glipizide-metformin",
    "glimepiride-pioglitazone",
    "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# Colonnes supprimées : fuite, identifiants, colonnes trop vides ou
# postérieures à la cible, diagnostics et spécialité (simplification)
AUTRES_COLONNES_A_SUPPRIMER = [
    "change",
    "encounter_id",
    "patient_nbr",
    "weight",
    "payer_code",
    "readmitted",
    "medical_specialty",
    "diag_1",
    "diag_2",
    "diag_3",
]

# Patients décédés / en hospice
DISCHARGE_EXCLUS = [11, 13, 14, 19, 20, 21]

# Colonnes numériques codant des catégories
COLONNES_CATEGORIELLES_ID = [
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

MODELES_DISPONIBLES = ("logreg", "rf", "xgb")
NOMS_MODELES = {
    "logreg": "Régression logistique",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}


# ----------------------------------------------------------------------
# 1. Chargement et nettoyage
# ----------------------------------------------------------------------
def load_data(path):
    """Charge le fichier CSV (les '?' sont lus comme valeurs manquantes).

    Args:
        path (str): chemin vers diabetic_data.csv.

    Returns:
        pd.DataFrame: données brutes.
    """
    df = pd.read_csv(path, na_values="?", low_memory=False)
    if "diabetesMed" not in df.columns:
        raise ValueError("La colonne cible 'diabetesMed' est absente du fichier.")
    return df


def check_leakage(df):
    """Vérifie la fuite de données : diabetesMed == Yes <=> au moins un médicament != 'No'.

    Returns:
        float: part des lignes concordantes (1.0 = fuite exacte).
    """
    a_un_medicament = (df[MEDICAMENTS] != "No").any(axis=1)
    return (a_un_medicament == (df["diabetesMed"] == "Yes")).mean()


def clean_data(df):
    """Nettoie les lignes et supprime les colonnes inutiles ou fuyantes.

    - un seul séjour par patient
    - retrait des patients décédés / hospice
    - retrait du genre 'Unknown/Invalid'
    - suppression des médicaments (fuite) et autres colonnes

    Args:
        df (pd.DataFrame): données brutes.

    Returns:
        pd.DataFrame: données nettoyées (avec la colonne cible diabetesMed).
    """
    df = df.drop_duplicates(subset="patient_nbr")
    df = df[~df["discharge_disposition_id"].isin(DISCHARGE_EXCLUS)]
    df = df[df["gender"] != "Unknown/Invalid"]
    df = df.drop(columns=MEDICAMENTS + AUTRES_COLONNES_A_SUPPRIMER)
    return df


def preprocess(df):
    """Prétraite les variables et extrait la cible.

    Args:
        df (pd.DataFrame): données nettoyées (issues de clean_data).

    Returns:
        tuple: (df_prepare, y) avec y = 1 (traité) / 0 (non traité).
    """
    df = df.copy()
    y = (df["diabetesMed"] == "Yes").astype(int)
    df = df.drop(columns="diabetesMed")

    # tranche d'âge -> milieu de la tranche
    df["age"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(int) + 5
    df["race"] = df["race"].fillna("Inconnu")
    df["max_glu_serum"] = df["max_glu_serum"].fillna("Non mesure")
    df["A1Cresult"] = df["A1Cresult"].fillna("Non mesure")
    for c in COLONNES_CATEGORIELLES_ID:
        df[c] = df[c].astype(str)  # codes numériques -> catégories
    return df, y


def prepare_data(path, test_size=0.2, random_state=RANDOM_STATE):
    """Charge, nettoie, encode et sépare les données en train/test.

    Args:
        path (str): chemin vers diabetic_data.csv.
        test_size (float): part du jeu de test.
        random_state (int): graine aléatoire.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    df = clean_data(load_data(path))
    df, y = preprocess(df)
    X = pd.get_dummies(df, drop_first=True)  # texte -> colonnes 0/1
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# ----------------------------------------------------------------------
# 2. Modélisation
# ----------------------------------------------------------------------
def build_model(model_name, y_train=None):
    """Construit un modèle non entraîné.

    Args:
        model_name (str): 'logreg', 'rf' ou 'xgb'.
        y_train (pd.Series): nécessaire pour 'xgb' (calcul de scale_pos_weight).

    Returns:
        Estimateur scikit-learn compatible.
    """
    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    if model_name == "xgb":
        if y_train is None:
            raise ValueError("y_train est requis pour construire le modèle 'xgb'.")
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        )
    raise ValueError(f"Modèle inconnu : '{model_name}'. Choix possibles : {MODELES_DISPONIBLES}")


def train_model(X_train, y_train, model_name="xgb"):
    """Entraîne le modèle choisi.

    Args:
        X_train, y_train: données d'entraînement.
        model_name (str): 'logreg', 'rf' ou 'xgb'.

    Returns:
        Modèle entraîné.
    """
    model = build_model(model_name, y_train)
    model.fit(X_train, y_train)
    return model


# ----------------------------------------------------------------------
# 3. Évaluation
# ----------------------------------------------------------------------
def evaluate_model(model, X_test, y_test):
    """Calcule les métriques (la classe d'intérêt est « No » = non traité).

    Args:
        model: modèle entraîné.
        X_test, y_test: données de test.

    Returns:
        dict: Accuracy, Précision (No), Rappel (No), F1 (No), AUC.
    """
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]  # probabilité de la classe 1 (Yes)
    return {
        "Accuracy": accuracy_score(y_test, pred),
        "Précision (No)": precision_score(1 - y_test, 1 - pred),
        "Rappel (No)": recall_score(1 - y_test, 1 - pred),
        "F1 (No)": f1_score(1 - y_test, 1 - pred),
        "AUC": roc_auc_score(y_test, proba),
    }


def compare_models(models, X_test, y_test):
    """Évalue plusieurs modèles et renvoie un tableau comparatif.

    Args:
        models (dict): {nom: modèle entraîné}.

    Returns:
        pd.DataFrame: une ligne par modèle.
    """
    lignes = {nom: evaluate_model(m, X_test, y_test) for nom, m in models.items()}
    resultats = pd.DataFrame(lignes).T.round(3)
    resultats.index.name = "Modèle"
    return resultats


def plot_results(resultats, models, X_test, y_test, show=True):
    """Trace la comparaison des modèles et les matrices de confusion.

    Séparé de evaluate_model pour pouvoir évaluer sans affichage (CI/CD).
    """
    resultats.plot.bar(figsize=(9, 4), rot=0)
    plt.title("Comparaison des modèles")
    plt.ylim(0, 1)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    fig, ax = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.8))
    ax = [ax] if len(models) == 1 else ax
    for a, (nom, m) in zip(ax, models.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            m.predict(X_test),
            display_labels=["No", "Yes"],
            cmap="Blues",
            ax=a,
        )
        a.set_title(nom)
    plt.tight_layout()
    if show:
        plt.show()


# ----------------------------------------------------------------------
# 4. Sauvegarde / chargement
# ----------------------------------------------------------------------
def save_model(model, path="model.joblib", columns=None):
    """Sauvegarde le modèle avec joblib.

    Args:
        model: modèle entraîné.
        path (str): fichier de sortie.
        columns (list): colonnes de X_train (utile pour réaligner de nouvelles données).
    """
    joblib.dump({"model": model, "columns": list(columns) if columns is not None else None}, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="model.joblib"):
    """Charge un modèle sauvegardé.

    Returns:
        tuple: (model, columns).
    """
    contenu = joblib.load(path)
    return contenu["model"], contenu["columns"]


def align_columns(X_new, columns):
    """Réaligne de nouvelles données encodées sur les colonnes d'entraînement."""
    return X_new.reindex(columns=columns, fill_value=0)


# ----------------------------------------------------------------------
# 5. Profils des patients sans traitement (objectif métier)
# ----------------------------------------------------------------------
def profile_untreated(path):
    """Compare les patients sans traitement (No) et traités (Yes).

    Returns:
        tuple: (moyennes par groupe, répartition A1Cresult en %).
    """
    df, y = preprocess(clean_data(load_data(path)))
    groupe = y.map({0: "No (sans traitement)", 1: "Yes (traité)"})
    moyennes = df.groupby(groupe)[["age", "time_in_hospital", "num_medications", "number_diagnoses"]].mean().T.round(2)
    a1c = (pd.crosstab(df["A1Cresult"], groupe, normalize="columns") * 100).round(1)
    return moyennes, a1c
