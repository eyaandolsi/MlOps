"""pipeline_prefect.py : orchestration Prefect du pipeline ML."""

import argparse
import os
import subprocess  # nosec B404
import sys

import pandas as pd
from prefect import flow, task, get_run_logger

from model_pipeline import (
    NOMS_MODELES,
    align_columns,
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

FICHIERS = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
LIGNE_MAX = "120"


def executer(cmd):
    """Lance une commande, journalise la sortie et échoue si le code retour != 0."""
    logger = get_run_logger()
    res = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603
    logger.info(res.stdout + res.stderr)
    if res.returncode != 0:
        raise RuntimeError(f"Échec : {' '.join(cmd)}")


# ---------------- TASKS : projet, environnement et code ----------------
@task(name="Récupérer le projet (git)")
def recuperer_projet(repo_url, dossier="projet"):
    if os.path.isdir(os.path.join(dossier, ".git")):
        executer(["git", "-C", dossier, "pull"])
    else:
        executer(["git", "clone", repo_url, dossier])


@task(name="Installer les dépendances")
def installer_dependances():
    executer(
        [
            sys.executable, "-m", "pip", "install",
            "--default-timeout=600", "--retries", "10",
            "-r", "requirements.txt",
        ]
    )


@task(name="Formatage du code (black)")
def formater_code():
    executer([sys.executable, "-m", "black", "-l", LIGNE_MAX, *FICHIERS])


@task(name="Qualité du code (flake8)")
def qualite_code():
    executer([sys.executable, "-m", "flake8", "--max-line-length", LIGNE_MAX, *FICHIERS])


@task(name="Sécurité du code (bandit)")
def securite_code():
    executer([sys.executable, "-m", "bandit", "-q", *FICHIERS])


@task(name="Tests unitaires (pytest)")
def tests_unitaires():
    executer([sys.executable, "-m", "pytest", "-q", "test_model_pipeline.py"])


# ---------------- TASKS : data / modèle ----------------
@task(name="Préparation des données")
def preparer_donnees(data):
    return prepare_data(data)


@task(name="Entraînement du modèle")
def entrainer_modele(X_train, y_train, model_name):
    return train_model(X_train, y_train, model_name)


@task(name="Sauvegarder le modèle")
def sauvegarder_modele(model, path, columns):
    save_model(model, path, columns=columns)


@task(name="Charger le modèle")
def charger_modele(path):
    return load_model(path)


@task(name="Évaluer le modèle")
def evaluer_modele(model, X_test, y_test):
    logger = get_run_logger()
    resultats = evaluate_model(model, X_test, y_test)
    for nom, valeur in resultats.items():
        logger.info(f"{nom:<16} {valeur:.3f}")
    return resultats


@task(name="Prédiction")
def predire(model, X, n=10):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    res = pd.DataFrame({"prediction": pred, "proba_traite": proba.round(3)}, index=X.index)
    return res.head(n)


# ---------------- FLOWS ----------------
@flow(name="install")
def install_flow():
    installer_dependances()


@flow(name="code")
def code_flow():
    formater_code()
    qualite_code()
    securite_code()
    tests_unitaires()


@flow(name="train")
def train_flow(data="diabetic_data.csv", model_name="xgb", model_path=None):
    model_path = model_path or f"model_{model_name}.joblib"
    X_train, X_test, y_train, y_test = preparer_donnees(data)
    model = entrainer_modele(X_train, y_train, model_name)
    sauvegarder_modele(model, model_path, X_train.columns)


@flow(name="evaluate")
def evaluate_flow(data="diabetic_data.csv", model_name="xgb", model_path=None):
    model_path = model_path or f"model_{model_name}.joblib"
    model, columns = charger_modele(model_path)
    _, X_test, _, y_test = preparer_donnees(data)
    X_test = align_columns(X_test, columns)
    return evaluer_modele(model, X_test, y_test)


@flow(name="predict")
def predict_flow(data="diabetic_data.csv", model_name="xgb", model_path=None, n=10):
    model_path = model_path or f"model_{model_name}.joblib"
    model, columns = charger_modele(model_path)
    _, X_test, _, _ = preparer_donnees(data)
    X_test = align_columns(X_test, columns)
    res = predire(model, X_test, n)
    get_run_logger().info("\n" + res.to_string())
    return res


@flow(name="all")
def all_flow(
    data="diabetic_data.csv",
    model_name="xgb",
    model_path=None,
    avec_install=False,
    repo_url=None,
):
    model_path = model_path or f"model_{model_name}.joblib"
    if repo_url:
        recuperer_projet(repo_url)  # 1re étape : récupération du projet
    if avec_install:
        installer_dependances()
    code_flow()  # étapes qualité AVANT la préparation des données
    X_train, X_test, y_train, y_test = preparer_donnees(data)
    model = entrainer_modele(X_train, y_train, model_name)
    sauvegarder_modele(model, model_path, X_train.columns)
    return evaluer_modele(model, X_test, y_test)


# ---------------- CLI ----------------
FLOWS = {
    "install": install_flow,
    "code": code_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "predict": predict_flow,
    "all": all_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline Prefect")
    parser.add_argument("--flow", required=True, choices=FLOWS)
    parser.add_argument("--data", default="diabetic_data.csv")
    parser.add_argument("--model", default="xgb", choices=list(NOMS_MODELES))
    parser.add_argument("--repo-url", default=None, help="URL git (flow all uniquement)")
    args = parser.parse_args()

    if args.flow in ("install", "code"):
        FLOWS[args.flow]()
    elif args.flow == "all":
        all_flow(data=args.data, model_name=args.model, repo_url=args.repo_url)
    else:
        FLOWS[args.flow](data=args.data, model_name=args.model)
