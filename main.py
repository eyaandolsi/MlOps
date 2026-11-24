"""
main.py
-------
Point d'entrée : exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --prepare
    python main.py --train --model xgb
    python main.py --evaluate --model xgb
    python main.py --profile
    python main.py --prepare --train --evaluate
    python main.py --all
    python main.py --all --model rf
"""

import argparse
import sys

from model_pipeline import (
    NOMS_MODELES,
    MODELES_DISPONIBLES,
    align_columns,
    evaluate_model,
    load_model,
    prepare_data,
    profile_untreated,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML : prédiction du traitement antidiabétique (diabetesMed)")
    parser.add_argument("--prepare", action="store_true", help="Charger et prétraiter les données")
    parser.add_argument("--train", action="store_true", help="Entraîner et sauvegarder le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer un modèle sauvegardé")
    parser.add_argument("--profile", action="store_true", help="Profils des patients sans traitement")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécuter toutes les étapes (prepare, train, evaluate, profile)",
    )
    parser.add_argument("--data", default="diabetic_data.csv", help="Chemin du fichier CSV")
    parser.add_argument(
        "--model",
        default="xgb",
        choices=MODELES_DISPONIBLES,
        help="Modèle à utiliser : logreg, rf ou xgb (défaut : xgb)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Fichier du modèle (défaut : model_<nom>.joblib)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --all active toutes les étapes (doit être avant le test "aucune action")
    if args.all:
        args.prepare = args.train = args.evaluate = args.profile = True

    model_path = args.model_path or f"model_{args.model}.joblib"

    if not (args.prepare or args.train or args.evaluate or args.profile):
        print("Aucune action demandée. Utilisez --prepare, --train, --evaluate, --profile ou --all (voir --help).")
        sys.exit(1)

    # Étape 1 : préparation
    if args.prepare:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print(f"Données préparées : X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"Part de patients traités (train) : {y_train.mean():.3f}")

    # Étape 2 : entraînement + sauvegarde
    if args.train:
        X_train, X_test, y_train, y_test = prepare_data(args.data)
        print(f"Entraînement : {NOMS_MODELES[args.model]}...")
        model = train_model(X_train, y_train, args.model)
        save_model(model, model_path, columns=X_train.columns)

    # Étape 3 : évaluation d'un modèle sauvegardé
    if args.evaluate:
        model, columns = load_model(model_path)
        _, X_test, _, y_test = prepare_data(args.data)  # même split (random_state=42)
        X_test = align_columns(X_test, columns)
        resultats = evaluate_model(model, X_test, y_test)
        print(f"Évaluation : {NOMS_MODELES[args.model]}")
        for nom, valeur in resultats.items():
            print(f"  {nom:<16} {valeur:.3f}")

    # Étape 4 : profils (objectif métier)
    if args.profile:
        moyennes, a1c = profile_untreated(args.data)
        print(moyennes)
        print()
        print(a1c)


if __name__ == "__main__":
    main()
