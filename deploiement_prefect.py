"""deploiement_prefect.py : déploie les flows avec planification quotidienne."""

from prefect import serve
from prefect.schedules import Cron

from pipeline_prefect import (
    all_flow,
    code_flow,
    evaluate_flow,
    install_flow,
    predict_flow,
    train_flow,
)

# Mets l'URL de ton repo GitHub ici, ou laisse None tant qu'il n'existe pas
REPO_URL = "https://github.com/eyaandolsi/MlOps.git"

if __name__ == "__main__":
    serve(
        all_flow.to_deployment(
            name="ml-pipeline-all",
            schedule=Cron("0 2 * * *", timezone="Africa/Tunis"),
            parameters={"repo_url": REPO_URL},
            tags=["full-pipeline", "mlops"],
        ),
        train_flow.to_deployment(name="ml-pipeline-train", tags=["training", "mlops"]),
        evaluate_flow.to_deployment(name="ml-pipeline-evaluate", tags=["evaluation", "mlops"]),
        predict_flow.to_deployment(name="ml-pipeline-predict", tags=["prediction", "mlops"]),
        code_flow.to_deployment(name="ml-pipeline-code", tags=["quality", "mlops"]),
        install_flow.to_deployment(name="ml-pipeline-install", tags=["setup", "mlops"]),
    )
