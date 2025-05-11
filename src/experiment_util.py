# src/experiment_util.py (FOR CITI BIKE PROJECT)

import logging
import os
import mlflow
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def set_mlflow_tracking():
    """
    Set up MLflow tracking server credentials and URI for Citi Bike project.
    """
    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    logger.info(f"✅ MLflow tracking URI set to {uri}")

    return mlflow

def log_citibike_model_to_mlflow(
    model,
    input_data,
    experiment_name,
    metric_name="mae",
    model_name=None,
    params=None,
    score=None,
):
    """
    Log a trained Citi Bike model, its parameters, and evaluation metrics to MLflow.

    Parameters:
    - model: Trained model object (e.g., LightGBM, XGBoost).
    - input_data: Input DataFrame used for training (to infer model signature).
    - experiment_name: Name of the MLflow experiment (e.g., 'citibike-ride-prediction').
    - metric_name: Name of the primary evaluation metric to log (e.g., "mae", "rmse").
    - model_name: Name under which to register the model in MLflow Model Registry.
    - params: Optional dictionary of hyperparameters.
    - score: Optional evaluation score (float) to log.
    """
    try:
        # Set the experiment
        mlflow.set_experiment(experiment_name)
        logger.info(f"✅ MLflow experiment set to: {experiment_name}")

        # Start a new MLflow run
        with mlflow.start_run():
            # Log hyperparameters if any
            if params:
                mlflow.log_params(params)
                logger.info(f"📋 Logged parameters: {params}")

            # Log metric (score) if provided
            if score is not None:
                mlflow.log_metric(metric_name, score)
                logger.info(f"📈 Logged {metric_name}: {score}")

            # Infer the input/output signature automatically
            signature = infer_signature(input_data, model.predict(input_data))
            logger.info("✅ Model input/output signature inferred.")

            # Set default model name if not provided
            if not model_name:
                model_name = model.__class__.__name__

            # Log the trained model
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model_artifact",
                signature=signature,
                input_example=input_data,
                registered_model_name=model_name,
            )
            logger.info(f"🚴‍♂️ Citi Bike model logged as: {model_name}")

            return model_info

    except Exception as e:
        logger.error(f"❌ Error while logging model to MLflow: {e}")
        raise
