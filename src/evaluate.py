
import mlflow
import os
from sklearn.metrics import classification_report
from data import load_data
import mlflow.sklearn


def main():
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    # Get latest run for our experiment
    exp = client.get_experiment_by_name("iris-exp")
    if exp is None:
        raise RuntimeError(
            "Experiment iris-exp not found. Run train.py first."
        )
    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        raise RuntimeError("No runs found. Run train.py first.")

    run = runs[0]
    run_id = run.info.run_id

    # Load model artifact and evaluate on test set
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    X_train, X_test, y_train, y_test, target_names = load_data()
    preds = model.predict(X_test)

    report = classification_report(y_test, preds, target_names=target_names)
    print(report)

    # Create artifacts directory if it doesn't exist
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/classification_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
