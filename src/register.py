
import argparse
import mlflow

def main(args):
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name("iris-exp")
    if exp is None:
        raise RuntimeError("Experiment iris-exp not found. Run train.py first.")

    # get latest run
    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise RuntimeError("No runs found.")

    run = runs[0]
    run_id = run.info.run_id
    metrics = run.data.metrics
    value = metrics.get(args.metric)
    if value is None:
        raise RuntimeError(f"Metric {args.metric} not found on latest run.")

    print(f"Latest run {run_id} has {args.metric}={value:.4f}")

    # Register or create a new version
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)
    print(f"Registered model version: name={args.model_name} version={mv.version}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--metric", default="f1", help="metric to check before registering")
    p.add_argument("--min-improve", type=float, default=0.0, help="(reserved) min delta to consider")
    p.add_argument("--model-name", default="iris_classifier")
    args = p.parse_args()
    main(args)
