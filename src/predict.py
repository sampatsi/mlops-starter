
import argparse
import mlflow
import mlflow.pyfunc
from schema import IrisInput


def main(args):
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    # get latest registered version in 'None' stage (local registry)
    # or use 'Production' if you promote stages later
    latest = client.get_latest_versions(args.model_name)
    if not latest:
        raise RuntimeError(
            "No registered versions found. Run register.py first."
        )
    # pick the highest version
    mv = sorted(latest, key=lambda v: int(v.version))[-1]
    model = mlflow.pyfunc.load_model(mv.source)

    # validate inputs
    request = IrisInput(
        sepal_length=args.sepal_length,
        sepal_width=args.sepal_width,
        petal_length=args.petal_length,
        petal_width=args.petal_width,
    )

    pred = model.predict(request.to_array())
    print(int(pred[0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="iris_classifier")
    parser.add_argument("--sepal_length", type=float, required=True)
    parser.add_argument("--sepal_width", type=float, required=True)
    parser.add_argument("--petal_length", type=float, required=True)
    parser.add_argument("--petal_width", type=float, required=True)
    args = parser.parse_args()
    main(args)
